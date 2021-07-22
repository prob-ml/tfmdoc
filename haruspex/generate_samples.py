import time
from math import floor

import numpy as np
import pandas as pd
from fastparquet import ParquetFile

from diseases import find_ald_ids, find_nash_ids, find_no_nafld_ids
from haruspex.optum_process import OptumProcess

ALT_CODE = "1742-6 "
AST_CODE = "1920-8 "
PLT_CODE = "777-3  "

NALD_CONFOUNDER_CODES = ("F10", "305", "5715", "070", "B16", "B18", "K70")


class SampleGenerator(OptumProcess):
    def __init__(self, data_dir, skip_labs=False, disease="nald"):
        super().__init__(data_dir, disease)
        self._skip_labs = skip_labs
        self.filtered_ids = None

    def run(self):
        self.start_time = time.time()
        if self._skip_labs:
            cohort2 = np.load(self.data_dir + "cohort2_patids.npy")
        else:
            self.read_patient_info()
            self.logger.info(f"{len(self.patient_info):,} patients in entire data set")
            n_obs, cohort2 = self._process_labs()
            self.log_time("Processed lab data")
            np.save(self.data_dir + "cohort2_patids", cohort2)
            n_obs.to_csv(self.data_dir + "n_lab_obs.csv")
        self.filtered_ids = pd.Series(cohort2, name="Patid")
        cases, controls = self._filter_diag_data()
        self.log_time("Processed diag data")
        np.save(f"{self.data_dir}{self.disease}_case_patids", cases)
        np.save(f"{self.data_dir}{self.disease}_control_patids", controls)
        self.log_time("Completed sample generation.")

    def _process_labs(self):

        labs = self.get_optum_files("lab")

        criterion1_group = []
        lab_obs_counts = []

        for lab in labs:
            # obtain subset of lab dataframe
            # with observations of alt, ast, and plt on same date
            filtered = self._filter_lab(lab)
            criterion1_group.append(filtered["Patid"].unique())
            # drop rows with excess AST
            filtered = filtered[filtered["Rslt_Nbr_Ast"] <= 1000]
            filtered.drop(columns=["Loinc_Cd_Ast", "Rslt_Nbr_Ast"], inplace=True)
            # drop rows with excess ALT
            filtered = filtered[filtered["Rslt_Nbr_Alt"] <= 1000]
            filtered.drop(columns=["Loinc_Cd_Alt", "Rslt_Nbr_Alt"], inplace=True)
            # get count of valid observations per year
            lab_obs_counts.append(filtered.groupby("Patid")["Fst_Dt"].count())
            self.log_time(f"Processed {lab}")
        criterion1 = np.concatenate(criterion1_group)
        criterion1_unique = np.unique(criterion1)
        self.logger.info(f"{criterion1_unique.shape[0]} unique ids for criterion 1")
        n_obs = pd.concat(lab_obs_counts)
        n_obs = n_obs.groupby(level=0).sum()
        n_obs = n_obs[n_obs.between(3, 200)]
        criterion2_unique = n_obs.index

        self.logger.info(f"{criterion2_unique.shape[0]} unique ids for criterion 2")

        return n_obs, criterion2_unique

    def _filter_lab(self, lab):
        lab_pf = ParquetFile(self.data_dir + lab)
        lab_df = lab_pf.to_pandas(["Patid", "Fst_Dt", "Loinc_Cd", "Rslt_Nbr"])
        # join patient_info
        lab_df = lab_df.merge(self.patient_info, on="Patid", how="inner")
        # filter to just adults (18+)
        year = floor(lab_df["Fst_Dt"][0] / 365.25 + 1960)
        lab_df["Age"] = year - lab_df["Yrdob"]
        lab_df.drop(columns="Yrdob", inplace=True)
        # why is this slow?
        lab_df = lab_df[lab_df["Age"] >= 18]
        lab_df.drop(columns="Age", inplace=True)
        lab_df["Loinc_Cd"] = lab_df["Loinc_Cd"].str.decode("UTF-8")
        # lab codes for ALT, AST, platelet count
        # are all three lab codes present on at least one date?
        alt = lab_df[lab_df["Loinc_Cd"] == ALT_CODE]
        ast = lab_df[lab_df["Loinc_Cd"] == AST_CODE]
        plt = lab_df[lab_df["Loinc_Cd"] == PLT_CODE].drop(
            columns=["Rslt_Nbr", "Loinc_Cd"]
        )
        filtered = alt.merge(ast, on=["Patid", "Fst_Dt"], suffixes=("_Ast", "_Alt"))
        return filtered.merge(plt, on=["Patid", "Fst_Dt"])

    def _filter_diag_data(self):
        diags = self.get_optum_files("diag")
        diag_dfs = []
        for diag in diags:
            pf = ParquetFile(self.data_dir + diag)
            df = pf.to_pandas(["Patid", "Diag"])
            df["Diag"] = df["Diag"].str.decode("UTF-8")
            df = df.merge(self.filtered_ids, on="Patid", how="inner")
            diag_dfs.append(df)
        df_diag = pd.concat(diag_dfs)

        if self.disease == "ald":
            cohort1_ids = find_ald_ids(df_diag, find_cases=True)
            cohort2_ids = find_ald_ids(df_diag, find_cases=False)
            df_diag.drop(columns="is_ald", inplace=True)

        elif self.disease == "nald":
            # condition 1: NASH code and 1 NAFLD code
            cohort1_ids = find_nash_ids(df_diag)

            # condition 2: no NAFLD codes
            cohort2_ids = find_no_nafld_ids(df_diag)
            df_diag.drop(columns="is_nafld", inplace=True)

        self.logger.info(
            f"""
        Cohort size before removing confounders:\n
        {len(cohort1_ids) :,} cases units\n
        {len(cohort2_ids) :,} control units"""
        )

        # this step will be a bit slow
        if self.disease == "nald":
            df_diag["is_conf"] = df_diag["Diag"].str.startswith(NALD_CONFOUNDER_CODES)
            not_confounded = ~df_diag.groupby("Patid")["is_conf"].any()
            no_cf_ids = not_confounded.index

            # drop patient ids with confounding conditions
            cohort1_ids = np.intersect1d(cohort1_ids, no_cf_ids, assume_unique=True)
            cohort2_ids = np.intersect1d(cohort2_ids, no_cf_ids, assume_unique=True)

            self.logger.info(
                f"""
            Cohort size after removing confounders:\n
            {len(cohort1_ids) :,} cases units\n
            {len(cohort2_ids) :,} control units"""
            )

        return cohort1_ids, cohort2_ids
