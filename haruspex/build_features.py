import numpy as np
import pandas as pd
from fastparquet import ParquetFile

from haruspex.optum_process import OptumProcess


class FeaturesBuilder(OptumProcess):
    def __init__(self, data_dir, skip_diags=False, disease="nald"):
        super().__init__(data_dir, disease)
        self._skip_diags = skip_diags
        self.features = None
        self.code_lookup = {"alt": "1742-6 ", "ast": "1920-8 ", "plt": "777-3  "}

    def run(self):
        if self._skip_diags:
            self.features = pd.read_csv(
                self.data_dir + f"{self.disease}_interm_diag_features.csv"
            )
        else:
            self.read_patient_info(gender_code=True)
            self.features = self._load_cohorts()
            self.features = self.patient_info.merge(
                self.features, on="Patid", how="right"
            )
            self.features = self._get_diabetes_status()
            self.log_time("Processed diagnosis data")
        self.features = self._get_temporal_summaries()
        self.log_time("Processed lab data")
        self.features.to_csv(
            self.data_dir + f"{self.disease}_features.csv", index=False
        )

    def _load_cohorts(self):
        cohorts = ("case", "control")
        cohort_ids = {}
        for cohort in cohorts:
            filename = f"{self.disease}_{cohort}_patids.npy"
            id_array = np.load(self.data_dir + filename)
            cohort_ids[cohort] = pd.Series(id_array, name="Patid").to_frame()

        # random subsample of controls group
        n_cases = len(cohort_ids["case"])
        cohort_ids["control"] = cohort_ids["control"].sample(
            n=n_cases + 200, random_state=74
        )

        cohort_ids["control"]["status"] = 0
        cohort_ids["case"]["status"] = 1

        features = pd.concat((cohort_ids["control"], cohort_ids["case"]))

        assert features["Patid"].duplicated().sum() == 0

        return features

    def _get_diabetes_status(self):
        diags = self.get_optum_files("diag")
        diabetes_ids = []
        for diag in diags:
            pf = ParquetFile(self.data_dir + diag)
            df = pf.to_pandas(["Patid", "Diag"])
            df["Diag"] = df["Diag"].str.decode("UTF-8")
            diabetes_ids.append(
                df[df["Diag"].str.startswith(("250", "E11"))]["Patid"].values
            )
        diabetes_ids = np.unique(np.concatenate(diabetes_ids))
        has_diabetes = pd.Series(data=1, index=diabetes_ids, name="Diabetes")
        features = self.features.join(has_diabetes, on="Patid", how="left")
        features["Diabetes"].fillna(0, inplace=True)

        features.to_csv(
            self.data_dir + f"{self.disease}_interm_diag_features.csv", index=False
        )

        return features

    def _get_temporal_summaries(self):
        # merge in number of observations
        # associated with each patient id
        n_obs = pd.read_csv(self.data_dir + "n_lab_obs.csv")
        n_obs.rename(columns={"Fst_Dt": "n_obs"}, inplace=True)
        features = self.features.merge(n_obs, on="Patid")
        labs = self.get_optum_files("lab")
        lab_dfs = []
        for lab in labs:
            pf = ParquetFile(self.data_dir + lab)
            df = pf.to_pandas(["Patid", "Fst_Dt", "Loinc_Cd", "Rslt_Nbr"])
            df["Loinc_Cd"] = df["Loinc_Cd"].str.decode("UTF-8")
            # this should remove unwanted rows
            # slower iterations, but an easier concatenation
            df = df.merge(features["Patid"], on="Patid", how="inner")
            lab_dfs.append(df)
        df_lab = pd.concat(lab_dfs)

        # get aggregate statistics for each type of test
        for test_type, code in self.code_lookup.items():
            test_subset = df_lab[df_lab["Loinc_Cd"] == code]
            test_subset.sort_values(["Patid", "Fst_Dt"], inplace=True)
            test_info = test_subset.groupby("Patid")["Rslt_Nbr"].agg(
                ["last", "max", "min", "mean"]
            )
            test_info = test_info.add_prefix(f"{test_type}_")
            features = features.join(test_info, on="Patid", how="left")

        # have to do ratio separately
        test_subset = df_lab[df_lab["Loinc_Cd"] == "1920-8 "].merge(
            df_lab[df_lab["Loinc_Cd"] == "1742-6 "],
            how="inner",
            on=["Patid", "Fst_Dt"],
            suffixes=("_ast", "_alt"),
        )
        test_subset["ratio"] = test_subset["Loinc_Cd_ast"] / test_subset["Loinc_Cd_alt"]
        test_info = (
            test_subset.groupby("Patid")["ratio"]
            .agg(["last", "max", "min", "mean"])
            .add_prefix("ratio_")
        )
        features = features.join(test_info, on="Patid", how="left")

        # get age in most recent lab, long. history
        bookends = df_lab.groupby("Patid")["Fst_Dt"].agg(["min", "max"])
        # convert back to years
        bookends = np.floor(bookends / 365.25 + 1960)
        bookends["long_history"] = bookends["max"] - bookends["min"]
        features = features.merge(bookends, on="Patid")
        features["latest_age"] = features["max"] - features["Yrdob"]

        features.drop(columns=["max", "min"], inplace=True)
        return features
