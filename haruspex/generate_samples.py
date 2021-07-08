import argparse
import logging
import os
import time
from math import floor

import numpy as np
import pandas as pd
from fastparquet import ParquetFile
from util import get_diags, get_labs, log_time

ALT_CODES = ("1742-6", "1743-4", "1744-2", "77144-4", "44785-4")
AST_CODES = ("1920-8", "27344-1", "30239-8", "44786-2")
PLT_CODES = ("778-1", "26515-7", "777-3")
CONFOUNDER_CODES = ("F10", "305", "5715", "070", "B16", "B18")
# look out for hepatitis C


def main():
    logging.basicConfig(
        filename="logs/generate_samples.log", encoding="utf-8", level=logging.INFO
    )
    parser = argparse.ArgumentParser(
        "Generate samples for cases, control, and at-risk NAFL groups"
    )
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--skip_labs", action="store_true")
    args = parser.parse_args()
    if args.test:
        data_dir = "/nfs/turbo/lsa-regier/OPTUM2/test_data/"
        pat_file = "zip5_mbr_test.parquet"
    else:
        data_dir = "/nfs/turbo/lsa-regier/OPTUM2/"
        pat_file = "zip5_mbr.parquet"

    start_time = time.time()
    if args.skip_labs:
        cohort2 = np.load(data_dir + "cohort2_patids.npy")
    else:
        patient_info = ParquetFile(data_dir + pat_file)
        patient_info = patient_info.to_pandas()
        patient_info = patient_info.drop_duplicates()
        log_time(start_time)
        n_patients = len(patient_info)
        logging.info(f"{n_patients:,} patients in entire data set")

        _, cohort2 = filter_lab_data(patient_info, data_dir, test=args.test)
        np.save(data_dir + "cohort2_patids", cohort2)

    filtered_ids = pd.Series(cohort2, name="Patid")
    filter_diag_data(data_dir, filtered_ids, test=args.test)
    logging.info("Total run time:")
    log_time(start_time)


def filter_lab_data(patient_info, data_dir, test=False):
    # criterion 1:
    # subset to adults with at least 1 observation of AST, ALT, platelets
    # criterion 2: at least 3 observations of AST, ALT, & PLT, no more than 200
    # AST/ALT values between 0-1000 u/l
    labs = get_labs(os.listdir(data_dir))
    if test:
        labs = labs[:3]

    criterion1_group = []
    lab_obs_counts = []

    for lab in labs:
        # obtain subset of lab dataframe
        # with observations of alt, ast, and plt on same date
        filtered = filter_lab(data_dir, lab, patient_info)
        criterion1_group.append(filtered["Patid"].unique())
        # drop rows with excess AST
        filtered = filtered[filtered["Rslt_Nbr_Ast"] <= 1000]
        filtered.drop(columns=["Loinc_Cd_Ast", "Rslt_Nbr_Ast"], inplace=True)
        # drop rows with excess ALT
        filtered = filtered[filtered["Rslt_Nbr_Alt"] <= 1000]
        filtered.drop(columns=["Loinc_Cd_Alt", "Rslt_Nbr_Alt"], inplace=True)

        # get count of valid observations per year
        lab_obs_counts.append(filtered.groupby("Patid")["Fst_Dt"].count())

    criterion1_unique = np.unique(np.concatenate(criterion1_group))
    logging.info(f"{criterion1_unique.shape[0]} unique ids for criterion 1")

    n_obs = pd.concat(lab_obs_counts)
    n_obs = n_obs.groupby(level=0).sum()
    n_obs = n_obs[n_obs.between(3, 200)]
    criterion2_unique = n_obs.index
    n_obs.to_csv(data_dir + "n_lab_obs.csv")

    logging.info(f"{criterion2_unique.shape[0]} unique ids for criterion 2")

    return criterion1_unique, criterion2_unique


def filter_diag_data(data_dir, filtered_ids, test):
    diags = get_diags(os.listdir(data_dir))
    if test:
        diags = diags[:3]
    diag_dfs = []
    for diag in diags:
        pf = ParquetFile(data_dir + diag)
        df = pf.to_pandas(["Patid", "Diag"])
        df["Diag"] = df["Diag"].str.decode("UTF-8")
        df = df.merge(filtered_ids, on="Patid", how="inner")
        diag_dfs.append(df)
    df_diag = pd.concat(diag_dfs)

    # condition 1: NASH code and 1 NAFLD code
    group1_ids = find_nash_ids(df_diag)

    # condition 2: no NAFLD codes
    group2_ids = find_healthy_ids(df_diag)
    df_diag.drop(columns="is_nafld", inplace=True)

    # condition 3: has NAFL code, no NASH codes
    group3_ids = find_nafl_ids(df_diag)
    df_diag.drop(columns="is_nash", inplace=True)

    logging.info(
        f"""
    Cohort size before removing confounders:\n
    {len(group1_ids) :,} cases units\n
    {len(group2_ids) :,} control units\n
    {len(group3_ids) :,} at-risk units"""
    )

    df_diag["is_conf"] = df_diag["Diag"].str.startswith(CONFOUNDER_CODES)
    not_confounded = ~df_diag.groupby("Patid")["is_conf"].any()
    no_cf_ids = not_confounded.index

    # drop patient ids with confounding conditions
    group1_ids = np.intersect1d(group1_ids, no_cf_ids, assume_unique=True)
    group2_ids = np.intersect1d(group2_ids, no_cf_ids, assume_unique=True)
    group3_ids = np.intersect1d(group3_ids, no_cf_ids, assume_unique=True)

    logging.info(
        f"""
    Cohort size after removing confounders:\n
    {len(group1_ids) :,} cases units\n
    {len(group2_ids) :,} control units\n
    {len(group3_ids) :,} at-risk units"""
    )

    np.save(data_dir + "nash_patids", group1_ids)
    np.save(data_dir + "healthy_patids", group2_ids)
    np.save(data_dir + "nafl_patids", group3_ids)


def filter_lab(data_dir, lab, patient_info):
    lab_pf = ParquetFile(data_dir + lab)
    lab_df = lab_pf.to_pandas(["Patid", "Fst_Dt", "Loinc_Cd", "Rslt_Nbr"])
    # join patient_info
    lab_df = lab_df.merge(patient_info, on="Patid", how="inner")
    # filter to just adults (18+)
    year = floor(lab_df["Fst_Dt"][0] / 365.25 + 1960)
    lab_df["Age"] = year - lab_df["Yrdob"]
    lab_df.drop(columns="Yrdob", inplace=True)
    lab_df = lab_df[lab_df["Age"] >= 18]
    lab_df.drop(columns="Age", inplace=True)
    lab_df["Loinc_Cd"] = lab_df["Loinc_Cd"].str.decode("UTF-8")
    # lab codes for ALT, AST, platelet count
    # are all three lab codes present on at least one date?
    alt = lab_df[lab_df["Loinc_Cd"].str.startswith(ALT_CODES)]
    ast = lab_df[lab_df["Loinc_Cd"].str.startswith(AST_CODES)]
    plt = lab_df[lab_df["Loinc_Cd"].str.startswith(PLT_CODES)].drop(
        columns=["Rslt_Nbr", "Loinc_Cd"]
    )
    filtered = alt.merge(ast, on=["Patid", "Fst_Dt"], suffixes=("_Ast", "_Alt"))
    return filtered.merge(plt, on=["Patid", "Fst_Dt"])


def find_nash_ids(df_diag):
    nafld_codes = {"5718   ", "5719   ", "K760   "}
    contains_nash = df_diag[df_diag["Diag"].str.startswith("K7581")]["Patid"].unique()
    contains_any_nafld = df_diag[df_diag["Diag"].isin(nafld_codes)]["Patid"].unique()
    return np.intersect1d(contains_nash, contains_any_nafld, assume_unique=True)


def find_healthy_ids(df_diag):
    all_nafld_codes = {"5718   ", "5719   ", "K760   ", "K7581  "}
    df_diag["is_nafld"] = df_diag["Diag"].isin(all_nafld_codes)
    no_nafld = ~df_diag.groupby("Patid")["is_nafld"].any()
    return no_nafld[no_nafld == True].index


def find_nafl_ids(df_diag):
    nash_codes = {"5718   ", "5719   ", "K7581  "}
    df_diag["is_nash"] = df_diag["Diag"].isin(nash_codes)
    contains_nafl = df_diag[df_diag["Diag"].str.startswith("K760")]["Patid"].unique()
    no_nash = ~df_diag.groupby("Patid")["is_nash"].any()
    no_nash = no_nash[no_nash == True]
    return np.intersect1d(contains_nafl, no_nash.index, assume_unique=True)


if __name__ == "__main__":
    main()
