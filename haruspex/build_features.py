import argparse
import logging
import os
import time

import numpy as np
import pandas as pd
from fastparquet import ParquetFile


def main():
    parser = argparse.ArgumentParser(
        "Build out features matrix for the NASH-prediction study cohort"
    )
    parser.add_argument("--skip_diags", action="store_true")
    args = parser.parse_args()
    data_dir = "/nfs/turbo/lsa-regier/OPTUM2/"

    log_filename = "logs/build_features.log"
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    logging.basicConfig(
        filename="logs/build_features.log", encoding="utf-8", level=logging.INFO
    )
    start_time = time.time()
    if args.skip_diags:
        features = pd.read_csv(data_dir + "interm_diag_features.csv")
    else:
        pat_file = "zip5_mbr.parquet"
        cohorts = ("nash", "healthy", "nafl")
        cohort_ids = {}
        for cohort in cohorts:
            filename = f"{cohort}_patids.npy"
            id_array = np.load(data_dir + filename)
            cohort_ids[cohort] = pd.Series(id_array, name="Patid").to_frame()

        # random subsample of controls group
        n_cases = len(cohort_ids["nash"])
        cohort_ids["healthy"] = cohort_ids["healthy"].sample(
            n=n_cases + 200, random_state=74
        )

        cohort_ids["healthy"]["status"] = 0
        cohort_ids["nash"]["status"] = 1
        cohort_ids["nafl"]["status"] = 2

        features = pd.concat(
            (cohort_ids["healthy"], cohort_ids["nash"], cohort_ids["nafl"])
        )

        assert features["Patid"].duplicated().sum() == 0

        patient_info = ParquetFile(data_dir + pat_file)
        patient_info = patient_info.to_pandas(["Patid", "Gdr_Cd", "Yrdob"])
        patient_info = patient_info.drop_duplicates()
        features = patient_info.merge(features, on="Patid", how="right")

        features = get_diabetes_status(data_dir, features)

    features = get_temporal_summaries(data_dir, features)

    logging.debug("Successfuly read patient info")
    logging.info("Total run time:")
    log_time(start_time)

    # write out data
    features.to_csv(data_dir + "nald_features.csv", index=False)


def get_temporal_summaries(data_dir, features):
    # merge in number of observations
    # associated with each patient id
    n_obs = pd.read_csv(data_dir + "n_lab_obs.csv")
    n_obs.rename(columns={"Fst_Dt": "n_obs"}, inplace=True)
    features = features.merge(n_obs, on="Patid")
    labs = get_labs(os.listdir(data_dir))
    lab_dfs = []
    for lab in labs:
        logging.debug(f"Reading {lab}")
        pf = ParquetFile(data_dir + lab)
        df = pf.to_pandas(["Patid", "Fst_Dt", "Loinc_Cd", "Rslt_Nbr"])
        df["Loinc_Cd"] = df["Loinc_Cd"].str.decode("UTF-8")
        # this should remove unwanted rows
        # slower iterations, but an easier concatenation
        df = df.merge(features["Patid"], on="Patid", how="inner")
        lab_dfs.append(df)
    df_lab = pd.concat(lab_dfs)

    code_lookup = {
        "alt": ("1742-6", "1743-4", "1744-2", "77144-4", "44785-4"),
        "ast": ("1920-8", "27344-1", "30239-8", "44786-2"),
        "plt": ("778-1", "26515-7", "777-3"),
        "ratio": ("1916-6"),
    }

    # in theory--we only need 1742-6, 1920-8, 777-3

    # get aggregate statistics for each type of test
    for test_type, codes in code_lookup.items():
        test_subset = df_lab[df_lab["Loinc_Cd"].str.startswith(codes, na="")]
        test_subset.sort_values(["Patid", "Fst_Dt"], inplace=True)
        test_info = test_subset.groupby("Patid")["Rslt_Nbr"].agg(
            ["last", "max", "min", "mean"]
        )
        test_info = test_info.add_prefix(f"{test_type}_")
        features = features.join(test_info, on="Patid", how="left")
        logging.debug(f"Incorporated info from {test_type}")
    # get age in most recent lab, long. history
    bookends = df_lab.groupby("Patid")["Fst_Dt"].agg(["min", "max"])
    # convert back to years
    bookends = np.floor(bookends / 365.25 + 1960)
    bookends["long_history"] = bookends["max"] - bookends["min"]
    features = features.merge(bookends, on="Patid")
    features["latest_age"] = features["max"] - features["Yrdob"]

    features.drop(columns=["max", "min"], inplace=True)

    return features


def get_diabetes_status(data_dir, features):
    diags = get_diags(os.listdir(data_dir))
    diabetes_ids = []
    for diag in diags:
        logging.debug(f"Reading {diag}")
        pf = ParquetFile(data_dir + diag)
        df = pf.to_pandas(["Patid", "Diag"])
        df["Diag"] = df["Diag"].str.decode("UTF-8")
        diabetes_ids.append(
            df[df["Diag"].str.startswith(("250", "E11"))]["Patid"].values
        )
    diabetes_ids = np.unique(np.concatenate(diabetes_ids))
    logging.debug("Pulled ids with diabetes diagnosis.")
    has_diabetes = pd.Series(data=1, index=diabetes_ids, name="Diabetes")
    features = features.join(has_diabetes, on="Patid", how="left")
    features["Diabetes"].fillna(0, inplace=True)
    logging.debug("Processed diabetes diagnoses")

    features.to_csv(data_dir + "interm_diag_features.csv", index=False)

    return features


def get_labs(all_files):
    return tuple(
        file
        for file in all_files
        if file.startswith("lab") and file.endswith(".parquet")
    )


def get_diags(all_files):
    return tuple(
        file
        for file in all_files
        if file.startswith("diag") and file.endswith(".parquet")
    )


def log_time(start_time):
    end_time = time.time()
    time_elapsed = (end_time - start_time) / 60
    logging.info(f"{time_elapsed: .0f} m")
