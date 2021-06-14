import os

import numpy as np
import pandas as pd
from fastparquet import ParquetFile

# path to data: /nfs/turbo/lsa-regier/OPTUM2/


def claims_pipeline(data_dir, output_dir="preprocessed_files/"):
    all_files = os.listdir(data_dir)
    diags = [f for f in all_files if is_parquet_file(f)]
    patient_group_cache = {}
    for diag in diags:
        parquet_file = ParquetFile(data_dir + diag)
        dataframe = parquet_file.to_pandas(["Patid", "Icd_Flag", "Diag", "Fst_Dt"])
        dataframe = clean_diag_data(dataframe)
        collate_patient_records(dataframe, patient_group_cache)
    # store offsets to concatenate later
    patient_offsets = []
    # store records (dates and diagnosis codes)
    records = []
    # concatenate
    for group in patient_group_cache.items():
        combined_years = pd.concat(patient_group_cache[group])
        combined_years.sort_values(["patid", "date"], inplace=True)
        patient_offsets.append(combined_years.groupby("patid")["date"].count())
        records.append(combined_years[["date", "diag"]])

    output_dir = data_dir + output_dir
    # we could save the index of patient_offsets if we wanted a
    # patient id lookup table
    patient_offsets = np.concatenate(patient_offsets)
    records = np.concatenate(records)
    patient_offsets, code_lookup, records = compile_preprocess_files(
        patient_offsets, records, output_dir
    )
    np.save(output_dir + "patient_offsets", patient_offsets)
    np.save(output_dir + "diag_code_lookup", code_lookup)
    np.save(output_dir + "diag_records", records)


def clean_diag_data(dataframe):
    dataframe["Icd_Flag"] = dataframe["Icd_Flag"].str.decode("UTF-8")
    dataframe["Diag"] = dataframe["Diag"].str.decode("UTF-8")
    # make icd flag more readable
    dataframe["Icd_Flag"] = dataframe["Icd_Flag"].str.replace("9 ", "09")
    # drop records without a diagnostic code
    dataframe = dataframe.dropna(subset=["Diag"])
    # combine ICD format with the code into one field; strip whitespace
    dataframe["DiagId"] = dataframe["Icd_Flag"] + ":" + dataframe["Diag"]
    dataframe.drop(columns=["Icd_Flag", "Diag"], inplace=True)
    dataframe["DiagId"] = dataframe["DiagId"].str.strip()
    # split code into general category and specific condition
    dataframe["DiagId"] = dataframe["DiagId"].apply(split_icd_codes)
    dataframe["Patid"] = dataframe["Patid"].astype(int)
    return dataframe


def collate_patient_records(dataframe, cache, grouping_index=1):
    dataframe["patient_group"] = dataframe["Patid"].apply(lambda x: x % 10)
    for group in range(10):
        patient_logs = dataframe[dataframe["patient_group"] == group]
        # this is probably the coolest pandas method ever
        patient_logs = patient_logs.explode("DiagId")
        column_map = {"Patid": "patid", "Fst_Dt": "date", "DiagId": "diag"}
        patient_logs.rename(columns=column_map, inplace=True)
        if cache.get(group) is None:
            cache[group] = [patient_logs]
        else:
            cache[group].append(patient_logs)


def compile_preprocess_files(patient_offsets, records, output_dir):
    dates = records[:, 0].astype(int)
    # might be more efficient to hardcode in the offset
    # presumably, Jan 1st 2002 is the earliest date in the dataset
    dates = dates - dates.min()
    # assign each diag code a unique integer key
    codes = records[:, 1]
    code_lookup, indexed_codes = np.unique(codes, return_inverse=True)
    records = np.column_stack((dates, indexed_codes))
    # write out
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    return patient_offsets, code_lookup, records


def split_icd_codes(code):
    """
    For each record in a diagnoses file, split the ICD code into the first
    3 digits (category) and the whole code (specific diagnosis).
    """
    return (code[:6], code)


def is_parquet_file(file):
    return file.startswith("diag") and file.endswith(".parquet")
