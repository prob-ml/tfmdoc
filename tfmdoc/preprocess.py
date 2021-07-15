import os

import numpy as np
import pandas as pd
from fastparquet import ParquetFile

# path to data: /nfs/turbo/lsa-regier/OPTUM2/


def claims_pipeline(
    data_dir, output_dir="preprocessed_files/", min_length=16, max_length=512
):
    all_files = os.listdir(data_dir)
    diags = [f for f in all_files if is_parquet_file(f)]
    patient_group_cache = {}
    for diag in diags:
        parquet_file = ParquetFile(data_dir + diag)
        dataframe = parquet_file.to_pandas(["Patid", "Icd_Flag", "Diag", "Fst_Dt"])
        dataframe = clean_diag_data(dataframe)
        collate_patient_records(dataframe, patient_group_cache)

    records, patient_offsets = combine_years(
        patient_group_cache, min_length, max_length
    )

    output_dir = data_dir + output_dir

    compile_output_files(patient_offsets, records, output_dir)


def compile_output_files(patient_offsets, records, output_dir):
    patient_ids = np.concatenate([po.index for po in patient_offsets])
    patient_offsets = np.cumsum(np.concatenate(patient_offsets))
    records = np.concatenate(records)
    # assign each diag code a unique integer key
    code_lookup, indexed_records = np.unique(records, return_inverse=True)
    # make sure that zero does not map to a code
    code_lookup = np.insert(code_lookup, 0, "pad")
    indexed_records += 1

    # write out
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    np.save(output_dir + "patient_offsets", patient_offsets)
    np.save(output_dir + "patient_ids", patient_ids)
    np.save(output_dir + "diag_code_lookup", code_lookup)
    np.save(output_dir + "diag_records", indexed_records)


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


def collate_patient_records(dataframe, cache):
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


def combine_years(cache, min_length, max_length):
    # store offsets to concatenate later
    patient_offsets = []
    # store records (dates and diagnosis codes)
    records = []
    # concatenate
    for group in cache:
        combined_years = pd.concat(cache[group])
        combined_years.sort_values(["patid", "date"], inplace=True)
        combined_years.drop(columns="date", inplace=True)
        counts = combined_years.groupby("patid")["diag"].count().rename("count")
        # filter out sequences that are too long or short
        valid_offsets = counts[(counts >= min_length) & (counts <= max_length)]
        patient_offsets.append(valid_offsets)
        valid_records = combined_years.join(valid_offsets, on="patid", how="right")
        records.append(valid_records[["diag"]])

    return records, patient_offsets


def split_icd_codes(code):
    """
    For each record in a diagnoses file, split the ICD code into the first
    3 digits (category) and the whole code (specific diagnosis).
    """
    return (code[:5], code[:6], code)


def is_parquet_file(file):
    return file.startswith("diag") and file.endswith(".parquet")
