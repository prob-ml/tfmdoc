import logging
import os

import numpy as np
import pandas as pd
from fastparquet import ParquetFile

log = logging.getLogger(__name__)


def claims_pipeline(
    data_dir,
    disease_codes,
    output_dir="preprocessed_files/",
    min_length=16,
    max_length=512,
):
    log.info("Began pipeline")
    diags = [f"diag_{yyyy}.parquet" for yyyy in range(2002, 2019)]
    frames = []
    disease_codes = [code.encode("utf-8") for code in disease_codes]
    for diag in diags:
        parquet_file = ParquetFile(data_dir + diag)
        dataframe = parquet_file.to_pandas(["Patid", "Icd_Flag", "Diag", "Fst_Dt"])
        dataframe = clean_diag_data(dataframe)
        # apply labels
        dataframe["is_case"] = dataframe["Diag"].isin(disease_codes)
        log.info("identified ALD patients")
        log.info(f"Read {diag}")
        frames.append(dataframe)
    dataframe = pd.concat(frames)
    records, offsets, labels = get_patient_info(dataframe, min_length, max_length)
    log.info("Obtained patient info")

    output_dir = data_dir + output_dir

    compile_output_files(offsets, records, labels, output_dir)

    log.info("Completed pipeline")


def compile_output_files(patient_offsets, records, labels, output_dir):
    patient_ids = patient_offsets.index.to_numpy()
    patient_offsets = np.cumsum(patient_offsets)
    records = records.to_numpy()
    patient_labels = labels.to_numpy()
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
    np.save(output_dir + "patient_labels", patient_labels)


def clean_diag_data(dataframe):
    # drop records without a diagnostic code
    dataframe = dataframe.dropna(subset=["Diag"])
    dataframe.drop_duplicates(inplace=True)
    log.info("Caught null and duplicate entries")
    # combine ICD format with the code into one field
    dataframe["DiagId"] = dataframe["Icd_Flag"] + b":" + dataframe["Diag"]
    dataframe.drop(columns=["Icd_Flag", "Diag"], inplace=True)
    log.info("Combined ICD flag and diag code")
    # split code into general category and specific condition
    dataframe["DiagId"] = dataframe["DiagId"].apply(split_icd_codes)
    log.info("Split up ICD codes")
    return dataframe


def get_patient_info(dataframe, min_length, max_length):
    # we could read in the data sequentially by patient id
    column_map = {"Patid": "patid", "Fst_Dt": "date", "DiagId": "diag"}
    dataframe.rename(columns=column_map, inplace=True)
    dataframe = dataframe.explode("diag")
    dataframe.sort_values(["patid", "date"], inplace=True)
    dataframe.drop(columns="date", inplace=True)
    # if a patient has any code associated with the disease, flag as a positive
    labels = dataframe.groupby("patid")["is_case"].any().astype(int)
    # drop rows with the disease's diag code to prevent leakage
    dataframe = dataframe[dataframe["is_case"] == False]
    counts = dataframe.groupby("patid")["diag"].count().rename("count")
    # filter out sequences that are too long or short
    offsets = counts[(counts >= min_length) & (counts <= max_length)]
    valid_records = dataframe.join(offsets, on="patid", how="right")
    records = valid_records["diag"]

    return records, offsets, labels


def split_icd_codes(code):
    """
    For each record in a diagnoses file, split the ICD code into the first
    3 digits (category) and the whole code (specific diagnosis).
    """
    return (code[:5], code[:6], code)


def is_parquet_file(file):
    return file.startswith("diag") and file.endswith(".parquet")
