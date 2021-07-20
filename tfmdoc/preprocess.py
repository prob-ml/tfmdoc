import logging
import os

import numpy as np
import pandas as pd
from fastparquet import ParquetFile

# path to data: /nfs/turbo/lsa-regier/OPTUM2/

ALD_CODES = (
    "5711   ",
    "5712   ",
    "5713   ",
    "K7010  ",
    "K7011  ",
    "K7041  ",
    "K7030  ",
    "K702   ",
    "K700   ",
    "K709   ",
    "K7031  ",
    "K7040  ",
)

log = logging.getLogger(__name__)


ALD_CODES = (
    "5711   ",
    "5712   ",
    "5713   ",
    "K7010  ",
    "K7011  ",
    "K7041  ",
    "K7030  ",
    "K702   ",
    "K700   ",
    "K709   ",
    "K7031  ",
    "K7040  ",
)


def claims_pipeline(
    data_dir, output_dir="preprocessed_files/", min_length=16, max_length=512
):
    all_files = os.listdir(data_dir)
    diags = [f for f in all_files if is_parquet_file(f)]
    frames = []
    for diag in diags:
        parquet_file = ParquetFile(data_dir + diag)
        dataframe = parquet_file.to_pandas(["Patid", "Icd_Flag", "Diag", "Fst_Dt"])
        frames.append(clean_diag_data(dataframe))
        log.info(f"Read {diag}")
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
    dataframe["Icd_Flag"] = dataframe["Icd_Flag"].str.decode("UTF-8")
    dataframe["Diag"] = dataframe["Diag"].str.decode("UTF-8")
    # apply labels
    dataframe["is_case"] = dataframe["Diag"].isin(ALD_CODES)
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
