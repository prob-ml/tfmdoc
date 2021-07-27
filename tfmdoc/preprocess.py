import logging
import os

import numpy as np
import pandas as pd
from fastparquet import ParquetFile

from tfmdoc.chunk_iterator import chunks_of_patients

log = logging.getLogger(__name__)

OUTPUT_DIR = "preprocessed_files/"


def claims_pipeline(
    data_dir,
    disease_codes,
    length_range=(16, 512),
    year_range=(2002, 2018),
    n_processed=None,
    test=False,
):
    log.info("Began pipeline")
    owd = os.getcwd()
    os.chdir(data_dir)
    if test:
        diags = ["diag_toydata1.parquet", "diag_toydata2.parquet"]
    else:
        diags = [f"diag_{yyyy}.parquet" for yyyy in range(*year_range)]
    diags = [ParquetFile(f) for f in diags]

    disease_codes = [f"{code: <7}".encode("utf-8") for code in disease_codes]

    offsets, records, labels = extract_patient_info(
        diags, disease_codes, length_range, n_processed
    )

    save_output_files(offsets, records, labels)
    os.chdir(owd)

    log.info("Completed pipeline")


def save_output_files(offsets, records, labels):
    offsets = pd.concat(offsets)
    patient_ids = offsets.index.to_numpy()
    offsets = np.cumsum(offsets.to_numpy())
    records = pd.concat(records).to_numpy()
    patient_labels = pd.concat(labels).to_numpy()
    # assign each diag code a unique integer key
    # this might be computationally taxing
    code_lookup, indexed_records = np.unique(records, return_inverse=True)
    # make sure that zero does not map to a code
    code_lookup = np.insert(code_lookup, 0, "pad")
    indexed_records += 1

    # write out
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    np.save(OUTPUT_DIR + "patient_offsets", offsets)
    np.save(OUTPUT_DIR + "patient_ids", patient_ids)
    np.save(OUTPUT_DIR + "diag_code_lookup", code_lookup)
    np.save(OUTPUT_DIR + "diag_records", indexed_records)
    np.save(OUTPUT_DIR + "patient_labels", patient_labels)


def transform_patients_chunk(chunk, disease_codes):
    # clean data
    chunk.drop_duplicates(inplace=True)
    chunk.sort_values(["Patid", "Fst_Dt"], inplace=True)
    chunk.drop(columns="Fst_Dt", inplace=True)
    # identify positive diagnoses
    chunk["is_case"] = chunk["Diag"].isin(disease_codes)
    # incorporate icd codes into diag codes
    chunk["DiagId"] = chunk["Icd_Flag"] + b":" + chunk["Diag"]
    chunk.drop(columns=["Icd_Flag", "Diag"], inplace=True)
    # split up codes
    chunk["DiagId"] = chunk["DiagId"].apply(lambda x: (x[:5], x[:6], x))
    # be careful as this will change the column naming within the generator!
    chunk.rename(columns={"Patid": "patid", "DiagId": "diag"}, inplace=True)
    # data go boom!
    return chunk.explode("diag")


def extract_patient_info(diags, disease_codes, length_range, n_processed):
    records = []
    labels = []
    offsets = []

    n_patients = 0
    n_records = 0

    for chunk in chunks_of_patients(diags, ("Patid", "Icd_Flag", "Diag", "Fst_Dt")):
        # break out of loop if there's an empty chunk
        if chunk.empty:
            log.warning("Empty chunk returned!")
            continue
        chunk = transform_patients_chunk(chunk, disease_codes)
        # if a patient has any code associated with the disease, flag as a positive
        labeled_ids = chunk.groupby("patid")["is_case"].any().astype(int)
        # drop rows with the disease's diag code to prevent leakage
        chunk = chunk[chunk["is_case"] == False]
        counts = chunk.groupby("patid")["diag"].count().rename("count")
        # drop patients with too few or too many records
        # a relatively small number of patients might comprise a
        # huge portion of the dataset due to extra-long (1k+) diag sequences
        counts = counts[counts.between(*length_range)]
        offsets.append(counts)
        chunk = chunk.join(counts, on="patid", how="right")
        records.append(chunk["diag"])
        labeled_ids = labeled_ids.to_frame().join(counts, on="patid", how="right")
        labels.append(labeled_ids["is_case"])
        n_patients += len(counts)
        n_records += len(chunk)
        log.info(f"{n_patients :,} patient ids, {n_records :,} records processed")
        if n_processed is not None and n_patients > n_processed:
            # stop processing chunks if enough patient ids have been processed
            break

    return offsets, records, labels
