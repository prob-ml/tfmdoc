import logging
import os

import numpy as np
import pandas as pd
from fastparquet import ParquetFile

from tfmdoc.chunk_iterator import chunks_of_patients

log = logging.getLogger(__name__)


def claims_pipeline(
    data_dir,
    disease_codes,
    output_dir="preprocessed_files/",
    min_length=16,
    max_length=512,
    test=False,
):
    log.info("Began pipeline")
    if test:
        diags = ["diag_toydata1.parquet", "diag_toydata2.parquet"]
    else:
        diags = [f"diag_{yyyy}.parquet" for yyyy in range(2002, 2019)]
    diags = [ParquetFile(data_dir + f) for f in diags]
    disease_codes = [code.encode("utf-8") for code in disease_codes]
    records = []
    labels = []
    offsets = []

    for chunk in chunks_of_patients(diags, ("Patid", "Icd_Flag", "Diag", "Fst_Dt")):
        # break out of loop if there's an empty chunk
        if chunk.empty:
            break
        chunk = transform_patients_chunk(chunk, disease_codes)
        log.info("Cleaned, labeled, and transformed patient chunk")
        # if a patient has any code associated with the disease, flag as a positive
        labels.append(chunk.groupby("patid")["is_case"].any().astype(int))
        # drop rows with the disease's diag code to prevent leakage
        chunk = chunk[chunk["is_case"] == False]
        counts = chunk.groupby("patid")["diag"].count().rename("count")
        # drop patients with too few or too many records
        # a relatively small number of patients might comprise a
        # huge portion of the dataset due to extra-long (1k+) diag sequences
        n_patients = counts.shape[0]
        counts = counts[counts.between(min_length, max_length)]
        log.info(
            f"{n_patients - counts.shape[0]:,} patients dropped out of {n_patients:,}"
        )
        offsets.append(counts)
        chunk = chunk.join(counts, on="patid", how="right")
        records.append(chunk["diag"])
        log.info("Extracted patient labels, offsets and records")

    output_dir = data_dir + output_dir

    compile_output_files(offsets, records, labels, output_dir)

    log.info("Completed pipeline")


def compile_output_files(offsets, records, labels, output_dir):
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
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    np.save(output_dir + "patient_offsets", offsets)
    np.save(output_dir + "patient_ids", patient_ids)
    np.save(output_dir + "diag_code_lookup", code_lookup)
    np.save(output_dir + "diag_records", indexed_records)
    np.save(output_dir + "patient_labels", patient_labels)


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
