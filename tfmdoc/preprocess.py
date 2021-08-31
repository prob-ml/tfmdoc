import logging
import pathlib

import h5py
import numpy as np
import pandas as pd
from fastparquet import ParquetFile

from tfmdoc.chunk_iterator import chunks_of_patients

log = logging.getLogger(__name__)


class ClaimsPipeline:
    def __init__(
        self,
        data_dir,
        output_dir,
        disease_codes,
        length_range=(16, 512),
        year_range=(2002, 2018),
        n=None,
        test=False,
        split_codes=True,
        include_labs=False,
    ):
        self.data_dir = data_dir
        self.disease_codes = [f"{code: <7}".encode("utf-8") for code in disease_codes]
        self.length_range = length_range
        self.n = n
        self._include_labs = include_labs
        self.patient_data = None
        if split_codes:
            self._splitter = (
                lambda x: (x[:5], x[:6], x.rstrip()) if pd.notnull(x) else x
            )
        else:
            self._splitter = lambda x: x
        self.output_dir = output_dir
        if test:
            self._parquets = ["diag_toydata1.parquet", "diag_toydata2.parquet"]
        else:
            self._parquets = [f"diag_{yyyy}.parquet" for yyyy in range(*year_range)]
        self._parq_cols = [
            ("Patid", "Fst_Dt", "Icd_Flag", "Diag") for _ in range(*year_range)
        ]
        if self._include_labs:
            self._parquets += [f"lab_{yyyy}.parquet" for yyyy in range(*year_range)]
            self._parq_cols += [
                ("Patid", "Fst_Dt", "Loinc_Cd", "Rslt_Nbr", "Hi_Nrml", "Low_Nrml")
                for _ in range(*year_range)
            ]

    def run(self):
        log.info("Began pipeline. Reading patient data...")
        self.patient_data = (
            ParquetFile(self.data_dir + "zip5_mbr.parquet")
            .to_pandas(["Patid", "Gdr_Cd", "Yrdob"])
            .drop_duplicates()
        )
        log.info("Read patient data")
        self._parquets = [ParquetFile(self.data_dir + f) for f in self._parquets]

        pathlib.Path(self.output_dir).mkdir(exist_ok=True)
        with h5py.File(self.output_dir + "preprocessed.hdf5", "w") as f:
            datasets = (
                ("offsets", np.dtype("uint16")),
                ("records", h5py.special_dtype(vlen=str)),
                ("labels", np.dtype("uint8")),
                ("ids", np.dtype("float64")),
            )
            for name, dtype in datasets:
                f.create_dataset(name, (0,), dtype=dtype, maxshape=(None,))

            f.create_dataset(
                "demo", (0, 2), dtype=np.dtype("float64"), maxshape=(None, 2)
            )

            self.process_chunks(f)

            records = f["records"]
            # assign each diag code a unique integer key
            code_lookup, indexed_records = np.unique(records, return_inverse=True)
            # make sure that zero does not map to a code
            code_lookup = np.insert(code_lookup, 0, "pad")
            indexed_records += 1
            f.create_dataset("tokens", data=indexed_records)
            f.create_dataset("diag_code_lookup", data=code_lookup)

        log.info("Completed pipeline")

    def process_chunks(self, file):
        n_patients = 0
        n_records = 0
        n_chunks = 0
        for chunk in chunks_of_patients(self._parquets, self._parq_cols):
            # break out of loop if there's an empty chunk
            if chunk.empty:
                log.warning("Empty chunk returned!")
                continue
            chunk = self.transform_patients_chunk(chunk)
            chunk_info = self.pull_patient_info(chunk)
            if not chunk_info["offsets"].size:
                continue
            n_patients += len(chunk_info["offsets"])
            n_records += len(chunk_info["records"])
            log.info(
                f"Wrote data for {n_patients :,} patient ids, {n_records :,} records"
            )
            for name, arr in chunk_info.items():
                if name == "demo":
                    file[name].resize((len(file[name]) + len(arr), 2))
                else:
                    file[name].resize(len(file[name]) + len(arr), axis=0)
                # fmt: off
                file[name][-len(arr) :, ] = arr
                # fmt: on
            log.info(f"Patient info written for chunk {n_chunks}")

            n_chunks += 1
            if self.n is not None and n_patients >= self.n:
                # stop processing chunks if enough patient ids have been processed
                break

    def transform_patients_chunk(self, chunk):
        # clean data
        chunk.drop_duplicates(inplace=True)
        chunk.sort_values(["Patid", "Fst_Dt"], inplace=True)
        if self._include_labs:
            chunk = discretize_labs(chunk)
        # identify positive diagnoses
        chunk["is_case"] = chunk["Diag"].isin(self.disease_codes)
        # incorporate icd codes into diag codes
        chunk["DiagId"] = chunk["Icd_Flag"] + b":" + chunk["Diag"]
        chunk.drop(columns=["Icd_Flag", "Diag"], inplace=True)
        # split up codes
        chunk["DiagId"] = chunk["DiagId"].apply(self._splitter)
        # be careful as this will change the column naming within the generator!
        chunk.rename(columns={"Patid": "patid", "DiagId": "diag"}, inplace=True)
        # data go boom!
        chunk = chunk.explode("diag")
        if self._include_labs:
            chunk["diag"].fillna(chunk["lab_code"], inplace=True)
            chunk.drop(columns="lab_code", inplace=True)
        return chunk

    def pull_patient_info(self, chunk):
        # if a patient has any code associated with the disease, flag as a positive
        labeled_ids = chunk.groupby("patid")["is_case"].any().astype(int)
        diagnosis_dates = (
            chunk[chunk["is_case"] == True].groupby("patid")["Fst_Dt"].first()
        )
        diagnosis_dates.name = "first_diag"
        chunk = chunk.join(diagnosis_dates, on="patid", how="left")
        chunk["first_diag"] = chunk["first_diag"].fillna(50000)
        # drop patient records after first diagnosis (and up to 1 month before)
        chunk = chunk[chunk["Fst_Dt"] < chunk["first_diag"] - 30]
        counts = chunk.groupby("patid")["diag"].count().rename("count")
        # drop patients with too few or too many records
        counts = counts[counts.between(*self.length_range)]
        chunk = chunk.join(counts, on="patid", how="right")
        labeled_ids = labeled_ids.to_frame().join(counts, on="patid", how="right")[
            "is_case"
        ]

        return sample_chunk_info(labeled_ids, counts, chunk, self.patient_data)


def sample_chunk_info(labeled_ids, counts, chunk, patient_data):
    cases = labeled_ids[labeled_ids == 1]
    controls = labeled_ids[labeled_ids == 0]
    try:
        controls = controls.sample(19 * len(cases))
    except ValueError:
        pass
    labeled_ids = pd.concat([controls, cases])
    counts = counts.loc[labeled_ids.index]
    chunk = chunk.set_index("patid").loc[labeled_ids.index]
    patient_data = patient_data.join(labeled_ids, how="right", on="Patid")
    last_date = (chunk.groupby("patid")["Fst_Dt"].last() / 365.25) + 1960
    last_date.name = "last_date"
    patient_data = patient_data.join(last_date, on="Patid")
    patient_data["latest_age"] = patient_data["last_date"] - patient_data["Yrdob"]
    # standard normalize
    patient_data["latest_age"] = (
        patient_data["latest_age"] - patient_data["latest_age"].mean()
    ) / patient_data["latest_age"].std()
    patient_data = patient_data.set_index("Patid").loc[labeled_ids.index]
    patient_data["is_fem"] = (patient_data["Gdr_Cd"] == b"F").astype(int)
    patient_data.drop(columns=["Yrdob", "last_date", "Gdr_Cd", "is_case"], inplace=True)

    chunk_info = {}

    chunk_info["offsets"] = counts.to_numpy()
    chunk_info["records"] = chunk["diag"].to_numpy().astype(bytes)
    chunk_info["labels"] = labeled_ids.to_numpy()
    chunk_info["ids"] = counts.index
    chunk_info["demo"] = patient_data.to_numpy()

    return chunk_info


def discretize_labs(chunk):
    chunk["Loinc_Cd"].replace(b"       ", b"unknown_lab", inplace=True)
    chunk["lab_status"] = chunk["Loinc_Cd"].mask(chunk["Loinc_Cd"].notna(), b"_normal")
    chunk["lab_status"] = chunk["lab_status"].mask(
        (chunk["Rslt_Nbr"] > chunk["Hi_Nrml"])
        | (chunk["Rslt_Nbr"] < chunk["Low_Nrml"]),
        b"_abnormal",
    )
    chunk["lab_code"] = chunk["Loinc_Cd"] + chunk["lab_status"]
    return chunk.drop(
        columns=["Loinc_Cd", "Rslt_Nbr", "lab_status", "Hi_Nrml", "Low_Nrml"]
    )
