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
        disease,
        disease_codes,
        length_range=(16, 512),
        year_range=(2002, 2018),
        n=None,
        test=False,
        split_codes=True,
        include_labs=False,
        prediction_window=30,
        output_name="preprocessed",
        early_detection=False,
    ):
        """ETL Pipeline for Health Insurance Claims data. Combines diagnoses
        and lab results into an encoded record for each patient.
        Labels each patient as a case or control (e.g. healthy)
        for a given disease based on the presence of ICD code(s).

        Args:
            data_dir (string): path to parquet files containing patient records
            output_dir (string): path to directory where output HDF5 files will be saved
            disease_codes (list): list of ICD codes indicating disease of interest
            length_range (tuple, optional): Min/max length of individual record.
            year_range (tuple, optional): Min/max year of data to ingest
            n ([type], optional): Max number of patient ids to process
            test (bool, optional): Shorten pipeline to help with testing
            split_codes (bool, optional): If true, split off prefixes of ICD codes
            include_labs (bool, optional): If true, include discretized lab results
        """  # noqa: RST301
        self.data_dir = data_dir
        self._disease = disease
        self.disease_codes = disease_codes
        self.length_range = length_range
        self.n = n
        self._include_labs = include_labs
        self.patient_data = None
        self._pred_window = prediction_window
        self._output_name = output_name
        self._early_detection = early_detection
        if split_codes:
            self._splitter = (
                lambda x: (x[:5], x[:6], x.rstrip()) if pd.notnull(x) else x
            )
        else:
            self._splitter = lambda x: x
        self._output_dir = output_dir
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

        pathlib.Path(self._output_dir).mkdir(exist_ok=True)
        with h5py.File(self._output_dir + self._output_name + ".hdf5", "w") as f:
            datasets = [
                ("offsets", np.dtype("uint16")),
                ("records", h5py.special_dtype(vlen=str)),
                ("labels", np.dtype("uint8")),
                ("ids", np.dtype("float64")),
                ("visits", np.dtype("uint8")),
            ]
            if self._early_detection:
                datasets += [
                    ("diagnosed_dates", np.dtype("uint16")),
                    ("dates", np.dtype("uint16")),
                ]
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
            # break loop for an empty processed chunk
            if not chunk_info["offsets"].size:
                continue
            n_patients += len(chunk_info["offsets"])
            n_records += len(chunk_info["records"])
            log.info(
                f"Wrote data for {n_patients :,} patient ids, {n_records :,} records"
            )
            for name, arr in chunk_info.items():
                # write out to an h5py object in chunks
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
        # summary:
        # cleans and sorts data
        # merges in lab data
        # splits ICD codes
        chunk.drop_duplicates(inplace=True)
        chunk.sort_values(["Patid", "Fst_Dt"], inplace=True)
        if self._include_labs:
            chunk = discretize_labs(chunk)
        # identify positive diagnoses
        if self._disease == "heart_failure":
            code = self.disease_codes[0].encode("utf-8")

            def starts_with(x):
                try:
                    return x.startswith(code)
                except AttributeError:
                    return False

            chunk["is_case"] = chunk["Diag"].apply(starts_with)
        elif self._disease == "ald":
            codes = [f"{code: <7}".encode("utf-8") for code in self.disease_codes]
            chunk["is_case"] = chunk["Diag"].isin(codes)
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
        # summary:
        # label data
        # mask records up to some window before first diagnosis
        # drop patients with too few records
        # if a patient has any code associated with the disease, flag as a positive
        labeled_ids = chunk.groupby("patid")["is_case"].any().astype(int)
        diagnosis_dates = (
            chunk[chunk["is_case"] == True].groupby("patid")["Fst_Dt"].first()
        )
        diagnosis_dates.name = "first_diag"
        chunk = chunk.join(diagnosis_dates, on="patid", how="left")
        chunk["first_diag"] = chunk["first_diag"].fillna(50000)
        # drop patient records after first diagnosis
        if not self._early_detection:
            chunk = chunk[chunk["Fst_Dt"] < chunk["first_diag"] - self._pred_window]
        counts = chunk.groupby("patid")["diag"].count().rename("count")
        # drop patients with too few or too many records
        counts = counts[counts.between(*self.length_range)]
        chunk = chunk.join(counts, on="patid", how="right")
        labeled_ids = labeled_ids.to_frame().join(counts, on="patid", how="right")[
            "is_case"
        ]
        return sample_cases(
            labeled_ids, counts, chunk, self.patient_data, self._early_detection
        )


def sample_cases(labeled_ids, counts, chunk, patient_data, early_detection):
    if early_detection:
        labeled_ids = labeled_ids[labeled_ids == 1]
        counts = counts.loc[labeled_ids.index]
        chunk = chunk.set_index("patid").loc[labeled_ids.index]
        diag_dates = chunk.groupby(chunk.index)["first_diag"].first()
    else:
        # downsample to a 19-1 control/case ratio
        cases = labeled_ids[labeled_ids == 1]
        controls = labeled_ids[labeled_ids == 0]
        try:
            controls = controls.sample(19 * len(cases))
        except ValueError:
            pass
        labeled_ids = pd.concat([controls, cases])
        counts = counts.loc[labeled_ids.index]
        chunk = chunk.set_index("patid").loc[labeled_ids.index]
        diag_dates = None
    patient_data = patient_data.join(labeled_ids, how="right", on="Patid")
    visits = chunk.groupby("patid")["Fst_Dt"].transform(lambda x: pd.factorize(x)[0])
    last_date = (chunk.groupby("patid")["Fst_Dt"].last() / 365.25) + 1960
    last_date.name = "last_date"
    patient_data = patient_data.join(last_date, on="Patid")

    return collect_chunk_info(
        chunk, counts, labeled_ids, visits, patient_data, diag_dates
    )


def collect_chunk_info(chunk, counts, labeled_ids, visits, patient_data, diag_date):
    patient_data["latest_age"] = patient_data["last_date"] - patient_data["Yrdob"]
    patient_data = patient_data.set_index("Patid")
    patient_data["is_fem"] = (patient_data["Gdr_Cd"] == b"F").astype(int)
    patient_data.drop(columns=["Yrdob", "last_date", "Gdr_Cd", "is_case"], inplace=True)
    chunk_info = {}
    chunk_info["offsets"] = counts.to_numpy()
    chunk_info["records"] = chunk["diag"].to_numpy().astype(bytes)
    chunk_info["labels"] = labeled_ids.to_numpy()
    chunk_info["ids"] = counts.index
    chunk_info["visits"] = visits.to_numpy()
    chunk_info["demo"] = patient_data.to_numpy()

    if diag_date is not None:
        chunk_info["diagnosed_dates"] = diag_date.to_numpy()
        chunk_info["dates"] = chunk["Fst_Dt"].to_numpy()

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
