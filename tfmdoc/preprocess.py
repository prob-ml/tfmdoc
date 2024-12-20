import logging
import pathlib
import uuid

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
        length_range=(16, 512),
        year_range=(2002, 2018),
        n=None,
        test=False,
        split_codes=True,
        output_name="test_etl",
        save_counts=False,
    ):
        """ETL Pipeline for Health Insurance Claims data. Combines diagnoses
        and lab results into an encoded record for each patient.
        Base class for pretraining pipeline.

        Args:
            data_dir (string): path to parquet files containing patient records
            output_dir (string): path to directory where output HDF5 files will be saved
            length_range (tuple, optional): Min/max length of individual record.
            year_range (tuple, optional): Min/max year of data to ingest
            n ([type], optional): Max number of patient ids to process
            test (bool, optional): Shorten pipeline to help with testing
            split_codes (bool, optional): If true, split off prefixes of ICD codes
        """  # noqa: RST301
        self.data_dir = data_dir
        self.length_range = length_range
        self.n = n
        self.patient_data = None
        self._output_name = output_name
        if split_codes:
            self._splitter = lambda x: (x[:6], x.rstrip()) if pd.notnull(x) else x
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
        # lab data
        if not test:
            self._parquets += [f"lab_{yyyy}.parquet" for yyyy in range(*year_range)]
            self._parq_cols += [
                ("Patid", "Fst_Dt", "Loinc_Cd", "Rslt_Nbr", "Hi_Nrml", "Low_Nrml")
                for _ in range(*year_range)
            ]
            # pharm data
            self._parquets += [f"pharm_{yyyy}.parquet" for yyyy in range(*year_range)]
            self._parq_cols += [
                ("Patid", "Fill_Dt", "Ahfsclss") for _ in range(*year_range)
            ]
        self._test = test
        self.datasets = [
            ("offsets", np.dtype("uint16")),
            ("records", h5py.special_dtype(vlen=str)),
            ("ids", np.dtype("float32")),
            ("visits", np.dtype("uint8")),
            ("ages", np.dtype("uint8")),
        ]
        self.write_demo = False
        self.save_counts = save_counts

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
            for name, dtype in self.datasets:
                f.create_dataset(name, (0,), dtype=dtype, maxshape=(None,))

            if self.write_demo:
                f.create_dataset(
                    "demo", (0, 2), dtype=np.dtype("float64"), maxshape=(None, 2)
                )
            self.process_chunks(f)
            if self.save_counts:
                unique, counts = np.unique(f["records"], return_counts=True)
                tag = str(uuid.uuid4())[:3]
                np.save(self._output_dir + f"code_counts/{tag}_unique", unique)
                np.save(self._output_dir + f"code_counts/{tag}_counts", counts)
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

    def diagnose(self, chunk):
        # placeholder
        pass

    def transform_patients_chunk(self, chunk):
        # summary:
        # cleans and sorts data
        # merges in lab data
        # splits ICD codes
        chunk.drop_duplicates(inplace=True)
        if not self._test:
            chunk = clean_pharm(chunk)
        # melt + merge back in i guess
        chunk.sort_values(["Patid", "Fst_Dt"], inplace=True)
        if not self._test:
            chunk = discretize_labs(chunk)
        # identify positive diagnoses
        self.diagnose(chunk)
        # incorporate icd codes into diag codes
        chunk["DiagId"] = chunk["Icd_Flag"] + b":" + chunk["Diag"]
        chunk.drop(columns=["Icd_Flag", "Diag"], inplace=True)
        # split up codes
        chunk["DiagId"] = chunk["DiagId"].apply(self._splitter)
        # be careful as this will change the column naming within the generator!
        chunk.rename(columns={"Patid": "patid", "DiagId": "diag"}, inplace=True)
        # data go boom!
        chunk = chunk.explode("diag")
        if not self._test:
            chunk["diag"].fillna(chunk["lab_code"], inplace=True)
            chunk.drop(columns="lab_code", inplace=True)

        # filter out invalid ages
        self.patient_data = self.patient_data[self.patient_data["Yrdob"] > 1900]
        # keep only patients for whom we have age info
        return chunk.merge(
            self.patient_data, left_on="patid", how="inner", right_on="Patid"
        )

    def pull_patient_info(self, chunk):
        counts = chunk.groupby("patid")["diag"].count().rename("count")
        # drop patients with too few or too many records
        # perhaps there's a clever way to truncate long sequences
        counts = counts[counts.between(*self.length_range)]
        chunk = chunk.join(counts, on="patid", how="right")
        visits = chunk.groupby("patid")["Fst_Dt"].transform(
            lambda x: pd.factorize(x)[0]
        )
        last_date = (chunk.groupby("patid")["Fst_Dt"].last() / 365.25) + 1960
        last_date.name = "last_date"
        chunk_info = {}
        chunk_info["offsets"] = counts.to_numpy()
        chunk_info["records"] = chunk["diag"].to_numpy().astype(bytes)
        chunk_info["ids"] = counts.index
        chunk_info["visits"] = visits.to_numpy()
        chunk_info["ages"] = (
            (chunk["Fst_Dt"] / 365.25 + 1960) - chunk["Yrdob"]
        ).astype(int)

        return chunk_info


#########


class DiagnosisPipeline(ClaimsPipeline):
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
        prediction_window=30,
        output_name="test_diagnosis_etl",
        mode="diagnosis",
        pad=7,
    ):
        """ETL Pipeline for Health Insurance Claims data. Combines diagnoses
        and lab results into an encoded record for each patient.
        Pipeline for fine-tuning prediction task.

        Args:
            data_dir (string): path to parquet files containing patient records
            output_dir (string): path to directory where output HDF5 files will be saved
            disease_codes (list): list of ICD codes indicating disease of interest
            length_range (tuple, optional): Min/max length of individual record.
            year_range (tuple, optional): Min/max year of data to ingest
            n ([type], optional): Max number of patient ids to process
            test (bool, optional): Shorten pipeline to help with testing
            split_codes (bool, optional): If true, split off prefixes of ICD codes

        Raises:
            ValueError: if an invalid task mode is provided
        """  # noqa: RST301
        super().__init__(
            data_dir,
            output_dir,
            length_range,
            year_range,
            n,
            test,
            split_codes,
            output_name,
        )
        self.disease_codes = disease_codes
        self._pred_window = prediction_window

        if mode not in {"diagnosis", "early_detection"}:
            raise ValueError("Invalid Preprocessing Mode")
        self._mode = mode
        self.datasets.append(("labels", np.dtype("uint8")))
        if self._mode == "early_detection":
            self.datasets += [
                ("diagnosed_dates", np.dtype("uint16")),
                ("dates", np.dtype("uint16")),
            ]
        self._pad = pad
        self.write_demo = True

    def diagnose(self, chunk):
        codes = [f"{code: <{self._pad}}".encode("utf-8") for code in self.disease_codes]
        chunk["is_case"] = chunk["Diag"].isin(codes)

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
        if self._mode == "diagnosis":
            chunk = chunk[chunk["Fst_Dt"] < chunk["first_diag"] - self._pred_window]
        counts = chunk.groupby("patid")["diag"].count().rename("count")
        # drop patients with too few or too many records
        # perhaps there's a clever way to truncate long sequences
        counts = counts[counts.between(*self.length_range)]
        chunk = chunk.join(counts, on="patid", how="right")
        if self._mode != "pretraining":
            labeled_ids = labeled_ids.to_frame().join(counts, on="patid", how="right")[
                "is_case"
            ]
        return sample_cases(labeled_ids, counts, chunk, self.patient_data, self._mode)


def sample_cases(labeled_ids, counts, chunk, patient_data, mode):
    if mode == "early_detection":
        labeled_ids = labeled_ids[labeled_ids == 1]
        counts = counts.loc[labeled_ids.index]
        chunk = chunk.set_index("patid").loc[labeled_ids.index]
        diag_dates = chunk.groupby(chunk.index)["first_diag"].first()
        patient_data = patient_data.join(labeled_ids, how="right", on="Patid")
    elif mode == "diagnosis":
        cases = labeled_ids[labeled_ids == 1]
        controls = labeled_ids[labeled_ids == 0]
        try:
            controls = controls.sample(len(cases), random_state=17)
        except ValueError:
            pass
        labeled_ids = pd.concat([controls, cases])
        counts = counts.loc[labeled_ids.index]
        chunk = chunk.set_index("patid").loc[labeled_ids.index]
        diag_dates = None
        patient_data = patient_data.join(labeled_ids, how="right", on="Patid")
    else:
        diag_dates = None
    visits = chunk.groupby("patid")["Fst_Dt"].transform(lambda x: pd.factorize(x)[0])
    last_date = (chunk.groupby("patid")["Fst_Dt"].last() / 365.25) + 1960
    last_date.name = "last_date"
    patient_data = patient_data.join(last_date, on="Patid", how="right")

    return collect_chunk_info(
        chunk, counts, labeled_ids, visits, patient_data, diag_dates, mode
    )


def collect_chunk_info(
    chunk, counts, labeled_ids, visits, patient_data, diag_date, mode
):
    patient_data["latest_age"] = patient_data["last_date"] - patient_data["Yrdob"]
    patient_data = patient_data.set_index("Patid")
    patient_data["is_fem"] = (patient_data["Gdr_Cd"] == b"F").astype(int)
    patient_data.drop(columns=["Yrdob", "last_date", "Gdr_Cd", "is_case"], inplace=True)
    chunk_info = {}
    chunk_info["offsets"] = counts.to_numpy()
    chunk_info["records"] = chunk["diag"].to_numpy().astype(bytes)
    chunk_info["ids"] = counts.index
    chunk_info["visits"] = visits.to_numpy()
    chunk_info["demo"] = patient_data.to_numpy()
    chunk_info["ages"] = ((chunk["Fst_Dt"] / 365.25 + 1960) - chunk["Yrdob"]).astype(
        int
    )
    chunk_info["labels"] = labeled_ids.to_numpy()

    if mode == "early_detection":
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


def clean_pharm(chunk):
    # six digit ahfsclss code. First two digits are a prefix
    pharm = chunk[["Patid", "Fill_Dt", "Ahfsclss"]]
    pharm.dropna(subset=["Fill_Dt"], inplace=True)
    chunk.dropna(subset=["Fst_Dt"], inplace=True)
    chunk.drop(columns=["Fill_Dt", "Ahfsclss"], inplace=True)
    pharm = pharm.rename(columns={"Fill_Dt": "Fst_Dt", "Ahfsclss": "Diag"})
    pharm["Icd_Flag"] = "rx "
    pharm["Icd_Flag"] = pharm["Icd_Flag"].str.encode("utf-8")
    return pd.concat([chunk, pharm])
