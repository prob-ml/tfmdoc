import logging
import os
import time

from fastparquet import ParquetFile


class OptumProcess:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.start_time = None
        self.patient_info = None
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def read_patient_info(self, gender_code=False):
        if gender_code:
            cols = ["Patid", "Gdr_Cd", "Yrdob"]
        else:
            cols = ["Patid", "Yrdob"]

        self.start_time = time.time()
        self.patient_info = (
            ParquetFile(self.data_dir + "zip5_mbr.parquet")
            .to_pandas(cols)
            .drop_duplicates()
        )
        self.log_time("Read patient info")

    def get_optum_files(self, optum_type):
        files = os.listdir(self.data_dir)
        return tuple(
            file
            for file in files
            if file.startswith(optum_type) and file.endswith(".parquet")
        )

    def log_time(self, message):
        end_time = time.time()
        time_elapsed = (end_time - self.start_time) / 60
        self.logger.info(message)
        self.logger.info(f"{time_elapsed: .0f} m")
