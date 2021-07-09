import logging
import os
import time


class OptumProcess:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.start_time = None
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

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
