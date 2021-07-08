import logging
import time


def get_labs(all_files):
    return tuple(
        file
        for file in all_files
        if file.startswith("lab") and file.endswith(".parquet")
    )


def get_diags(all_files):
    return tuple(
        file
        for file in all_files
        if file.startswith("diag") and file.endswith(".parquet")
    )


def log_time(start_time):
    end_time = time.time()
    time_elapsed = (end_time - start_time) / 60
    logging.info(f"{time_elapsed: .0f} m")
