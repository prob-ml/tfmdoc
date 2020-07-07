import logging
import os


def get_logger(log_path):
    """from github: ffjord"""
    makedirs(log_path)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    info_file_handler = logging.FileHandler(log_path, mode="a")
    info_file_handler.setLevel(logging.INFO)
    logger.addHandler(info_file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    return logger


def makedirs(*dirnames):
    for dirname in dirnames:
        if not os.path.exists(dirname):
            os.makedirs(dirname)

