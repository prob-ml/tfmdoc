import logging
import os


def get_logger(job):
    """from github: ffjord"""
    log_path = '/home/liutianc/emr/logs'
    makedirs(log_path)

    log_path += '/' + job
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

