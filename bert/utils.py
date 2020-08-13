import logging
import os
import csv

EMR_PATH = 'nfs/turbo/lsa-regier'
DATA_PATH = '/nfs/turbo/lsa-regier/emr-data'
VOCAB_PATH = os.path.join(DATA_PATH, 'vocabs')
ICD_PATH = os.path.join('~/emr/data/', 'ICD_9_10_d_v1.1.csv')

# ICD_CSV = csv.reader(open(ICD_PATH), delimiter='|')
# ICD_DICT = {}
# for code in open(ICD_CSV).readlines():
#     code=code.strip()
#     ICD_DICT[code.split("|")[1]] = code

def get_logger(job):
    """from github: ffjord"""
    cur_path = os.path.abspath(os.path.dirname(__file__))
    root_path = os.path.split(cur_path)[0]
    log_path = os.path.join(root_path, 'logs')
    make_dirs(log_path)

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


def make_dirs(*dirnames):
    for dirname in dirnames:
        if not os.path.exists(dirname):
            os.makedirs(dirname)

# def to_icd10(icd9):
#     raw_code = icd9
#     icd9_decimal = 
    