import pandas as pd
import numpy as np

from pathlib import Path
import os
import shutil
import logging
import argparse
import re

from utils import get_logger, makedirs


def create_field_seq():
    
    logger.info('*' * 100)
    logger.info('Start: create_field_seq.')

#     for file in diags:
#         diag = pd.read_csv(file, sep=',', dtype = {'Patid': str})
# #         diag = diag.assign(DiagId = 'icd:' + diag['Icd_Flag'].astype(str) + '_loc:' + diag['Loc_cd'].astype(str) + '_diag:' + diag['Diag'])
#         diag = diag.assign(DiagId = 'icd:' + diag['Icd_Flag'].astype(str) + '_diag:' + diag['Diag'])
#         diag = diag.assign(PatGroup = diag['Patid'].apply(lambda x: x[-1]))
    
#         year = re.findall(pattern, file)[0]
#         for group in user_group:
#             logger.info(f'Start: {file}, group: {group}.')
#             sub_diag = diag[diag['PatGroup'] == group]
#             sub_diag_merged = sub_diag.groupby(['Patid', 'Fst_Dt'])['DiagId'].apply(lambda x: ' '.join(x))
#             sub_diag_merged_df = sub_diag_merged.to_frame().reset_index()
            
#             sub_diag_merged_df.rename(columns={'Patid': 'patid', 'Fst_Dt': 'date', 'DiagId': 'diags'}, inplace=True)
            
#             to_write = os.path.join(result_path, f'diag_{year}_{group}.csv')
#             # if os.path.exists(to_write):
#             #     sub_diag_merged_df.to_csv(to_write, mode='a', header=False, index=False)
#             # else:
#             #     sub_diag_merged_df.to_csv(to_write, index=False)
#             sub_diag_merged_df.to_csv(to_write, index=False)
#             logger.info(f'Finish: {file}, group: {group}.')

#     for file in procs:
#         proc = pd.read_csv(file, sep=',', dtype = {'Patid': str})
#         proc = proc.assign(ProcId = 'icd:' + proc['Icd_Flag'].astype(str) + '_proc:' + proc['Proc'])
#         proc = proc.assign(PatGroup = proc['Patid'].apply(lambda x: x[-1]))
#         year = re.findall(pattern, file)[0]
#         for group in user_group:
#             logger.info(f'Start: {file}, group: {group}.')
#             sub_proc = proc[proc['PatGroup'] == group]
#             sub_proc_merged = sub_proc.groupby(['Patid', 'Fst_Dt'])['ProcId'].apply(lambda x: ' '.join(x))
#             sub_proc_merged_df = sub_proc_merged.to_frame().reset_index()
            
#             sub_proc_merged_df.rename(columns={'Patid': 'patid', 'Fst_Dt': 'date', 'ProcId': 'procs'}, inplace=True)
            
#             to_write = os.path.join(result_path, f'proc_{year}_{group}.csv')
#             # if os.path.exists(to_write):
#             #     sub_proc_merged_df.to_csv(to_write, mode='a', header=False, index=False)
#             # else:
#             #     sub_proc_merged_df.to_csv(to_write, index=False)
#             sub_proc_merged_df.to_csv(to_write, index=False)
            
#             logger.info(f'Finish: {file}, group: {group}.')

    for file in pharms:
        pharm = pd.read_csv(file, sep=',', dtype = {'Patid': str})
        pharm = pharm.assign(PatGroup = pharm['Patid'].apply(lambda x: x[-1]))
        year = re.findall(pattern, file)[0]
        for group in user_group:
            logger.info(f'Start: {file}, group: {group}.')
            sub_pharm = pharm[pharm['PatGroup'] == group]
            sub_pharm_merged = sub_pharm.groupby(['Patid', 'Fill_Dt'])['Gnrc_Nm'].apply(lambda x: ' '.join(x))
            sub_pharm_merged_df = sub_pharm_merged.to_frame().reset_index()
            sub_pharm_merged_df.rename(columns={'Patid': 'patid', 'Fill_Dt': 'date', 'Gnrc_Nm': 'drugs'}, inplace=True)
            
            to_write = os.path.join(result_path, f'pharm_{year}_{group}.csv')
            # if os.path.exists(to_write):
            #     sub_pharm_merged_df.to_csv(to_write, mode='a', header=False, index=False)
            # else:
            #     sub_pharm_merged_df.to_csv(to_write, index=False)
            sub_pharm_merged_df.to_csv(to_write, index=False)

            logger.info(f'Finish: {file}, group: {group}.')

    logger.info('Finish: create_field_seq.')
    logger.info('*' * 100)


def merge_field():
    
    logger.info('*' * 100)
    logger.info('Start: merge_field.')

    for year in years:
        for group in user_group:
            diag_file = os.path.join(result_path, f'diag_{year}_{group}.csv')
            proc_file = os.path.join(result_path, f'proc_{year}_{group}.csv')
            pharm_file = os.path.join(result_path, f'pharm_{year}_{group}.csv')

            if os.path.exists(diag_file) and os.path.exists(proc_file) and os.path.exists(pharm_file):
                logger.info(f'Start: Year: {year}, Group: {group}.')
                diag = pd.read_csv(diag_file, sep=',', dtype = {'patid': str, 'date': str})
                proc = pd.read_csv(proc_file, sep=',', dtype = {'patid': str, 'date': str})
                pharm = pd.read_csv(pharm_file, sep=',', dtype = {'patid': str, 'date': str})
                
                tmp = pd.merge(diag, proc, how='outer', on=['patid', 'date'])
                tmp = pd.merge(tmp, pharm, how='outer', on=['patid', 'date'])
                tmp = tmp.fillna('')
                tmp['seq'] = tmp['diags'] + ' ' + tmp['procs'] + ' ' + tmp['drugs']
                tmp = tmp[['patid', 'date', 'seq']]
                
                to_write = os.path.join(result_path, f'{year}_{group}.csv')
                tmp.to_csv(to_write, index=False)
                logger.info(f'Finish: Year: {year}, Group: {group}.')
                
            else:
                logger.info(f'Data doesn\'t exist: Year: {year}, Group: {group}.')

    logger.info('Finish: merge_field.')
    logger.info('*' * 100)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev', action='store_true', help='Dev mode, just use much smaller tmp files.')
    parser.add_argument('--create_field_seq', action='store_true', help='Create in-year & in-field daily record.')
    parser.add_argument('--merge_field', action='store_true', help='Merge in-year daily record from field_seq.')
    parser.add_argument('--force_new', action='store_true', help='Force recreate the whole result folders.')
    parser.add_argument('--path', type=str)
    args = parser.parse_args()

#     data_path = '/home/liutianc/emr-data'
#     result_path = os.path.join(data_path, 'merge')
    data_path = args.path
    result_path = data_path
    if os.path.exists(result_path) and args.force_new:
        shutil.rmtree(result_path)
    makedirs(result_path)

    job = os.path.basename(__file__)
    job = job.split('.')[0]
    job += args.create_field_seq * '_create_field_seq' + args.merge_field * '_merge_field'
    logger = get_logger(job)

    if args.dev:
        diags = [str(x) for x in Path(data_path).glob("**/diag_201[0-9]_tmp.csv")]
        procs = [str(x) for x in Path(data_path).glob("**/proc_201[0-9]_tmp.csv")]
        pharms = [str(x) for x in Path(data_path).glob("**/pharm_201[0-9]_tmp.csv")]

    else:
        diags = [str(x) for x in Path(data_path).glob("**/diag_201[0-9].csv")]
        procs = [str(x) for x in Path(data_path).glob("**/proc_201[0-9].csv")]
        pharms = [str(x) for x in Path(data_path).glob("**/pharm_201[0-9].csv")]
        print(diags)
    user_group = [str(i) for i in range(10)]
    years = [str(i) for i in range(2010, 2019)]

    pattern = '\w*_(\d*)_tmp.csv' if args.dev else '\w*_(\d*).csv'

    if args.create_field_seq:
        create_field_seq()

    if args.merge_field:
        merge_field()






