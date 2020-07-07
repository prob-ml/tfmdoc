import pandas as pd
import numpy as np

from pathlib import Path
import os
import logging
import argparse


from utils import get_logger

parser = argparse.ArgumentParser()
parser.add_argument('--dev', action='store_true')
args = parser.parse_args()


data_path = '/home/liutianc/emr-data'
result_path = os.path.join(data_path, 'merge')

job = os.path.basename(__file__)
job = job.split('.')[0]
logger = get_logger(job)


if args.dev:

	diags = [str(x) for x in Path(data_path).glob("**/diag_201*_tmp.csv")]
	procs = [str(x) for x in Path(data_path).glob("**/proc_201*_tmp.csv")]
	pharms = [str(x) for x in Path(data_path).glob("**/pharm_201*_tmp.csv")]

else:

	diags = [str(x) for x in Path(data_path).glob("**/diag_201*.csv")]
	procs = [str(x) for x in Path(data_path).glob("**/proc_201*.csv")]
	pharms = [str(x) for x in Path(data_path).glob("**/pharm_201*.csv")]


userGroup = [str(i) for i in range(10)]


if __name__ == '__main__':
	for file in diags:
	    diag = pd.read_csv(file, sep=',', dtype = {'Patid': str})
	    diag = diag.assign(DiagId = 'icd:' + diag['Icd_Flag'].astype(str) + '_loc:' + diag['Loc_cd'].astype(str) + '_diag:' + diag['Diag'])
	    diag = diag.assign(PatGroup = diag['Patid'].apply(lambda x: x[-1]))
	    
	    for group in userGroup:
	        logger.info(f'Start: {file}, group: {group}.')
	        sub_diag = diag[diag['PatGroup'] == group]
	        sub_diag_merged = sub_diag.groupby(['Patid', 'Fst_Dt'])['DiagId'].apply(lambda x: ' '.join(x))
	        sub_diag_merged_df = sub_diag_merged.reset_index()
	        
	        sub_diag_merged_df.rename({'Patid': 'patid', 'Fst_Dt': 'date', 'DiagId': 'diags'})
	        to_write = os.path.join(result_path, 'diag_' + group + '.csv')
	        if os.path.exists(to_write):
	            sub_diag_merged.to_csv(to_write, mode='a', header=False, index=False)
	        else:
	            sub_diag_merged.to_csv(to_write, index=False)
	        
	        logger.info(f'Finish: {file}, group: {group}.')



	for file in procs:
	    proc = pd.read_csv(file, sep=',', dtype = {'Patid': str})
	    proc = proc.assign(ProcId = 'icd:' + proc['Icd_Flag'].astype(str) + '_proc:' + proc['Proc'])
	    proc = proc.assign(PatGroup = proc['Patid'].apply(lambda x: x[-1]))
	    
	    for group in userGroup:
	        logger.info(f'Start: {file}, group: {group}.')
	        sub_proc = proc[proc['PatGroup'] == group]
	        sub_proc_merged = sub_proc.groupby(['Patid', 'Fst_Dt'])['ProcId'].apply(lambda x: ' '.join(x))
	        sub_proc_merged_df = sub_proc_merged.reset_index()
	        
	        sub_proc_merged_df.rename({'Patid': 'patid', 'Fst_Dt': 'date', 'ProcId': 'procs'})

	        to_write = os.path.join(result_path, 'proc_' + group + '.csv')
	        if os.path.exists(to_write):
	            sub_proc_merged_df.to_csv(to_write, mode='a', header=False, index=False)
	        else:
	            sub_proc_merged_df.to_csv(to_write, index=False)
	        logger.info(f'Finish: {file}, group: {group}.')


	for file in pharms:
		print(file)
	    pharm = pd.read_csv(file, sep=',', dtype = {'Patid': str})
	    pharm = pharm.assign(PatGroup = pharm['Patid'].apply(lambda x: x[-1]))
	    
	    for group in userGroup:
	        logger.info(f'Start: {file}, group: {group}.')
	        sub_pharm = pharm[pharm['PatGroup'] == group]
	        sub_pharm_merged = sub_pharm.groupby(['Patid', 'Fill_Dt'])['Gnrc_Nm'].apply(lambda x: ' '.join(x))
	        sub_pharm_merged_df = sub_pharm_merged.reset_index()
	        sub_pharm_merged_df.rename({'Patid': 'patid', 'Fill_Dt': 'date', 'Gnrc_Nm': 'drugs'})

	        
	        to_write = os.path.join(result_path, 'pharm_' + group + '.csv')
	        if os.path.exists(to_write):
	            sub_pharm_merged_df.to_csv(to_write, mode='a', header=False, index=False)
	        else:
	            sub_pharm_merged_df.to_csv(to_write, index=False)
	        logger.info(f'Finish: {file}, group: {group}.')




