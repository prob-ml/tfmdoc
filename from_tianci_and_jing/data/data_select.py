import csv
import os
import argparse
from pathlib import Path

from utils import get_logger, makedirs


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()
    
    input_path = args.input_path
    output_path = args.output_path
    makedirs(output_path)
    
    job = os.path.basename(__file__)
    job = job.split('.')[0]
    logger = get_logger(job)
    
    diags = [str(x) for x in Path(input_path).glob("**/diag_201[0-9].csv")]
    procs = [str(x) for x in Path(input_path).glob("**/proc_201[0-9].csv")]
    pharms = [str(x) for x in Path(input_path).glob("**/pharm_201[0-9].csv")]
    
    diag_pattern = '{},' * 6 + '{}\n'
    proc_pattern = '{},' * 5 + '{}\n'
    pharm_pattern = '{},' * 5 + '{}\n'
    
    logger.info('*' * 100)
    logger.info('Start: Select diag data.')
    for file in diags:
        file_name = os.path.split(file)[1]
        output_file = os.path.join(output_path, file_name)
        
        logger.info(f'Start: {file_name}.')
        with open(file, newline='') as infile:
            with open(output_file, 'w') as outfile:
                diagreader = csv.reader(infile)
                
                for row in diagreader:
                    row = [cell.replace('.0', '').strip() for cell in row]
                    
                    patid = row[0].split('.0')[0].strip()
                    claimid = row[2].split('.0')[0].strip()
                    Diag = row[3].split('.0')[0].strip()
                    Diag_Position = row[4].split('.0')[0].strip()
                    Icd_Flag = row[5].split('.0')[0].strip()
                    Loc_cd = row[6].split('.0')[0].strip()
                    Fst_Dt = row[10].strip()
                    
                    select_row = diag_pattern.format(patid, claimid, Diag, Diag_Position, Icd_Flag, Loc_cd, Fst_Dt)
                    if '.0' in select_row:
                        print(f'Failure at: {file}.')
                        print(select_row)
                        exit(1)
                        
                    outfile.write(select_row)
                    
        logger.info(f'Finish: {file_name}.')
    logger.info('Finish: Select diag data.')
    
    logger.info('*' * 100)
    logger.info('Start: Select proc data.')
    for file in procs:
        file_name = os.path.split(file)[1]
        output_file = os.path.join(output_path, file_name)

        logger.info(f'Start: {file_name}.')
        with open(file, newline='') as infile:
            with open(output_file, 'w') as outfile:
                procreader = csv.reader(infile)
                
                for row in procreader:
                    row = [cell.replace('.0', '').strip() for cell in row]
                    
                    patid = row[0].split('.0')[0].strip()
                    claimid = row[2].split('.0')[0].strip()
                    Icd_Flag = row[3].split('.0')[0].strip()
                    Proc = row[4].split('.0')[0].strip()
                    Proc_Position = row[5].split('.0')[0].strip()
                    Fst_Dt = row[8]
                    select_row = proc_pattern.format(patid, claimid, Icd_Flag, Proc, Proc_Position, Fst_Dt)
                    
                    if '.0' in select_row:
                        print(f'Failure at: {file}.')
                        print(select_row)
                        exit(1)
                        
                    outfile.write(select_row)
                    
        logger.info(f'Finish: {file_name}.')
    logger.info('Finish: Select proc data.')

    logger.info('*' * 100)
    logger.info('Start: Select pharm data.')
    for file in pharms:
        file_name = os.path.split(file)[1]
        output_file = os.path.join(output_path, file_name)

        logger.info(f'Start: {file_name}.')
        with open(file, newline='') as infile:
            with open(output_file, 'w') as outfile:
                pharmreader = csv.reader(infile)
                
                for row in pharmreader:
                    row = [cell.strip() for cell in row]
                    
                    patid = row[0].split('.0')[0].strip()
                    claimid = row[7].split('.0')[0].strip()
                    Fill_Date = row[14].split('.0')[0].strip()
                    Gnrc_Nm = row[19].split('.0')[0].strip()
                    Quantity = row[25].split('.0')[0].strip()
                    Rfl_Nbr = row[26].split('.0')[0].strip()
                    
                    Gnrc_Nm = Gnrc_Nm.replace('"', '').replace(',', '_').replace(' ', '_').strip()

                    select_row = pharm_pattern.format(patid, claimid, Fill_Date, Gnrc_Nm, Quantity, Rfl_Nbr)
                    outfile.write(select_row)
                    
        logger.info(f'Finish: {file_name}.')
    logger.info('Finish: Select pharm data.')

