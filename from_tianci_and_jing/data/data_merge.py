import os
import json
import time
import argparse
from pathlib import Path

from utils import get_logger, makedirs

invalid_tokens = {
    'icd:nan_diag:5601': 'icd:9_diag:5601',
     'icd:nan_proc:0DJD8ZZ': 'icd:10_proc:0DJD8ZZ',
     'icd:nan_diag:5733': 'icd:9_diag:5733',
     'icd:nan_proc:0DP67UZ': 'icd:10_proc:0DP67UZ',
     'icd:nan_proc:0DHA7UZ': 'icd:10_proc:0DHA7UZ',
     'icd:nan_proc:B2151ZZ': 'icd:10_proc:B2151ZZ',
     'icd:nan_proc:3E0H8GC': 'icd:10_proc:3E0H8GC',
     'icd:nan_proc:B2111ZZ': 'icd:9_proc:B2111ZZ',
     'icd:nan_proc:4A023N7': 'icd:10_proc:4A023N7',
     'icd:nan_proc:0DB68ZX': 'icd:10_proc:0DB68ZX',
     'icd:nan_proc:02HV33Z': 'icd:10_proc:02HV33Z',
     'icd:nan_diag:2630': 'icd:9_diag:2630',
     'icd:nan_diag:5770': 'icd:9_diag:5770'
}


def get_icd_map():
    path = '/nfs/turbo/lsa-regier/emr-data/icd_map.json'
    with open(path, 'r') as file:
        icd_map = json.load(file)
    
    return icd_map
    
    
def get_icd_10():
    path = '/nfs/turbo/lsa-regier/emr-data/icd_10.json'
    with open(path, 'r') as file:
        icd_10 = json.load(file)
    
    return icd_10
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--unidiag', action='store_true', default=True)
    args = parser.parse_args()
    
    job = os.path.basename(__file__)
    job = job.split('.')[0]
    logger = get_logger(job)
    
    path = args.path

    if args.unidiag:
        icd_map = get_icd_map()
        icd_10 = get_icd_10()

    user_group = [str(i) for i in range(10)]
    start = time.time()

    for group in user_group:
        
        read = os.path.join(path, f'group_{group}.csv')
        write = os.path.join(path, f'group_{group}_merged.csv' if not args.unidiag else f'group_{group}_merged_unidiag.csv')
        
        with open(read, 'r') as infile:
            with open(write, 'w') as outfile:
                outfile.write('patid,document\n')
                current_user = ''
                current_document = ''
                for line in infile:
                    user, date, hist = line.split(",")
                    user = user.strip()
                    hist = hist.strip()
                    
                    # Convert to Uni Diag codes hist and clean some other outliers.
                    if args.unidiag:
                        reg_hist = ''
                        for token in hist.split(' '):
                            # Remove meaningless '-' in diag/proc codes.
                            if 'proc' in token or 'diag' in token:
                                token = token.replace('-', '')
                            
                            # Convert ICD-9 diag codes.
                            if 'icd:9_diag' in token:
                                diag_code = token.split('_')[1]
                                diag_code = diag_code.split(':')[1]

                                is_success = False

                                # Some ICD-9 are exact ICD 10 codes.
                                converted_diag_code = icd_10.get(diag_code, None)
                                if converted_diag_code is not None:
                                    is_success = True

                                # Try to convert diag codes based on icd_map
                                if not is_success:
                                    converted_diag_code = icd_map.get(diag_code, None)
                                    if converted_diag_code is not None:
                                        is_success = True
                                    else:
                                        # Zero may be padded at the beginning, which is not handled in icd_map,
                                        # try to remove it and match.
                                        cur_diag_code = diag_code
                                        while cur_diag_code[0] == '0':
                                            cur_diag_code = cur_diag_code[1:]
                                            converted_diag_code = icd_map.get(cur_diag_code, None)
                                            if converted_diag_code is not None:
                                                is_success = True
                                                break

                                # Successful conversion
                                if is_success:
                                    # converted_diag_code is a list, where the first element is the target.
                                    converted_diag_code = converted_diag_code[0]
                                    # Replace icd flag from 9 to 10.
                                    token = token.replace('icd:9_', 'icd:10_')
                                    # Replace icd codes from 9 to 10.
                                    token = token.replace(diag_code, converted_diag_code)
                            reg_hist += token + ' '
                        hist = reg_hist.strip()
                                
                    # first record of current user
                    if current_user == '':
                        current_user = user
                        current_document = hist
                    # Same user: just append
                    elif current_user == user:
                        current_document += ' [SEP] ' + hist
                    # next user
                    else:
                        # There are some missing tokens, but ALL of them has valid form in the data.
                        for invalid in invalid_tokens:
                            current_document = current_document.replace(invalid, invalid_tokens[invalid])
                        
                        outfile.write(current_user + ',' + current_document + '\n')
                        current_user = ''
                        current_document = ''
                
        logger.info(f'Finish: group: {group}, time cost: {round(time.time() - start, 2)}.')
        start = time.time()
    
    logger.info(f'Finish all...')
                        
                        