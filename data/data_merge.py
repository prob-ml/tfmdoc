import os
import time
import argparse
from pathlib import Path

from utils import get_logger, makedirs

invalid_tokens = {'icd:nan_diag:5601': 'icd:9_diag:5601',
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
 'icd:nan_diag:5770': 'icd:9_diag:5770'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    args = parser.parse_args()
    
    job = os.path.basename(__file__)
    job = job.split('.')[0]
    logger = get_logger(job)
    
    path = args.path

    user_group = [str(i) for i in range(10)]
    start = time.time()


    for group in user_group:
        
        read = os.path.join(path, f'group_{group}.csv')
        write = os.path.join(path, f'group_{group}_merged.csv')
        
        with open(read, 'r') as infile:
            with open(write, 'w') as outfile:
                outfile.write('patid,document\n')
                current_user = ''
                current_document = ''
                for line in infile:
                    user, date, hist = line.split(",")
                    user = user.strip()
                    hist = hist.strip()
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
                        
                        