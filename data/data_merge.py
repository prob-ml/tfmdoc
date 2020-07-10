import os
import time
import argparse
from pathlib import Path

from utils import get_logger, makedirs


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
                        outfile.write(current_user + ',' + current_document + '\n')
                        current_user = ''
                        current_document = ''
                
        logger.info(f'Finish: group: {group}, time cost: {round(time.time() - start, 2)}.')
        start = time.time()
    
    logger.info(f'Finish all...')
                        
                        