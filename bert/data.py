import os
from typing import Any, List
import random
import re 

import numpy as np

import torch
from torch.distributions.binomial import Binomial
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from utils import DATA_PATH


class MLMDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 data_type: str,
                 is_unidiag: bool,
                 max_length: int,
                 min_length: int,
                 device: str,
                 group: list = None,
                 add_special_tokens: bool = True,
                 truncate_method: str = 'first'):

        if group is None: group = range(10)
        self.user_group = [str(i) for i in group]# if group is None else [str(group)]
        self.data_type = data_type
        self.is_unidiag = is_unidiag
        self.device = device
        lines = []
        for file in os.listdir(DATA_PATH):
            if self.is_target(file):
                with open(os.path.join(DATA_PATH, file), encoding='utf-8') as f:
                    # f.read().splitlines() will drop the '\n' at the end of each line automatically.
                    lines += [line.replace('\n', '').split(',')[1] for line in f.read().splitlines() if
                              (len(line) > 0 and not line.isspace())][1:] #[1:] drop HEADER row.
        truncated_lines = []
        
        for line in lines:
            token_list = line.split(' ')
            if len(token_list) <= min_length:
                continue

            if len(token_list) <= max_length:
                truncated_lines.append(line)
            else:
                if truncate_method == 'first':
                    token_list = token_list[:max_length]
                elif truncate_method == 'last':
                    token_list = token_list[-max_length:]
                elif truncate_method == 'random':
                    token_idx = random.sample(range(len(token_list)), max_length)
                    token_list = [token_list[idx] for idx in token_idx.sort()]
                
                truncated_lines.append(' '.join(token_list))
        lines = truncated_lines
        # Padding is handled by MLMCollator.
        batch_encoding = tokenizer.batch_encode_plus(lines, add_special_tokens=add_special_tokens,
                                                     max_length=max_length, truncation=True)
        self.examples = batch_encoding["input_ids"]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)#, device=self.device)

    def is_target(self, file):

        # First check if 'unidiag' is correct.
        if self.is_unidiag:
            if 'unidiag' not in file:
                return False
        else:
            if 'unidiag' in file:
                return False

        # Since a file name may contain 'merged', but will not contain 'daily',
        # so we need to use nested if-condition.
        if 'merged' in file:
            if self.data_type == 'merged':
                return max(map(lambda x: x in file, self.user_group))
        else:
            if self.data_type != 'merged':
                return max(map(lambda x: x in file, self.user_group))

        return False


class CausalBertDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 data_type: str,
                 is_unidiag: bool,
                 max_length: int,
                 min_length: int,
                 device: str,
                 group: list = None,
                 add_special_tokens: bool = True,
                 truncate_method: str = 'first',
                 ):

        if group is None: group = range(10)
        self.user_group = [str(i) for i in group]# if group is None else [str(group)]
        self.data_type = data_type
        self.is_unidiag = is_unidiag
        self.device = device
        lines = []
        for file in os.listdir(DATA_PATH):
            if self.is_target(file):
                with open(os.path.join(DATA_PATH, file), encoding='utf-8') as f:
                    # f.read().splitlines() will drop the '\n' at the end of each line automatically.
                    lines += [line.replace('\n', '').split(',')[1] for line in f.read().splitlines() if
                              (len(line) > 0 and not line.isspace())][1:] #[1:] drop HEADER row.

        truncated_lines = []
        prop_score = []
        for line in lines:
            token_list = line.split(' ')
            
            # Drop too short sequence.
            if len(token_list) <= min_length:
                continue
                
            # Sample from too long sequence.
            if len(token_list) > max_length:
                if truncate_method == 'first':
                    token_list = token_list[:max_length]
                elif truncate_method == 'last':
                    token_list = token_list[-max_length:]
                elif truncate_method == 'random':
                    token_idx = random.sample(range(len(token_list)), max_length)
                    token_list = [token_list[idx] for idx in token_idx.sort()]
                line = ' '.join(token_list)
                
            truncated_lines.append(line)
            prop_score.append(self.treat_portion(line))
        lines = truncated_lines
        batch_encoding = tokenizer.batch_encode_plus(lines, add_special_tokens=add_special_tokens,
                                                     max_length=max_length, truncation=True, padding=True)
        self.tokens = batch_encoding["input_ids"]

        # Create propensity score, treatment and response
        self.prop_score = torch.tensor(prop_score, dtype=torch.float32)
        self.treatment = Binomial(1, self.prop_score).sample()
        self.response = self.generate_response(self.treatment, self.prop_score)
        self.pseudo_response = self.generate_response(1. - self.treatment, self.prop_score)

        self.prop_score = self.prop_score.reshape(-1, 1).to(self.device)
        self.treatment = self.treatment.reshape(-1, 1).to(self.device)
        self.response = self.response.reshape(-1, 1).to(self.device)
        self.pseudo_response = self.pseudo_response.reshape(-1, 1).to(self.device)

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, i) -> list:
        token = torch.tensor(self.tokens[i], dtype=torch.long, device=self.device)
        treatment = self.treatment[i]
        response = self.response[i]
        return token, treatment, response

    def treat_portion(self, x):
        score = 0.2
        pattern = 'diag:J45' # Asthma
        if pattern in x:
            score = 0.8
        return score

    def generate_response(self, treatment, prop_score, alpha=0.25, beta=1., c=0.2):

        prob = torch.sigmoid(alpha * treatment + beta * (prop_score - c))
        return Binomial(1, prob).sample()
    
    def is_target(self, file):

        # First check if 'unidiag' is correct.
        if self.is_unidiag:
            if 'unidiag' not in file:
                return False
        else:
            if 'unidiag' in file:
                return False

        # Since a file name may contain 'merged', but will not contain 'daily',
        # so we need to use nested if-condition.
        if 'merged' in file:
            if self.data_type == 'merged':
                return max(map(lambda x: x in file, self.user_group))
        else:
            if self.data_type != 'merged':
                return max(map(lambda x: x in file, self.user_group))

        return False

