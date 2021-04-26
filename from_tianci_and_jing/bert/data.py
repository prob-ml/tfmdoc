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


class CausalBertSynDataset(Dataset):
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
                 alpha: float = 0.25,
                 beta: float = 1.,
                 offset_t: float = 0.,
                 offset_p: float = 0.,
                 seed=2020,
                 ):
        
        np.random.seed(seed)
        torch.manual_seed(seed)

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
        prop_scores = []
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
            prop_scores.append(self.treat_portion(line))
        lines = truncated_lines
        batch_encoding = tokenizer.batch_encode_plus(lines, add_special_tokens=add_special_tokens,
                                                     max_length=max_length, truncation=True, padding=True)
        self.tokens = batch_encoding["input_ids"]

        # Create propensity score, treatment and response
        self.prop_scores = torch.tensor(prop_scores, dtype=torch.float32)
        self.treatment = Binomial(1, self.prop_scores).sample()

        self.alpha = alpha
        self.beta = beta
        self.offset_t = offset_t
        self.offset_p = offset_p
        
        self.response = self.generate_response(self.treatment, self.prop_scores)
        self.pseudo_response = self.generate_response(1. - self.treatment, self.prop_scores)

        self.prop_scores = self.prop_scores.reshape(-1, 1)#.to(self.device)
        self.treatment = self.treatment.reshape(-1, 1)#.to(self.device)
        self.response = self.response.reshape(-1, 1)#.to(self.device)
        self.pseudo_response = self.pseudo_response.reshape(-1, 1)#.to(self.device)

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, i) -> list:
        token = torch.tensor(self.tokens[i], dtype=torch.long, device=self.device)
        treatment = self.treatment[i].to(self.device)
        response = self.response[i].to(self.device)
        prop_score = self.prop_scores[i].to(self.device)
        return token, treatment, response, prop_score

    def treat_portion(self, x):
        score = 0.2
        pattern = 'diag:J45' # Asthma
        if pattern in x:
            score = 0.8
        return score

    def generate_response(self, treatment, prop_scores):
        prob = torch.sigmoid(self.alpha * (treatment - self.offset_t) + self.beta * (prop_scores - self.offset_p))
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


class CausalBertRealDataset(Dataset):
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
                 pretreat_tokens: list = None,
                 treat_tokens: list = None,
                 response_tokens: list = None,
                 group: list = None,
                 add_special_tokens: bool = True,
                 truncate_method: str = 'last',
                 seed=2020, ):

        # np.random.seed(seed)
        # torch.manual_seed(seed)
        assert treat_tokens is not None and response_tokens is not None
        self.pretreat_tokens = pretreat_tokens
        self.treat_tokens = treat_tokens
        self.response_tokens = response_tokens

        if group is None: group = range(10)
        self.user_group = [str(i) for i in group]
        self.data_type = data_type
        self.is_unidiag = is_unidiag
        self.device = device
        lines = []
        for file in os.listdir(DATA_PATH):
            if self.is_target(file):
                with open(os.path.join(DATA_PATH, file), encoding='utf-8') as f:
                    # Note: f.read().splitlines() will drop the '\n' at the end of each line automatically.
                    lines += [line.replace('\n', '').split(',')[1] for line in f.read().splitlines() if
                              (len(line) > 0 and not line.isspace())][1:]  # [1:] drop HEADER row.

        truncated_lines = []
        treatments = []
        responses = []
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

            treatment, response, line = self.get_causal_data(line)
            if line is not None:
                treatments.append(treatment)
                responses.append(response)
                truncated_lines.append(line)

        lines = truncated_lines
        batch_encoding = tokenizer.batch_encode_plus(lines, add_special_tokens=add_special_tokens,
                                                     max_length=max_length, truncation=True, padding=True)
        self.tokens = batch_encoding["input_ids"]

        # Create propensity score, treatment and response
        # print(treatments[:10])
        self.treatments = torch.tensor(treatments, dtype=torch.float32).reshape(-1, 1).to(self.device)
        self.responses = torch.tensor(responses, dtype=torch.float32).reshape(-1, 1).to(self.device)

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, i) -> list:
        token = torch.tensor(self.tokens[i], dtype=torch.long, device=self.device)
        treatment = self.treatments[i]#.to(self.device)
        response = self.responses[i]#.to(self.device)
        return token, treatment, response

    def get_causal_data(self, x):
        token_list = x.split(' ')

        first_pretreat_idx = 0
        if self.pretreat_tokens:
            is_target = False
            for idx, token in enumerate(token_list):
                for pretreat in self.pretreat_tokens:
                    if pretreat in token and is_target == False:
                        is_target = True
                        first_pretreat_idx = idx
                        break
                if is_target:
                    break
            if not is_target:
                return None, None, None

        treatment, response = False, False
        first_treat_idx = first_pretreat_idx
        for idx, token in enumerate(token_list[first_pretreat_idx:]):
            for treat in self.treat_tokens:
                if treat in token and treatment == False:
                    treatment = True
                    first_treat_idx = idx
                    break
            if treatment:
                break

        if treatment:
            # discard treatment tokens.
            token_list_new = token_list[:first_treat_idx] + token_list[(first_treat_idx + 1):]
            line = ' '.join(token_list_new)

            if first_treat_idx == len(token_list) - 1:
                return treatment, response, line

            while token_list[first_treat_idx] != '[SEP]':
                # Since patient is probably prescribed pain killer right after the surgery,
                # we move the starting point of "after treatment" to the next visit, namely,
                # after it passes a special token [SEP]
                first_treat_idx += 1
                if first_treat_idx == len(token_list) - 1:
                    return treatment, response, line

            # Find responses tokens in tokens after the first treatment occurs.
            response = self.response_match(token_list[first_treat_idx:])
        else:
            # Find responses tokens after the first pretreatment (diagnosis) occurs, note now
            # first_pretreat_idx == first_treat_idx, we separate codes here for better readability.
            response = self.response_match(token_list[first_pretreat_idx:])

            line = x
        return treatment, response, line

    def response_match(self, token_list):
        if len(token_list) <= 0:
            return False

        for token in token_list:
            for response_token in self.response_tokens:
                if response_token in token:
                    return True
        return False


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


