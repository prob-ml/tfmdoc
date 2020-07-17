import os
from typing import Any, List
import random

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from utils import DATA_PATH


class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 data_type: str,
                 max_length: int,
                 group: int = None,
                 add_special_tokens: bool = True,
                 truncate_method: str = 'first'):

        self.group = [str(i) for i in range(10)] if group is None else [str(group)]
        self.data_type = data_type

        lines = []
        for file in os.listdir(DATA_PATH):
            if self.is_target(file):
                with open(os.path.join(DATA_PATH, file), encoding='utf-8') as f:
                    lines += [line.split(',')[1] for line in f.read().splitlines() if
                              (len(line) > 0 and not line.isspace())][1:] # drop HEADER row.

        truncated_lines = []
        for line in lines:
            line = line.replace('\n', '')
            token_list = line.split(' ')
            if len(token_list) <= max_length:
                truncated_lines.append(line)
            else:
                if truncate_method == 'first':
                    token_list = token_list[:max_length]
                elif truncate_method == 'last':
                    token_list = token_list[-max_length:]
                elif truncate_method == 'random':
                    token_idx = random.sample(range(len(token_list)), max_length)
                    token_list = [token_list[idx] for idx in token_idx]
                truncated_lines.append(' '.join(token_list))
        lines = truncated_lines

        batch_encoding = tokenizer.batch_encode_plus(lines, add_special_tokens=add_special_tokens,
                                                     max_length=max_length, truncation=True)
        self.examples = batch_encoding["input_ids"]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)

    def is_target(self, file):

        if self.data_type in file:
            return max(map(lambda x: x in file, self.group))

        return False


