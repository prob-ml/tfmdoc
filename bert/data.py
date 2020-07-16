import os
from typing import Any, List

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
                 add_special_tokens: bool = True):

        lines = []
        for file in os.listdir(DATA_PATH):
            if data_type == 'merged':
                if data_type not in file:
                    continue
                else:
                    with open(os.path.join(DATA_PATH, file), encoding='utf-8') as f:
                        lines += [line.split(',')[1] for line in f.read().splitlines() if
                                  (len(line) > 0 and not line.isspace())][1:] # drop HEADER row.

        batch_encoding = tokenizer.batch_encode_plus(lines, add_special_tokens=add_special_tokens,
                                                     max_length=max_length, truncation=True)
        self.examples = batch_encoding["input_ids"]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)
