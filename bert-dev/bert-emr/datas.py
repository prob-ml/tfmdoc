import os

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class LineByLineTextDataset(Dataset):
    """This will be superseded by a framework-agnostic approach soon.
    """
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 file_path: str,
                 block_size: int,
                 add_special_tokens: bool = True):

        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
#         logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        batch_encoding = tokenizer.batch_encode_plus(lines, add_special_tokens=add_special_tokens,
                                                     max_length=block_size, truncation=True)
        self.examples = batch_encoding["input_ids"]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)
