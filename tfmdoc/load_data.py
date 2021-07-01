import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class ClaimsDataset(Dataset):
    def __init__(self, preprocess_dir):
        self.offsets = np.load(preprocess_dir + "patient_offsets.npy")
        self.records = torch.from_numpy(np.load(preprocess_dir + "diag_records.npy"))
        self.code_lookup = np.load(
            preprocess_dir + "diag_code_lookup.npy", allow_pickle=True
        )
        self.ids = np.load(preprocess_dir + "patient_ids.npy", allow_pickle=True)
        self.length = self.records.shape[0]
        # there should be labels/a dependent variable for each patient, right?
        # this is a placeholder
        self.labels = torch.from_numpy(np.random.randint(2, size=self.length))

    def __len__(self):
        # sufficient to return the number of patients
        return self.length

    def __getitem__(self, index):
        if index + 1 < self.length:
            start, stop = self.offsets[index], self.offsets[index + 1]
        elif index == self.length:
            start, stop = self.offsets[index], self.length
        patient_records = self.records[start:stop]
        # return array of diag codes and patient labels
        return patient_records, self.labels[index]


def padded_collate(batch):
    # each element in a batch is a pair (x, y)
    # un zip batch
    xs, ys = zip(*batch)
    xs = pad_sequence(xs, batch_first=True, padding_value=0)
    ys = torch.stack(ys)
    return xs, ys
