import h5py
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class ClaimsDataset(Dataset):
    def __init__(self, preprocess_dir):
        file = h5py.File(preprocess_dir + "preprocessed.hdf5", "r")
        self.offsets = np.cumsum(np.array(file["offsets"]))
        self.records = torch.from_numpy(np.array(file["tokens"]))
        self.code_lookup = np.array(file["diag_code_lookup"])
        self.ids = np.array(file["ids"])
        self._length = self.ids.shape[0]
        self.labels = torch.from_numpy(np.array(file["labels"])).long()
        demog = np.array(file["demo"])
        demog[:, 0] = np.nan_to_num(demog[:, 0])
        self.demo = torch.from_numpy(demog).float()

    def __len__(self):
        # sufficient to return the number of patients
        return self._length

    def __getitem__(self, index):
        if index > 0:
            start, stop = self.offsets[index - 1], self.offsets[index]
        elif index == 0:
            start, stop = 0, self.offsets[index]
        else:
            raise IndexError(f"Index {index:,} may be out of range ({self._length:,})")
        patient_records = self.records[start:stop]
        # return array of diag codes and patient labels
        return self.demo[index], patient_records, self.labels[index]


def padded_collate(batch):
    # each element in a batch is a pair (x, y)
    # un zip batch
    ws, xs, ys = zip(*batch)
    ws = torch.stack(ws)
    xs = pad_sequence(xs, batch_first=True, padding_value=0)
    ys = torch.stack(ys)
    return ws, xs, ys
