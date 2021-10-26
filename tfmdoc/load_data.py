import h5py
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.sampler import WeightedRandomSampler


class ClaimsDataset(Dataset):
    def __init__(self, preprocess_dir, bag_of_words=False):
        file = h5py.File(preprocess_dir + "preprocessed.hdf5", "r")
        self.offsets = np.cumsum(np.array(file["offsets"]))
        self.records = np.array(file["tokens"])
        if not bag_of_words:
            self.records = torch.from_numpy(self.records)
        self.code_lookup = np.array(file["diag_code_lookup"])
        self.ids = np.array(file["ids"])
        self._length = self.ids.shape[0]
        self.labels = torch.from_numpy(np.array(file["labels"])).long()
        demog = np.array(file["demo"])
        demog[:, 0] = np.nan_to_num(demog[:, 0])
        self.demo = torch.from_numpy(demog).float()
        self._bow = bag_of_words

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
        if self._bow:
            # get counts of each code
            patient_records = np.bincount(patient_records)
            # pad each vector to length T, all possible codes
            padded = np.zeros(self.code_lookup.shape)
            padded[: len(patient_records)] = patient_records
            patient_records = torch.from_numpy(padded).float()
        # return array of diag codes and patient labels
        return self.demo[index], patient_records, self.labels[index]


# HELPERS


def padded_collate(batch, pad=True):
    # each element in a batch is a pair (x, y)
    # un zip batch
    ws, xs, ys = zip(*batch)
    ws = torch.stack(ws)
    if pad:
        xs = pad_sequence(xs, batch_first=True, padding_value=0)
    else:
        xs = torch.stack(xs)
    ys = torch.stack(ys)
    return ws, xs, ys


def balanced_sampler(ix, labels):
    p = labels[ix].sum() / len(ix)
    weights = 1.0 / torch.tensor([1 - p, p], dtype=torch.float)
    sample_weights = weights[labels[ix]]
    return WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )


def build_loaders(dataset, train_size, val_size, test_size, pad, batch_size):
    if val_size > 0:
        keys = ("train", "val", "test")
        lengths = (train_size, val_size, test_size)
    else:
        keys = ("train", "test")
        lengths = (train_size, test_size)
    keys = ("train", "test")
    subsets = random_split(dataset, lengths, torch.Generator().manual_seed(42))
    loaders = {"val": None}

    collate_fn = lambda x: padded_collate(x, pad=pad)

    for key, subset in zip(keys, subsets):
        loaders[key] = DataLoader(
            subset,
            collate_fn=collate_fn,
            batch_size=batch_size,
            sampler=balanced_sampler(subset.indices, dataset.labels),
        )
    return loaders
