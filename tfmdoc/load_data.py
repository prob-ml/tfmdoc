import h5py
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.sampler import WeightedRandomSampler


class ClaimsDataset(Dataset):
    def __init__(
        self, preprocess_dir, bag_of_words=False, shuffle=False, synth_labels=None
    ):
        """Object containing features and labeling for each patient
            in the processed data set.

        Args:
            preprocess_dir (string): Path to directory containing
                hdf5 outputs from preprocessing
            bag_of_words (bool, optional): If true, transform patient features
                (e.g. medical history) into an unordered array of counts of all
                possible codes. If false, return an encoded
                sequence (for the Transformer model).
        """  # noqa: RST301
        file = h5py.File(preprocess_dir + "preprocessed.hdf5", "r")
        # indicates starting point of each patient's individual record
        self.offsets = np.cumsum(np.array(file["offsets"]))
        # flat file containing all records, concatenated
        self.records = np.array(file["tokens"])
        if not bag_of_words:
            self.records = torch.from_numpy(self.records)
        # array of all codes indicating numerical encoding
        self.code_lookup = np.array(file["diag_code_lookup"])
        # array of all patient IDs
        self.ids = np.array(file["ids"])
        self._length = self.ids.shape[0]
        self._shuffle = shuffle
        # array of binary labels (is a patient a case or control?)
        if synth_labels:
            self.labels = torch.load(preprocess_dir + synth_labels).long()
        else:
            self.labels = torch.from_numpy(np.array(file["labels"])).long()
        demog = np.array(file["demo"])
        demog[:, 0] = np.nan_to_num(demog[:, 0])
        # array of patient demographic data
        # first column is binary (sex) and second column is
        # normalized age at time of last record
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
        if self._shuffle:
            reindex = torch.randperm(patient_records.shape[0])
            patient_records = patient_records[reindex]
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
    # unzip batch
    ws, xs, ys = zip(*batch)
    ws = torch.stack(ws)
    if pad:
        xs = pad_sequence(xs, batch_first=True, padding_value=0)
    else:
        xs = torch.stack(xs)
    ys = torch.stack(ys)
    return ws, xs, ys


def balanced_sampler(ix, labels):
    # given a very imbalanced data set
    # downsample the majority class and upsample the minority class
    # this will result in an approximate 50-50 split per batch
    p = labels[ix].sum() / len(ix)
    weights = 1.0 / torch.tensor([1 - p, p], dtype=torch.float)
    sample_weights = weights[labels[ix]]
    return WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )


def build_loaders(
    dataset, train_size, val_size, test_size, pad, batch_size, save_test_index
):
    # create dataloaders for training, test, and (optionally)
    # validation set
    if val_size > 0:
        keys = ("train", "val", "test")
        lengths = (train_size, val_size, test_size)
    else:
        keys = ("train", "test")
        lengths = (train_size, test_size)
    subsets = random_split(dataset, lengths, torch.Generator().manual_seed(42))
    loaders = {}

    collate_fn = lambda x: padded_collate(x, pad=pad)

    for key, subset in zip(keys, subsets):
        if key == "test":
            sampler = None
            # save index of test samples for post hoc analysis
            if save_test_index:
                torch.save(torch.tensor(subset.indices), "test_index.pt")
        else:
            sampler = balanced_sampler(subset.indices, dataset.labels)
        loaders[key] = DataLoader(
            subset,
            collate_fn=collate_fn,
            batch_size=batch_size,
            sampler=sampler,
        )
    return loaders
