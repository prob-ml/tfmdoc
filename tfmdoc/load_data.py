import h5py
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.sampler import WeightedRandomSampler


class ClaimsDataset(Dataset):
    def __init__(
        self,
        preprocess_dir,
        bag_of_words=False,
        shuffle=False,
        synth_labels=None,
        filename="preprocessed",
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
        self.file = h5py.File(preprocess_dir + filename + ".hdf5", "r")
        # indicates starting point of each patient's individual record
        self.offsets = np.cumsum(np.array(self.file["offsets"]))
        # flat file containing all records, concatenated
        self.records = np.array(self.file["tokens"])
        if not bag_of_words:
            self.records = torch.from_numpy(self.records)
        # array of all codes indicating numerical encoding
        self.code_lookup = np.array(self.file["diag_code_lookup"])
        # array of all patient IDs
        self.ids = np.array(self.file["ids"])
        self.visits = torch.from_numpy(np.array(self.file["visits"]))
        self._length = self.ids.shape[0]
        self._shuffle = shuffle
        # array of binary labels (is a patient a case or control?)
        if synth_labels:
            self.labels = torch.load(preprocess_dir + synth_labels).long()
        else:
            self.labels = torch.from_numpy(np.array(self.file["labels"])).long()
        demog = np.array(self.file["demo"])
        demog[:, 0] = np.nan_to_num(demog[:, 0])
        demog[:, 0] = (demog[:, 0] - demog[:, 0].mean()) / demog[:, 0].std()
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
        # add one to allow for zero padding
        visits = self.visits[start:stop] + 1
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
        else:
            patient_records = patient_records.flip(0)
            visits = visits.flip(0)
        # return array of diag codes and patient labels
        return visits, self.demo[index], patient_records, self.labels[index]


class EarlyDetectionDataset(ClaimsDataset):
    def __init__(
        self,
        preprocess_dir,
        bag_of_words=False,
        shuffle=False,
        synth_labels=None,
        filename="preprocessed",
        late_cutoff=30,
        early_cutoff=90,
    ):
        super().__init__(preprocess_dir, bag_of_words, shuffle, synth_labels, filename)
        self.diagnosed_dates = np.array(self.file["diagnosed_dates"])
        self.dates = np.array(self.file["dates"])
        self.late_cutoff = late_cutoff
        self.early_cutoff = early_cutoff
        self.viable_indices = []
        self.stops = []

        self.validate_sequences()

    def __len__(self):
        return len(self.viable_indices)

    def __getitem__(self, index):
        if index > 0:
            start = self.offsets[index - 1]
        elif index == 0:
            start = 0

        early_stop = int(start + self.stops[index][0])
        late_stop = int(start + self.stops[index][1])

        early_visits = self.visits[start:early_stop] + 1
        late_visits = self.visits[start:late_stop] + 1

        early_records = self.records[start:early_stop]
        late_records = self.records[start:late_stop]

        return (
            (early_visits, self.demo[index], early_records, 0),
            (late_visits, self.demo[index], late_records, 1),
        )

    def validate_sequences(self):
        for index, offset in enumerate(self.offsets):
            if index > 0:
                start, stop = self.offsets[index - 1], offset
            elif index == 0:
                start, stop = 0, offset

            ind_dates = self.dates[start:stop]

            late_stop = np.argmax(
                ind_dates >= self.diagnosed_dates[index] - self.late_cutoff
            )

            early_stop = np.argmax(
                ind_dates >= self.diagnosed_dates[index] - self.early_cutoff
            )

            if (early_stop > 0) & (late_stop > 0):
                if (late_stop >= 8) & (early_stop <= 512):
                    self.viable_indices.append(index)
                    self.stops.append((early_stop, late_stop))


# HELPERS


def padded_collate(batch, pad=True, early_detection=False):
    if early_detection:
        early, late = zip(*batch)
        vs, ws, xs, ys = zip(*(early + late))
    else:
        vs, ws, xs, ys = zip(*batch)
    ws = torch.stack(ws)
    if pad:
        vs = pad_sequence(vs, batch_first=True, padding_value=0)
        xs = pad_sequence(xs, batch_first=True, padding_value=0)
    else:
        # can ignore visits in bag-of-words model
        vs = None
        xs = torch.stack(xs)
    ys = torch.stack(ys)
    return vs, ws, xs, ys


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
    dataset,
    lengths,
    pad,
    batch_size,
    save_test_index,
    random_seed,
    early_detection,
):
    # create dataloaders for training, test, and (optionally)
    # check whether to create validation set
    if lengths[1] > 0:
        keys = ("train", "val", "test")
    else:
        keys = ("train", "test")
        lengths = (lengths[0], lengths[2])
    subsets = random_split(dataset, lengths, torch.Generator().manual_seed(random_seed))
    loaders = {}

    collate_fn = lambda x: padded_collate(x, pad, early_detection)

    for key, subset in zip(keys, subsets):
        if key == "test" or early_detection:
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
