import logging
import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class ClaimsDataset(Dataset):
    def __init__(
        self,
        preprocess_dir,
        filename="preprocessed",
        age_quantum=0,
    ):
        """Object containing features and labeling for each patienxs[t
            in the processed data set.

        Args:
            preprocess_dir (string): Path to directory containing
                hdf5 outputs from preprocessing
        """  # noqa: RST301
        self.file = h5py.File(preprocess_dir + filename + ".hdf5", "r")
        # indicates starting point of each patient's individual record
        self.offsets = np.cumsum(np.array(self.file["offsets"]))
        # flat file containing all records, concatenated
        self.records = np.array(self.file["tokens"])
        # array of all codes indicating numerical encoding
        self.code_lookup = np.array(self.file["diag_code_lookup"])
        # array of all patient IDs
        self.ids = np.array(self.file["ids"])
        self.visits = torch.from_numpy(np.array(self.file["visits"]))
        self.ages = torch.from_numpy(np.array(self.file["ages"]))
        self.ages = torch.clamp(self.ages, max=99)
        self._length = self.ids.shape[0]
        # bin ages to nearest multiple of q
        self._age_q = age_quantum
        self.mask = True

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
        # flip so that visits increase going back in time
        # from the last date
        visits = self.visits[start:stop]
        visits = visits.max() - visits
        ages = self.ages[start:stop].int()
        if self._age_q:
            ages = (ages / self._age_q).round() * self._age_q
        labels = []
        if self.mask:
            patient_records, labels = random_mask(
                patient_records, labels, self.code_lookup.shape[0]
            )
            patient_records = torch.tensor(patient_records)
            labels = torch.tensor(labels)

        # return array of diag codes and patient labels
        return ages, visits, None, patient_records, labels


class DiagnosisDataset(ClaimsDataset):
    def __init__(
        self,
        preprocess_dir,
        bag_of_words=False,
        synth_labels=None,
        filename="preprocessed",
        shuffle=False,
    ):
        super().__init__(preprocess_dir, filename)
        # array of binary labels (is a patient a case or control?)
        demog = np.array(self.file["demo"])
        demog[:, 0] = np.nan_to_num(demog[:, 0])
        demog[:, 0] = (demog[:, 0] - demog[:, 0].mean()) / demog[:, 0].std()
        # array of patient demographic data
        # first column is binary (sex) and second column is
        # normalized age at time of last record
        self.demo = torch.from_numpy(demog).float()
        self._bow = bag_of_words
        if not bag_of_words:
            self.records = torch.from_numpy(self.records)
        if synth_labels:
            self.labels = torch.load(preprocess_dir + synth_labels).long()
        else:
            self.labels = torch.from_numpy(np.array(self.file["labels"])).long()
        self.mask = False
        self._shuffle = shuffle

    def __getitem__(self, index):
        ages, visits, _, patient_records, _ = super().__getitem__(index)
        if self._bow:
            patient_records = pad_bincount(patient_records, self.code_lookup.shape)
            ages = pad_bincount(ages, 100)
        if self._shuffle:
            reindex = torch.randperm(patient_records.shape[0])
            patient_records = patient_records[reindex]
        return ages, visits, self.demo[index], patient_records, self.labels[index]


class EarlyDetectionDataset(ClaimsDataset):
    def __init__(
        self,
        preprocess_dir,
        bag_of_words=False,
        filename="preprocessed",
        late_cutoff=30,
        early_cutoff=90,
    ):
        super().__init__(preprocess_dir, filename)
        # array of binary labels (is a patient a case or control?)
        demog = np.array(self.file["demo"])
        demog[:, 0] = np.nan_to_num(demog[:, 0])
        demog[:, 0] = (demog[:, 0] - demog[:, 0].mean()) / demog[:, 0].std()
        # array of patient demographic data
        # first column is binary (sex) and second column is
        # normalized age at time of last record
        self.demo = torch.from_numpy(demog).float()
        self._bow = bag_of_words
        if not bag_of_words:
            self.records = torch.from_numpy(self.records)
        self.diagnosed_dates = np.array(self.file["diagnosed_dates"])
        self.dates = np.array(self.file["dates"])
        self.late_cutoff = late_cutoff
        self.early_cutoff = early_cutoff
        self.starts = []
        self.stops = []

        self.validate_sequences()

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        # this is hacky, but
        # it prevents the same patient from appearing twice in the same batch
        # or at least makes it very unlikely
        late_index = (index + 1000) % len(self)
        early_start = self.starts[index]
        late_start = self.starts[late_index]
        early_stop = int(early_start + self.stops[index][0])
        late_stop = int(late_start + self.stops[late_index][1])

        # incorporate ages
        early_ages = self.ages[early_start:early_stop].int()
        late_ages = self.ages[late_start:late_stop].int()

        # flip visit ordering
        early_visits = self.visits[early_start:early_stop]
        early_visits = early_visits.max() - early_visits
        late_visits = self.visits[late_start:late_stop]
        late_visits = late_visits.max() - late_visits

        early_records = self.records[early_start:early_stop]
        late_records = self.records[late_start:late_stop]

        if self._age_q:
            early_ages = (early_ages / self._age_q).round() * self._age_q
            late_ages = (late_ages / self._age_q).round() * self._age_q

        if self._bow:
            early_records = pad_bincount(early_records, self.code_lookup.shape)
            late_records = pad_bincount(late_records, self.code_lookup.shape)
            early_ages = pad_bincount(early_ages, 100)
            late_ages = pad_bincount(late_ages, 100)

        return (
            (early_ages, early_visits, self.demo[index], early_records, 0),
            (late_ages, late_visits, self.demo[index], late_records, 1),
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
                if ((late_stop >= 8) & (early_stop <= 512)) & (early_stop < late_stop):
                    self.starts.append(start)
                    self.stops.append((early_stop, late_stop))

        log.info(f"{len(self):,} viable patient records.")


def pad_bincount(records, n_codes):
    # get counts of each cod
    records = np.bincount(records)
    # pad each vector to length T, all possible codes
    padded = np.zeros(n_codes)
    padded[: len(records)] = records
    return torch.from_numpy(padded).float()


def random_mask(patient_records, labels, n_tokens):
    for i in range(len(patient_records)):
        prob = random.random()
        if prob < 0.15:
            prob /= 0.15
            labels.append(patient_records[i])
            if prob < 0.8:
                # mask out token
                patient_records[i] = 1
            elif prob < 0.9:
                # replace with random token
                patient_records[i] = random.randrange(n_tokens)
            # keep the same
        else:
            labels.append(0)

    return patient_records, labels
