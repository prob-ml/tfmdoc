import numpy as np
from torch.utils.data import Dataset


class ClaimsDataset(Dataset):
    def __init__(self, preprocess_dir):
        self.offsets = np.load(preprocess_dir + "patient_offsets.npy")
        self.records = np.load(preprocess_dir + "diag_records.npy")
        self.code_lookup = np.load(
            preprocess_dir + "diag_code_lookup.npy", allow_pickle=True
        )
        self.ids = np.load(preprocess_dir + "patient_ids.npy", allow_pickle=True)
        self.length = self.records.shape[0]
        # there should be labels/a dependent variable for each patient, right?

    def __len__(self):
        # sufficient to return the number of patients
        return self.length

    def __getitem__(self, index):
        if index + 1 < self.length:
            start, stop = self.offsets[index], self.offsets[index + 1]
        elif index == self.length:
            start, stop = self.offsets[index], self.length
        return self.records[start:stop]
