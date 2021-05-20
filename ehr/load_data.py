import os

import pandas as pd
from fastparquet import ParquetFile
from torch.utils.data import Dataset

# path to data: /nfs/turbo/lsa-regier/OPTUM2/


class ClaimsDataset(Dataset):
    def __init__(self, data_dir="/nfs/turbo/lsa-regier/OPTUM2/", test=False):
        if test:
            # this is a sample of the first 1 mil rows from diag_2007
            self.dataframe = pd.read_csv(
                "/nfs/turbo/lsa-regier/OPTUM2/sample_diag.csv",
                dtype={"Patid": str, "Diag": str, "Icd_Flag": str},
            )
        else:
            all_files = os.listdir(data_dir)
            self.diags = [
                file
                for file in all_files
                if file.startswith("diag") and file.endswith(".parquet")
            ]
            # this is merely placeholder logic
            # goal is to concatenate several parquet files into one object
            parquet_file = ParquetFile(data_dir + self.diags[0])
            self.dataframe = parquet_file.to_pandas(
                ["Patid", "Icd_Flag", "Diag", "Diag_Position", "Fst_Dt"]
            )

        self._clean_dataframe()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        return self.dataframe.iloc[index]

    def _clean_dataframe(self):
        # drop records without a diagnostic code
        self.dataframe = self.dataframe.dropna(subset=["Diag"])
        # combine ICD format with the code into one field
        self.dataframe["DiagId"] = (
            "icd:" + self.dataframe["Icd_Flag"] + "_diag:" + self.dataframe["Diag"]
        )
        # sort by patient id and date, concatenate all diagnoses into a sequence
        # unclear whether position of a claim actually matters. Assuming not.
        grouped = self.dataframe.groupby(["Patid", "Fst_Dt"])["DiagId"].apply(" ".join)
        self.dataframe = grouped.to_frame().reset_index()

        self.dataframe.rename(
            columns={"Patid": "patid", "Fst_Dt": "date", "DiagId": "diags"},
            inplace=True,
        )
