import os

from fastparquet import ParquetFile

# path to data: /nfs/turbo/lsa-regier/OPTUM2/


def split_icd_codes(code):
    """
    For each record in a diagnoses file, split the ICD code into the first
    3 digits (category) and the whole code (specific diagnosis).
    """
    return code[:6] + ", " + code


def claims_pipeline(data_dir):
    all_files = os.listdir(data_dir)
    diags = [
        file
        for file in all_files
        if file.startswith("diag") and file.endswith(".parquet")
    ]
    # placeholder logic
    # generalize this to iterating through all diags, not just one
    parquet_file = ParquetFile(data_dir + diags[0])
    dataframe = parquet_file.to_pandas(["Patid", "Icd_Flag", "Diag", "Fst_Dt"])
    # convert byte arrays to strings
    # should be able to do this in parquet...
    dataframe["Icd_Flag"] = dataframe["Icd_Flag"].str.decode("UTF-8")
    dataframe["Diag"] = dataframe["Diag"].str.decode("UTF-8")
    return clean_dataframe(dataframe)


def clean_dataframe(dataframe):
    # make icd flag more readable
    dataframe["Icd_Flag"] = dataframe["Icd_Flag"].str.replace("9 ", "09")
    # drop records without a diagnostic code
    dataframe = dataframe.dropna(subset=["Diag"])
    # combine ICD format with the code into one field; strip whitespace
    dataframe["DiagId"] = dataframe["Icd_Flag"] + "." + dataframe["Diag"]
    dataframe["DiagId"] = dataframe["DiagId"].str.strip()

    # split code into general category and specific condition
    dataframe["Diag"] = dataframe["Diag"].apply(split_icd_codes)

    # sort by patient id and date, concatenate all diagnoses into a sequence
    # assume that "position" (of code on a form) does not matter
    grouped = dataframe.groupby(["Patid", "Fst_Dt"])["DiagId"].apply(" ".join)
    cleaned = dataframe = grouped.to_frame().reset_index()

    column_map = {"Patid": "patid", "Fst_Dt": "date", "DiagId": "diags"}
    cleaned.rename(columns=column_map, inplace=True)

    return cleaned
