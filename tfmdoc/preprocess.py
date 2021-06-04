import os

from fastparquet import ParquetFile

# path to data: /nfs/turbo/lsa-regier/OPTUM2/


def claims_pipeline(data_dir):
    all_files = os.listdir(data_dir)
    diags = [
        file
        for file in all_files
        if file.startswith("diag") and file.endswith(".parquet")
    ]
    patient_group_cache = {}
    for diag in diags:
        parquet_file = ParquetFile(data_dir + diag)
        dataframe = parquet_file.to_pandas(["Patid", "Icd_Flag", "Diag", "Fst_Dt"])
        cleaned = clean_diag_data(dataframe)
        compile_patient_records(cleaned, patient_group_cache)
    return patient_group_cache


def clean_diag_data(dataframe):
    dataframe["Icd_Flag"] = dataframe["Icd_Flag"].str.decode("UTF-8")
    dataframe["Diag"] = dataframe["Diag"].str.decode("UTF-8")
    # make icd flag more readable
    dataframe["Icd_Flag"] = dataframe["Icd_Flag"].str.replace("9 ", "09")
    # drop records without a diagnostic code
    dataframe = dataframe.dropna(subset=["Diag"])
    # combine ICD format with the code into one field; strip whitespace
    dataframe["DiagId"] = dataframe["Icd_Flag"] + "." + dataframe["Diag"]
    dataframe["DiagId"] = dataframe["DiagId"].str.strip()
    # split code into general category and specific condition
    dataframe["DiagId"] = dataframe["DiagId"].apply(split_icd_codes)
    dataframe["Patid"] = dataframe["Patid"].astype(int).astype(str)
    return dataframe


def compile_patient_records(dataframe, cache, grouping_index=1):
    dataframe["patient_group"] = dataframe["Patid"].str[-grouping_index:]
    patient_groups = id_groups(grouping_index)
    for group in patient_groups:
        subframe = dataframe[dataframe["patient_group"] == group]
        # sort by patient id and date, concatenate all diagnoses into a sequence
        # assume that "position" (of code on a form) does not matter
        patient_logs = subframe.groupby(["Patid", "Fst_Dt"])["DiagId"].apply(" ".join)
        patient_logs = patient_logs.to_frame().reset_index()
        column_map = {"Patid": "patid", "Fst_Dt": "date", "DiagId": "diags"}
        patient_logs.rename(columns=column_map, inplace=True)
        if cache.get(group) is None:
            cache[group] = [patient_logs]
        else:
            cache[group].append(patient_logs)


# UTILS


def split_icd_codes(code):
    """
    For each record in a diagnoses file, split the ICD code into the first
    3 digits (category) and the whole code (specific diagnosis).
    """
    return code[:6] + ", " + code


def id_groups(n_digits):
    if n_digits == 1:
        ans = [str(i) for i in range(10)]
    elif n_digits == 2:
        ans = []
        for i in range(100):
            if i < 10:
                ans.append("0" + str(i))
            else:
                ans.append(str(i))
    return ans
