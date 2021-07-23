import numpy as np

# NALD


def find_nash_ids(df_diag):
    nafld_codes = {"5718   ", "5719   ", "K760   "}
    contains_nash = df_diag[df_diag["Diag"] == "K7581  "]["Patid"].unique()
    contains_any_nafld = df_diag[df_diag["Diag"].isin(nafld_codes)]["Patid"].unique()
    return np.intersect1d(contains_nash, contains_any_nafld, assume_unique=True)


def find_no_nafld_ids(df_diag):
    all_nafld_codes = {"5718   ", "5719   ", "K760   ", "K7581  "}
    df_diag["is_nafld"] = df_diag["Diag"].isin(all_nafld_codes)
    no_nafld = ~df_diag.groupby("Patid")["is_nafld"].any()
    return no_nafld[no_nafld == True].index


# ALD


def find_ald_ids(df_diag, find_cases=True):
    ald_codes = {
        "5711   ",
        "5712   ",
        "5713   ",
        "K7010  ",
        "K7011  ",
        "K7041  ",
        "K7030  ",
        "K702   ",
        "K700   ",
        "K709   ",
        "K7031  ",
        "K7040  ",
    }
    df_diag["is_ald"] = df_diag["Diag"].isin(ald_codes)
    ald = df_diag.groupby("Patid")["is_ald"].any()
    # if find_cases is false, returns healthy ids/control units
    return ald[ald == find_cases].index
