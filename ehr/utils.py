import copy
import random
from string import ascii_uppercase, digits

import pandas as pd
import torch


def clones(module, n_copies):
    """Clone n identical copies of a module"""
    return torch.nn.ModuleList([copy.deepcopy(module) for i in range(n_copies)])


### TEST UTILITIES


def random_icd():
    ans = random.choice(ascii_uppercase)
    ans += "".join(random.choices(digits, k=2))
    return ans


def set_outcome(codes):
    outcome = any(t in codes for t in ("14", "28", "42"))
    return int(outcome)


def generate_dummy_data(n_patients, seq_length):
    """
    Generate dataset for testing: each patient id has 16 dates on record, each with
    a random chance of containing a (fake) ICD code.
    """
    data = {}
    random.seed(12)
    for i in range(n_patients):
        patid = f"P{str.zfill(str(i), 4)}"
        data[patid] = {}
        for date in range(seq_length):
            # date is currently just an integer, could make it a datetime
            prob = random.random()
            if prob <= 0.25:
                # put in a randomized icd code
                data[patid][date] = random_icd()
            else:
                # null entry
                data[patid][date] = "000"

    df_dummy = pd.concat(
        {
            k: pd.DataFrame.from_dict(v, "index", columns=["icd_code"])
            for k, v in data.items()
        },
        axis=0,
    )

    df_dummy.index = df_dummy.index.set_names(["patient_id", "date"])
    df_dummy.reset_index(inplace=True)

    return df_dummy
