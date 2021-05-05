import random
from string import ascii_uppercase, digits

import pandas as pd


def random_icd():
    ans = random.choice(ascii_uppercase)
    # k = random.randrange(2, 6)
    ans += "".join(random.choices(digits, k=2))
    return ans


def set_outcome(codes):
    outcome = any(t in codes for t in ["14", "28", "42"])
    return int(outcome)


def generate_dummy_data():
    """
    Generate dataset for testing: each patient id has 16 dates on record, each with
    a random chance of containing a (fake) ICD code.
    """
    data = dict()
    random.seed(12)
    for i in range(1024):
        patid = "P" + str.zfill(str(i), 4)
        data[patid] = {}
        for date in range(16):
            # date is currently just an integer, could make it a datetime
            prob = random.random()
            if prob <= 0.25:
                # put in a randomized icd code
                data[patid][date] = random_icd()
            else:
                # null entry
                data[patid][date] = "_"

    df = pd.concat(
        {
            k: pd.DataFrame.from_dict(v, "index", columns=["icd_code"])
            for k, v in data.items()
        },
        axis=0,
    )

    df.index = df.index.set_names(["patient_id", "date"])
    df.reset_index(inplace=True)

    return df
