import random
from string import ascii_uppercase, digits

import numpy as np
import pandas as pd
import torch

from tfmdoc import Trainer, Transformer
from tfmdoc.preprocess import claims_pipeline


def test_dummy_data():
    n_patients = 1024
    seq_length = 16
    dummy_df = generate_dummy_data(n_patients, seq_length)
    patient_history = np.stack(
        dummy_df.groupby("patient_id")["icd_code"].apply(np.array).values
    )
    mapping, mapped = np.unique(patient_history, return_inverse=True)
    # check that null string comes first
    assert mapping[0] == "000"
    x_train = mapped.reshape(patient_history.shape)
    flattened = dummy_df.groupby("patient_id")["icd_code"].apply("|".join)
    y_train = flattened.apply(set_outcome).values
    assert x_train.shape[0] == y_train.shape[0]

    tf_model = Transformer(
        n_tokens=mapping.shape[0], d_model=32, n_blocks=1, seq_length=seq_length
    )
    adam = torch.optim.Adam(tf_model.parameters())

    trainer = Trainer(model=tf_model, batch_size=32, optimizer=adam)
    losses = trainer.train(training_data=(x_train, y_train), epochs=1)
    assert losses[-1] < 0.2


def test_pipeline():
    patient_group_cache = claims_pipeline(
        data_dir="/nfs/turbo/lsa-regier/OPTUM2/test_data/"
    )
    assert len(patient_group_cache.keys()) == 10


# TEST UTILITIES


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
