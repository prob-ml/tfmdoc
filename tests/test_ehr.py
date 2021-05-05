import numpy as np

import ehr
from helpers import utils


def test_version():
    assert ehr


def test_dummy_data():
    df = utils.generate_dummy_data()
    X_train = np.stack(df.groupby("patient_id")["icd_code"].apply(np.array).values)
    flattened = df.groupby("patient_id")["icd_code"].apply("|".join)
    y_train = flattened.apply(utils.set_outcome).values
    print(X_train.shape, y_train.shape)

    assert True
