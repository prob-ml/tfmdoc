import numpy as np
import torch

import ehr
from ehr import Trainer, Transformer, utils


def test_version():
    assert ehr


def test_dummy_data():
    n_patients = 1024
    seq_length = 16
    dummy_df = utils.generate_dummy_data(n_patients, seq_length)
    patient_history = np.stack(
        dummy_df.groupby("patient_id")["icd_code"].apply(np.array).values
    )
    mapping, mapped = np.unique(patient_history, return_inverse=True)
    # check that null string comes first
    assert mapping[0] == "000"
    X_train = mapped.reshape(patient_history.shape)
    flattened = dummy_df.groupby("patient_id")["icd_code"].apply("|".join)
    y_train = flattened.apply(utils.set_outcome).values
    assert X_train.shape[0] == y_train.shape[0]

    tf_model = Transformer(
        n_tokens=mapping.shape[0], d_model=32, n_blocks=1, seq_length=seq_length
    )
    adam = torch.optim.Adam(tf_model.parameters())

    trainer = Trainer(model=tf_model, batch_size=32, optimizer=adam)
    losses = trainer.train(training_data=(X_train, y_train), epochs=1)
    assert True
