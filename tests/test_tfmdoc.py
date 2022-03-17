import os

import pytorch_lightning as pl
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from torch.utils.data import DataLoader

from tfmdoc import load_data
from tfmdoc.aipw import aipw_estimator
from tfmdoc.preprocess import DiagnosisPipeline


def test_lightning():

    with initialize(config_path=".."):
        cfg = compose(
            config_name="config",
            overrides=[
                "model.d_model=32",
                "model.n_blocks=1",
                "disease_codes.ald=['7231']",
                "preprocess.data_dir=tests/test_data/",
                "preprocess.output_dir=tests/test_data/test_lightning/",
                "preprocess.filename=preprocessed",
            ],
        )

        assert cfg["model"]["n_blocks"] == 1

        cpl = DiagnosisPipeline(
            cfg.preprocess.data_dir,
            cfg.preprocess.output_dir,
            cfg.disease_codes.ald,
            output_name=cfg.preprocess.filename,
            test=True,
        )
        cpl.run()

        preprocess_dir = "tests/test_data/test_lightning/"
        dataset = load_data.ClaimsDataset(preprocess_dir)
        train_loader = DataLoader(
            dataset, collate_fn=load_data.padded_collate, batch_size=4
        )
        mapping = dataset.code_lookup
        tfmd = instantiate(cfg.model, n_tokens=mapping.shape[0])
        trainer = pl.Trainer(fast_dev_run=True)
        trainer.fit(tfmd, train_loader)
        os.remove(preprocess_dir + "preprocessed.hdf5")
        os.rmdir(preprocess_dir)


def test_early_etl():

    with initialize(config_path=".."):
        cfg = compose(
            config_name="config",
            overrides=[
                "model.d_model=32",
                "model.n_blocks=1",
                "disease_codes.ald=['7231']",
                "preprocess.data_dir=tests/test_data/",
                "preprocess.output_dir=tests/test_data/test_lightning/",
                "preprocess.filename=preprocessed",
                "preprocess.mode=early_detection",
            ],
        )

        cpl = DiagnosisPipeline(
            cfg.preprocess.data_dir,
            cfg.preprocess.output_dir,
            cfg.disease_codes.ald,
            mode=cfg.preprocess.mode,
            output_name=cfg.preprocess.filename,
            test=True,
        )
        cpl.run()

        preprocess_dir = "tests/test_data/test_lightning/"
        dataset = load_data.EarlyDetectionDataset(
            preprocess_dir, late_cutoff=10, early_cutoff=30
        )
        assert len(dataset)
        assert len(dataset[0]) == 2
        assert len(dataset[0][0]) == 5
        assert dataset[0][1][4] - dataset[0][0][4] == 1
        os.remove(preprocess_dir + "preprocessed.hdf5")
        os.rmdir(preprocess_dir)


def test_model():
    with initialize(config_path=".."):
        cfg = compose(config_name="config")
        model = instantiate(cfg.model, n_tokens=6)
        t = torch.randint(low=20, high=60, size=(4, 32))
        v = torch.randint(high=3, size=(4, 32))
        w = torch.randn(size=(4, 2))
        # fmt: off
        w[:, 1] = (w[:, 1] >= 0)
        # fmt: on
        x = torch.randint(high=6, size=(4, 32))
        y = torch.LongTensor([0, 1, 1, 0])
        assert model.configure_optimizers().defaults
        assert model.training_step((t, v, w, x, y), 0) > 0
        assert model.validation_step((t, v, w, x, y), 0) > 0


def test_bow():
    with initialize(config_path=".."):
        cfg = compose(
            config_name="config",
            overrides=["model.transformer=False", "model.d_model=40"],
        )
        model = instantiate(cfg.model, n_tokens=200)
        t = torch.randint(low=20, high=60, size=(4, 100))
        v = None
        w = torch.randn(size=(4, 2))
        # fmt: off
        w[:, 1] = (w[:, 1] >= 0)
        # fmt: on
        x = torch.randint(high=4, size=(4, 200)).float()
        y = torch.LongTensor([0, 1, 1, 0])
        assert model.configure_optimizers().defaults
        assert model.training_step((t, v, w, x, y), 0) > 0
        assert model.validation_step((t, v, w, x, y), 0) > 0


def test_pipeline():
    with initialize(config_path=".."):
        cfg = compose(
            config_name="config",
            overrides=[
                "disease_codes.ald=['7231']",
                "preprocess.data_dir=tests/test_data/",
                "preprocess.output_dir=tests/test_data/test_pipeline/",
                "preprocess.filename=preprocessed",
            ],
        )
        cpl = DiagnosisPipeline(
            cfg.preprocess.data_dir,
            cfg.preprocess.output_dir,
            cfg.disease_codes.ald,
            output_name=cfg.preprocess.filename,
            test=True,
        )
        cpl.run()
        preprocess_dir = "tests/test_data/test_pipeline/"
        torch_dataset = load_data.ClaimsDataset(preprocess_dir)
        assert torch_dataset.offsets[-1] == torch_dataset.records.shape[0]
        t, v, w, x, y = torch_dataset[7]
        assert len(x) == torch_dataset.offsets[7] - torch_dataset.offsets[6]
        assert y.item() in {0, 1}
        assert w.shape[0] == 2
        assert v.max().item() == 1
        assert t.shape[0] == x.shape[0]
        os.remove(preprocess_dir + "preprocessed.hdf5")
        os.rmdir(preprocess_dir)


def test_aipw():
    torch.manual_seed(35)
    # ey_x contains both treatment and control estimates
    y, t, pi_x, ey_x = simulate_causality(1000, 40, 32, 22, 4)
    est_ate = aipw_estimator(y, t, pi_x, ey_x)

    # simulate true ATE empirically over 100 trials
    assigned_treatment = torch.bernoulli(ey_x[:, 1].repeat(100, 1)).mean(1)
    assigned_control = torch.bernoulli(ey_x[:, 0].repeat(100, 1)).mean(1)
    simulated_ates = assigned_treatment - assigned_control
    assert abs(est_ate - simulated_ates.mean()) < 0.05


# helpers


def simulate_causality(n, n_codes, seq_length, treatment_event, outcome_event):
    x = torch.randint(n_codes, (n, seq_length))
    t_mask = (x == treatment_event).any(1)
    prop_score = torch.empty((n,)).fill_(0.8)
    prop_score = prop_score.masked_fill_(t_mask, 0.2)
    treatment = torch.bernoulli(prop_score)
    outcome_mask = (x == outcome_event).any(1)
    outcome_probs = torch.empty((n,)).fill_(0.8)
    outcome_probs.masked_fill_(torch.logical_and(outcome_mask, treatment), 0.2)
    outcome = torch.bernoulli(outcome_probs)

    ey_x0 = torch.empty((n,)).fill_(0.8)
    ey_x1 = torch.empty((n,)).fill_(0.8)
    ey_x0.masked_fill_(
        torch.logical_and(outcome_mask, torch.empty((n,)).fill_(False)), 0.2
    )
    ey_x1.masked_fill_(
        torch.logical_and(outcome_mask, torch.empty((n,)).fill_(True)), 0.2
    )

    estimated_outcomes = torch.cat([ey_x0.unsqueeze(1), ey_x1.unsqueeze(1)], dim=1)

    return outcome, treatment, prop_score, estimated_outcomes
