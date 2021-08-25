import os

import pytorch_lightning as pl
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from torch.utils.data import DataLoader

from tfmdoc.load_data import ClaimsDataset, padded_collate
from tfmdoc.preprocess import ClaimsPipeline


def test_lightning():
    with initialize(config_path=".."):
        cfg = compose(
            config_name="config",
            overrides=[
                "transformer.d_model=32",
                "transformer.n_blocks=1",
                "disease_codes.ald=['7231']",
                "preprocess.data_dir=tests/test_data/",
                "preprocess.output_dir=tests/test_data/test_lightning/",
            ],
        )

        assert cfg["transformer"]["n_blocks"] == 1

        cpl = ClaimsPipeline(
            cfg.preprocess.data_dir,
            cfg.preprocess.output_dir,
            cfg.disease_codes.ald,
            test=True,
        )
        cpl.run()

        preprocess_dir = "tests/test_data/test_lightning/"
        dataset = ClaimsDataset(preprocess_dir)
        train_loader = DataLoader(dataset, collate_fn=padded_collate, batch_size=4)
        mapping = dataset.code_lookup
        transformer = instantiate(cfg.transformer, n_tokens=mapping.shape[0])
        trainer = pl.Trainer(fast_dev_run=True)
        trainer.fit(transformer, train_loader)
        os.remove(preprocess_dir + "preprocessed.hdf5")
        os.rmdir(preprocess_dir)


def test_transformer():
    with initialize(config_path=".."):
        cfg = compose(config_name="config")
        transformer = instantiate(cfg.transformer, n_tokens=6)
        w = torch.randn(size=(4, 2))
        w[:, 1] = (w[:, 1] > 0)
        x = torch.randint(high=6, size=(4, 32))
        y = torch.randint(high=2, size=(4,))
        assert transformer.configure_optimizers().defaults
        assert transformer.training_step((w, x, y), 0) > 0
        assert transformer.validation_step((w, x, y), 0) > 0
        encoding = transformer.pos_encode
        assert encoding.pe.shape[1] == 6000


def test_pipeline():
    with initialize(config_path=".."):
        cfg = compose(
            config_name="config",
            overrides=[
                "disease_codes.ald=['7231']",
                "preprocess.data_dir=tests/test_data/",
                "preprocess.output_dir=tests/test_data/test_pipeline/",
            ],
        )
        cpl = ClaimsPipeline(
            cfg.preprocess.data_dir,
            cfg.preprocess.output_dir,
            cfg.disease_codes.ald,
            test=True,
        )
        cpl.run()
        preprocess_dir = "tests/test_data/test_pipeline/"
        torch_dataset = ClaimsDataset(preprocess_dir)
        assert torch_dataset.offsets[-1] == torch_dataset.records.shape[0]
        w, x, y = torch_dataset[7]
        assert len(x) == torch_dataset.offsets[7] - torch_dataset.offsets[6]
        assert y.item() in {0, 1}
        assert w.shape[0] == 2
        os.remove(preprocess_dir + "preprocessed.hdf5")
        os.rmdir(preprocess_dir)
