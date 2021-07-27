import os

import pytorch_lightning as pl
from hydra import compose, initialize
from hydra.utils import instantiate
from torch.utils.data import DataLoader

from tfmdoc.load_data import ClaimsDataset, padded_collate
from tfmdoc.preprocess import claims_pipeline


def test_lightning():
    with initialize(config_path=".."):
        cfg = compose(
            config_name="config",
            overrides=[
                "transformer.d_model=32",
                "transformer.n_blocks=1",
                "disease_codes.ald=['7231   ']",
                "preprocess.data_dir=tests/test_data/",
            ],
        )

        assert cfg["transformer"]["n_blocks"] == 1

        if "preprocessed_files" not in os.listdir("tests/test_data/"):
            # preprocess data if required
            claims_pipeline(cfg.preprocess.data_dir, cfg.disease_codes.ald, test=True)

        preprocess_dir = "tests/test_data/preprocessed_files/"
        dataset = ClaimsDataset(preprocess_dir, test=True)
        train_loader = DataLoader(dataset, collate_fn=padded_collate, batch_size=4)
        mapping = dataset.code_lookup
        transformer = instantiate(cfg.transformer, n_tokens=mapping.shape[0])
        trainer = pl.Trainer(fast_dev_run=True)
        trainer.fit(transformer, train_loader)


# TEST UTILITIES
def test_pipeline():
    with initialize(config_path=".."):
        cfg = compose(
            config_name="config",
            overrides=[
                "disease_codes.ald=['7231   ']",
                "preprocess.data_dir=tests/test_data/",
            ],
        )
        claims_pipeline(cfg.preprocess.data_dir, cfg.disease_codes.ald, test=True)
        preprocess_dir = "tests/test_data/preprocessed_files/"
        torch_dataset = ClaimsDataset(preprocess_dir)
        assert torch_dataset.offsets[-1] == len(torch_dataset)
        x, y = torch_dataset[7]
        assert len(x) == torch_dataset.offsets[8] - torch_dataset.offsets[7]
        assert y.item() in {0, 1}
