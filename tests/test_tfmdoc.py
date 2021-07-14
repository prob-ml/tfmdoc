import os

import pytorch_lightning as pl
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from torch.utils.data import DataLoader

from tfmdoc.load_data import ClaimsDataset, padded_collate
from tfmdoc.preprocess import claims_pipeline


def test_lightning():
    if "preprocessed_files/" not in os.listdir("tests/test_data/"):
        # preprocess data if required
        claims_pipeline(data_dir="tests/test_data/")
    preprocess_dir = "tests/test_data/preprocessed_files/"
    dataset = ClaimsDataset(preprocess_dir)
    train_loader = DataLoader(dataset, collate_fn=padded_collate, batch_size=4)
    mapping = dataset.code_lookup
    with initialize(config_path="../conf"):
        cfg = compose(
            config_name="transformer",
            overrides=["transformer.d_model=32", "transformer.n_blocks=1"],
        )
        assert cfg["transformer"]["n_blocks"] == 1

        transformer = instantiate(cfg.transformer, n_tokens=mapping.shape[0])
        trainer = pl.Trainer(fast_dev_run=True)
        trainer.fit(transformer, train_loader)


# TEST UTILITIES
def test_pipeline():
    claims_pipeline(data_dir="tests/test_data/")
    preprocess_dir = "tests/test_data/preprocessed_files/"
    torch_dataset = ClaimsDataset(preprocess_dir)
    assert torch_dataset.offsets[-1] == len(torch_dataset)
    t, x, y = torch_dataset[7]
    assert torch.equal(t, torch.sort(t)[0])
    assert len(x) == torch_dataset.offsets[8] - torch_dataset.offsets[7]
    assert y.item() in {0, 1}
