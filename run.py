import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import WeightedRandomSampler

from tfmdoc.load_data import ClaimsDataset, padded_collate
from tfmdoc.preprocess import ClaimsPipeline


@hydra.main(config_path=".", config_name="config.yaml")
def main(cfg=None):
    if cfg.preprocess.do:
        cpl = ClaimsPipeline(
            data_dir=cfg.preprocess.data_dir,
            disease_codes=cfg.disease_codes.ald,
            length_range=(cfg.preprocess.min_length, cfg.preprocess.max_length),
            year_range=(cfg.preprocess.min_year, cfg.preprocess.max_year + 1),
            n_processed=cfg.preprocess.n_processed,
            split_codes=cfg.preprocess.split_codes,
        )
        cpl.run()
    preprocess_dir = cfg.preprocess.data_dir + "preprocessed_files/"
    dataset = ClaimsDataset(preprocess_dir)
    train_size = int(cfg.train.train_frac * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, (train_size, val_size))
    sampler = balanced_sampler(train_dataset.indices, dataset.labels)
    train_loader = DataLoader(
        train_dataset,
        collate_fn=padded_collate,
        batch_size=cfg.train.batch_size,
        sampler=sampler,
    )
    val_loader = DataLoader(
        val_dataset, collate_fn=padded_collate, batch_size=cfg.train.batch_size
    )
    mapping = dataset.code_lookup
    transformer = instantiate(cfg.transformer, n_tokens=mapping.shape[0])
    trainer = pl.Trainer(gpus=cfg.train.gpus, max_epochs=3)
    trainer.fit(transformer, train_loader, val_loader)


def balanced_sampler(train_ix, labels):
    p = labels[train_ix].sum() / len(train_ix)
    weights = 1.0 / torch.tensor([1 - p, p], dtype=torch.float)
    sample_weights = weights[labels[train_ix]]
    return WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )


if __name__ == "__main__":
    main()
