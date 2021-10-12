import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from torch.utils.data import DataLoader, random_split

from tfmdoc.load_data import ClaimsDataset, balanced_sampler, padded_collate
from tfmdoc.preprocess import ClaimsPipeline


@hydra.main(config_path=".", config_name="config.yaml")
def main(cfg=None):
    if cfg.preprocess.do:
        cpl = ClaimsPipeline(
            data_dir=cfg.preprocess.data_dir,
            output_dir=cfg.preprocess.output_dir,
            disease_codes=cfg.disease_codes.ald,
            length_range=(cfg.preprocess.min_length, cfg.preprocess.max_length),
            year_range=(cfg.preprocess.min_year, cfg.preprocess.max_year + 1),
            n=cfg.preprocess.n,
            split_codes=cfg.preprocess.split_codes,
            include_labs=cfg.preprocess.include_labs,
        )
        cpl.run()
        if cfg.preprocess.etl_only:
            return
    preprocess_dir = cfg.preprocess.data_dir + "preprocessed_files/"
    dataset = ClaimsDataset(preprocess_dir)
    train_size = int(cfg.train.train_frac * len(dataset))
    val_size = len(dataset) - train_size
    keys = ("train", "val")
    subsets = random_split(
        dataset, (train_size, val_size), torch.Generator().manual_seed(42)
    )
    loaders = {}
    for key, subset in zip(keys, subsets):
        loaders[key] = DataLoader(
            subset,
            collate_fn=padded_collate,
            batch_size=cfg.train.batch_size,
            sampler=balanced_sampler(subset.indices, dataset.labels),
        )
    mapping = dataset.code_lookup
    tfmd = instantiate(cfg.model, n_tokens=mapping.shape[0])
    trainer = pl.Trainer(
        gpus=cfg.train.gpus,
        max_epochs=cfg.train.max_epochs,
        limit_train_batches=cfg.train.limit_train_batches,
    )
    trainer.fit(tfmd, loaders["train"], loaders["val"])


if __name__ == "__main__":
    main()
