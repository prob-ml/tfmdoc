import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate

from tfmdoc.load_data import ClaimsDataset, build_loaders
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
            include_labs=cfg.preproccess.include_labs,
        )
        cpl.run()
        if cfg.preprocess.etl_only:
            return
    preprocess_dir = cfg.preprocess.data_dir + "preprocessed_files/"
    dataset = ClaimsDataset(preprocess_dir, bag_of_words=(not cfg.model.transformer))
    train_size = int(cfg.train.train_frac * len(dataset))
    val_size = int(cfg.train.val_frac * len(dataset))
    test_size = len(dataset) - train_size - val_size
    loaders = build_loaders(
        dataset,
        train_size,
        val_size,
        test_size,
        pad=cfg.model.transformer,
        batch_size=cfg.train.batch_size,
    )
    mapping = dataset.code_lookup
    tfmd = instantiate(cfg.model, n_tokens=mapping.shape[0])
    trainer = pl.Trainer(
        gpus=cfg.train.gpus,
        max_epochs=cfg.train.max_epochs,
        limit_train_batches=cfg.train.limit_train_batches,
    )
    trainer.fit(tfmd, loaders["train"], loaders["val"])
    trainer.test(test_dataloaders=loaders["test"], ckpt_path="best")


if __name__ == "__main__":
    main()
