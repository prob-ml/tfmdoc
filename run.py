import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from torch.utils.data import DataLoader

from tfmdoc.load_data import ClaimsDataset, padded_collate
from tfmdoc.preprocess import claims_pipeline


@hydra.main(config_path=".", config_name="config.yaml")
def main(cfg=None):
    if cfg.preprocess.do:
        claims_pipeline(
            data_dir=cfg.preprocess.data_dir,
            disease_codes=cfg.disease_codes.ald,
            length_range=(cfg.preprocess.min_length, cfg.preprocess.max_length),
            year_range=(cfg.preprocess.min_year, cfg.preprocess.max_year + 1),
            n_processed=cfg.preprocess.n_processed,
        )
    preprocess_dir = cfg.preprocess.data_dir + "preprocessed_files/"
    dataset = ClaimsDataset(preprocess_dir)
    train_loader = DataLoader(dataset, collate_fn=padded_collate, batch_size=8)
    mapping = dataset.code_lookup
    transformer = instantiate(cfg.transformer, n_tokens=mapping.shape[0])
    trainer = pl.Trainer()
    trainer.fit(transformer, train_loader)


if __name__ == "__main__":
    main()
