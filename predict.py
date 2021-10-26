import hydra
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from tfmdoc import Tfmd
from tfmdoc.load_data import ClaimsDataset, padded_collate


@hydra.main(config_path=".", config_name="config.yaml")
def main(cfg=None):
    preprocess_dir = cfg.preprocess.data_dir + "preprocessed_files/"
    dataset = ClaimsDataset(preprocess_dir, bag_of_words=(not cfg.model.transformer))

    collate_fn = lambda x: padded_collate(x, pad=cfg.model.transformer)
    model = Tfmd.load_from_checkpoint(cfg.predict.checkpoint)
    loader = DataLoader(
        dataset, collate_fn=collate_fn, batch_size=cfg.predict.batch_size
    )
    trainer = pl.Trainer(
        gpus=cfg.train.gpus,
        max_epochs=cfg.train.max_epochs,
    )
    trainer.predict(model, dataloaders=loader)

    # controlling recall rate happens here, e.g.
    # do the reporting here


if __name__ == "__main__":
    main()
