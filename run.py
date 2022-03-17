import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint

from tfmdoc import load_data as ld
from tfmdoc.preprocess import ClaimsPipeline, DiagnosisPipeline


@hydra.main(config_path=".", config_name="config.yaml")
def main(cfg=None):
    """Script for preprocessing health insurance claims data and training a
    classification model.

    Args:
        cfg (OmegaConf object, optional): Placeholder. Defaults to None.
    """
    if cfg.preprocess.do:
        if cfg.preprocess.mode == "pretraining":
            pipeline = ClaimsPipeline(
                data_dir=cfg.preprocess.data_dir,
                output_dir=cfg.preprocess.output_dir,
                length_range=(cfg.preprocess.min_length, cfg.preprocess.max_length),
                year_range=(cfg.preprocess.min_year, cfg.preprocess.max_year + 1),
                n=cfg.preprocess.n,
                split_codes=cfg.preprocess.split_codes,
                output_name=cfg.preprocess.filename,
            )
        else:
            pipeline = DiagnosisPipeline(
                data_dir=cfg.preprocess.data_dir,
                output_dir=cfg.preprocess.output_dir,
                disease_codes=cfg.disease_codes[cfg.preprocess.disease],
                length_range=(cfg.preprocess.min_length, cfg.preprocess.max_length),
                year_range=(cfg.preprocess.min_year, cfg.preprocess.max_year + 1),
                n=cfg.preprocess.n,
                split_codes=cfg.preprocess.split_codes,
                prediction_window=cfg.preprocess.prediction_window,
                output_name=cfg.preprocess.filename,
                mode=cfg.preprocess.mode,
                pad=cfg.pad[cfg.preprocess.disease],
            )
        # run preprocessing pipeline
        pipeline.run()
        return
    preprocess_dir = cfg.preprocess.data_dir + "preprocessed_files/"
    if cfg.preprocess.early_detection:
        dataset = ld.EarlyDetectionDataset(
            preprocess_dir,
            bag_of_words=(not cfg.model.transformer),
            shuffle=cfg.train.shuffle,
            synth_labels=cfg.train.synth_labels,
            filename=cfg.preprocess.filename,
            late_cutoff=cfg.preprocess.prediction_window,
            early_cutoff=cfg.preprocess.early_detection,
        )
    else:
        dataset = ld.ClaimsDataset(
            preprocess_dir,
            bag_of_words=(not cfg.model.transformer),
            shuffle=cfg.train.shuffle,
            synth_labels=cfg.train.synth_labels,
            filename=cfg.preprocess.filename,
        )
    train_size = int(cfg.train.train_frac * len(dataset))
    val_size = int(cfg.train.val_frac * len(dataset))
    test_size = len(dataset) - train_size - val_size
    loaders = ld.build_loaders(
        dataset,
        (train_size, val_size, test_size),
        pad=cfg.model.transformer,
        batch_size=cfg.train.batch_size,
        save_test_index=cfg.train.save_prediction,
        random_seed=cfg.train.random_seed,
        early_detection=bool(cfg.preprocess.early_detection),
    )
    mapping = dataset.code_lookup
    # initialize tfmd model from config settings
    tfmd = instantiate(cfg.model, n_tokens=mapping.shape[0])
    # ensure model with least validation loss is used for testing
    if val_size:
        callbacks = [ModelCheckpoint(monitor="val_loss")]
    else:
        callbacks = None
    trainer = pl.Trainer(
        gpus=cfg.train.gpus,
        max_epochs=cfg.train.max_epochs,
        limit_train_batches=cfg.train.limit_train_batches,
        callbacks=callbacks,
    )
    # train and validate
    trainer.fit(tfmd, loaders["train"], loaders.get("val"))
    # test model
    trainer.test(test_dataloaders=loaders["test"], ckpt_path="best")
    diagnostic_plot(tfmd, trainer)
    if cfg.train.save_prediction:
        torch.save(tfmd.results[0], "test_predictions.pt")
        torch.save(tfmd.results[1], "test_targets.pt")


def diagnostic_plot(model, trainer):
    # Plot precision/recall curve for the test data set
    precs, recs, _ = model.pr_curve(*model.results)
    fig = plt.figure()
    plt.plot(recs.cpu(), precs.cpu())
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    trainer.logger.experiment.add_figure("pr_curve", fig)


if __name__ == "__main__":
    main()
