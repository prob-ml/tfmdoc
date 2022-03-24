import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint

from tfmdoc import datasets as ds
from tfmdoc.loader import build_loaders, calc_sizes
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
    if cfg.preprocess.mode == "pretraining":
        dataset = ds.ClaimsDataset(
            preprocess_dir,
            filename=cfg.preprocess.filename,
        )
    elif cfg.preprocess.mode == "diagnosis":
        dataset = ds.DiagnosisDataset(
            preprocess_dir,
            bag_of_words=(not cfg.model.transformer),
            synth_labels=cfg.train.synth_labels,
            filename=cfg.preprocess.filename,
        )
    elif cfg.preprocess.mode == "early_detection":
        dataset = ds.EarlyDetectionDataset(
            preprocess_dir,
            bag_of_words=(not cfg.model.transformer),
            shuffle=cfg.train.shuffle,
            synth_labels=cfg.train.synth_labels,
            filename=cfg.preprocess.filename,
            late_cutoff=cfg.preprocess.prediction_window,
            early_cutoff=cfg.preprocess.early_detection,
        )
    sizes = calc_sizes(cfg.train.train_frac, cfg.train.val_frac, len(dataset))
    loaders = build_loaders(
        dataset,
        sizes,
        pad=cfg.model.transformer,
        batch_size=cfg.train.batch_size,
        save_test_index=cfg.train.save_prediction,
        random_seed=cfg.train.random_seed,
        mode=cfg.preprocess.mode,
    )
    mapping = dataset.code_lookup
    if cfg.preprocess.mode == "pretraining":
        model = instantiate(cfg.bert, n_tokens=mapping.shape[0])
    else:
        model = instantiate(cfg.model, n_tokens=mapping.shape[0])
    callbacks = [ModelCheckpoint(monitor="val_loss")]
    trainer = pl.Trainer(
        gpus=cfg.train.gpus,
        max_epochs=cfg.train.max_epochs,
        limit_train_batches=cfg.train.limit_train_batches,
        callbacks=callbacks,
    )
    # train and validate
    trainer.fit(model, loaders["train"], loaders.get("val"))
    # test model
    if sizes[2]:
        trainer.test(test_dataloaders=loaders["test"], ckpt_path="best")
        diagnostic_plot(model, trainer)
        if cfg.train.save_prediction:
            torch.save(model.results[0], "test_predictions.pt")
            torch.save(model.results[1], "test_targets.pt")


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
