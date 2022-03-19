import os

from hydra import compose, initialize

from tfmdoc import datasets
from tfmdoc.preprocess import ClaimsPipeline, DiagnosisPipeline


def test_bert_pipeline():
    with initialize(config_path=".."):
        cfg = compose(
            config_name="config",
            overrides=[
                "preprocess.data_dir=tests/test_data/",
                "preprocess.output_dir=tests/test_data/test_pipeline/",
                "preprocess.filename=preprocessed",
            ],
        )
        cpl = ClaimsPipeline(
            cfg.preprocess.data_dir,
            cfg.preprocess.output_dir,
            output_name=cfg.preprocess.filename,
            test=True,
        )
        cpl.run()
        preprocess_dir = "tests/test_data/test_pipeline/"
        os.remove(preprocess_dir + "preprocessed.hdf5")
        os.rmdir(preprocess_dir)


def test_diag_pipeline():
    with initialize(config_path=".."):
        cfg = compose(
            config_name="config",
            overrides=[
                "disease_codes.ald=['7231']",
                "preprocess.data_dir=tests/test_data/",
                "preprocess.output_dir=tests/test_data/test_pipeline/",
                "preprocess.filename=preprocessed",
            ],
        )
        cpl = DiagnosisPipeline(
            cfg.preprocess.data_dir,
            cfg.preprocess.output_dir,
            cfg.disease_codes.ald,
            output_name=cfg.preprocess.filename,
            test=True,
        )
        cpl.run()
        preprocess_dir = "tests/test_data/test_pipeline/"
        torch_dataset = datasets.DiagnosisDataset(preprocess_dir)
        assert torch_dataset.offsets[-1] == torch_dataset.records.shape[0]
        t, v, w, x, y = torch_dataset[7]
        assert len(x) == torch_dataset.offsets[7] - torch_dataset.offsets[6]
        assert y.item() in {0, 1}
        assert w.shape[0] == 2
        assert v.max().item() == 1
        assert t.shape[0] == x.shape[0]
        os.remove(preprocess_dir + "preprocessed.hdf5")
        os.rmdir(preprocess_dir)


def test_early_etl():

    with initialize(config_path=".."):
        cfg = compose(
            config_name="config",
            overrides=[
                "model.d_model=32",
                "model.n_blocks=1",
                "disease_codes.ald=['7231']",
                "preprocess.data_dir=tests/test_data/",
                "preprocess.output_dir=tests/test_data/test_lightning/",
                "preprocess.filename=preprocessed",
                "preprocess.mode=early_detection",
            ],
        )

        cpl = DiagnosisPipeline(
            cfg.preprocess.data_dir,
            cfg.preprocess.output_dir,
            cfg.disease_codes.ald,
            mode=cfg.preprocess.mode,
            output_name=cfg.preprocess.filename,
            test=True,
        )
        cpl.run()

        preprocess_dir = "tests/test_data/test_lightning/"
        dataset = datasets.EarlyDetectionDataset(
            preprocess_dir, late_cutoff=10, early_cutoff=30
        )
        assert len(dataset)
        assert len(dataset[0]) == 2
        assert len(dataset[0][0]) == 5
        assert dataset[0][1][4] - dataset[0][0][4] == 1
        os.remove(preprocess_dir + "preprocessed.hdf5")
        os.rmdir(preprocess_dir)
