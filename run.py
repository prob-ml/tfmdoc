import hydra

from tfmdoc.preprocess import claims_pipeline


@hydra.main(config_path=".", config_name="config.yaml")
def main(cfg=None):
    claims_pipeline(
        data_dir=cfg.preprocess.data_dir,
        disease_codes=cfg.disease_codes.ald,
        length_range=(cfg.preprocess.min_length, cfg.preprocess.max_length),
        year_range=(cfg.preprocess.min_year, cfg.preprocess.max_year + 1),
        n_processed=cfg.preprocess.n_processed,
    )


if __name__ == "__main__":
    main()
