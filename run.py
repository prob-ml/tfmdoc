import hydra

from tfmdoc.preprocess import claims_pipeline


@hydra.main(config_path=".", config_name="config.yaml")
def main(cfg=None):
    claims_pipeline(data_dir=cfg.data_dir, min_length=16, max_length=512)


if __name__ == "__main__":
    main()
