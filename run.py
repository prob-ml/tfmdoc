import argparse

from tfmdoc.preprocess import claims_pipeline


def main():
    parser = argparse.ArgumentParser("Run the tfm-doc model")
    parser.add_argument(
        "--data_dir", type=str, default="/nfs/turbo/lsa-regier/OPTUM2/test_data/"
    )

    args = parser.parse_args()

    claims_pipeline(data_dir=args.data_dir)


if __name__ == "__main__":
    main()
