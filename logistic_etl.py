import argparse

from haruspex import SampleGenerator


def main():
    parser = argparse.ArgumentParser("ETL for logistic liver disease prediction")
    parser.add_argument("--data_dir", type=str, default="/nfs/turbo/lsa-regier/OPTUM2/")
    parser.add_argument("--disease", type=str, default="ald")
    parser.add_argument("--skip_labs", action="store_true")

    args = parser.parse_args()

    sg = SampleGenerator(args.data_dir, args.skip_labs, args.disease)

    sg.run()


if __name__ == "__main__":
    main()
