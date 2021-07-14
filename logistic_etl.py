import argparse

from haruspex import FeaturesBuilder, SampleGenerator


def main():
    parser = argparse.ArgumentParser("ETL for logistic liver disease prediction")
    parser.add_argument("--data_dir", type=str, default="/nfs/turbo/lsa-regier/OPTUM2/")
    parser.add_argument("--disease", type=str, default="ald")
    parser.add_argument("--generate_samples", action="store_true")
    parser.add_argument("--skip_labs", action="store_true")
    parser.add_argument("--skip_diags", action="store_true")

    args = parser.parse_args()

    if args.generate_samples:
        sg = SampleGenerator(args.data_dir, args.skip_labs, args.disease)
        sg.run()

    fb = FeaturesBuilder(args.data_dir, args.skip_diags, args.disease)
    fb.run()


if __name__ == "__main__":
    main()
