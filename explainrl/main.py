import argparse

from explainrl.environment.display import GridRender


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', required=True, help='Input File Path')
    args = parser.parse_args()

    GridRender.load(args.input_file)
