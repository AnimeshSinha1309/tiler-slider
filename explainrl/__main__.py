import tqdm
import os
import argparse

from data_generator import generator
# from explainrl.environment.display import GridRender


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', required=True,
                        help='Output Data Path')
    parser.add_argument('--count', type = int, required=True,
                        help='Number of test files')
    args = parser.parse_args()

    for i in tqdm.tqdm(range(args.count)):
        path = os.path.join(args.output_path, str(i) + '.txt')
        generator.generate(path)

