import os
import argparse
import pandas as pd

from environment.state import GridState


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', required=True,
                        help='Path for folder containing dataset')
    args = parser.parse_args()

    for file in sorted(os.listdir(args.dataset_path)):
        path = os.path.join(args.dataset_path, file)
        obj = GridState(path)
        moves = "ULULDRURDL"
        for direction in moves:
            obj.move(direction)
            print(obj.tiles, obj.targets)
            if obj.tiles == obj.targets:
                break
                print("Puzzle Solved")
        break
    #     df = df.append({'filename': file, 'moves': moves}, ignore_index = False)
    # df = pd.DataFrame(columns={'filename', 'moves'})
    # print(df.head())
