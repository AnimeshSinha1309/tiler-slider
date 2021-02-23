import numpy as np


class GridState:
    def __init__(self, input_file):
        self.UP = (-1, 0)
        self.DOWN = (1, 0)
        self.LEFT = (0, -1)
        self.RIGHT = (0, 1)
        self.load(input_file)

    def load(self, input_file):
        file = open(input_file, 'r')
        # Get the Grid
        self.n, self.m = list(map(int, file.readline().strip().split()))
        self.grid = np.array(
            [[cell == "." for cell in file.readline().strip()] for _ in range(self.n)])
        k = int(file.readline().strip())
        self.tiles, self.targets = [], []
        for _ in range(k):
            line = list(map(int, file.readline().strip().split()))
            self.tiles.append((line[0], line[1]))
            self.targets.append((line[2], line[3]))
        file.close()

    def move_value(self, direction):
        if direction == 'U':
            return self.UP
        elif direction == 'D':
            return self.DOWN
        elif direction == 'L':
            return self.LEFT
        elif direction == 'R':
            return self.RIGHT

    def move(self, move):
        delta_r, delta_c = self.move_value(move)
        flag = 0
        for _ in range(max(self.n, self.m) + 1):
            for idx, tile in enumerate(self.tiles):
                next_r, next_c = tile[0] + delta_r, tile[1] + delta_c
                if 0 <= next_r < self.n and 0 <= next_c < self.m and \
                        self.grid[next_r, next_c] and (next_r, next_c) not in self.tiles:
                    flag = 1
                    tile = tile[0] + delta_r, tile[1] + delta_c
                self.tiles[idx] = tile
            # no change in any tile
            if flag == 0:
                break

    def __str__(self):
        labels = np.full(shape=self.grid.shape, fill_value='.')
        for x, row in enumerate(self.grid):
            for y, cell in enumerate(row):
                if not self.grid[x, y]:
                    labels[x, y] = '#'
        for x, y in self.tiles:
            labels[x, y] = 'a'
        for x, y in self.targets:
            labels[x, y] = 'A' if labels[x, y] != '.' else '1'
        return "\n".join(list(map(lambda line: "".join(line), labels)))
