import copy
from enum import Enum
import typing as ty

import numpy as np


class Move(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    @classmethod
    def to_tuple(cls, direction):
        if direction == cls.UP:
            return -1, 0
        elif direction == cls.DOWN:
            return 1, 0
        elif direction == cls.LEFT:
            return 0, -1
        elif direction == cls.RIGHT:
            return 0, 1

    @classmethod
    def from_char(cls, direction):
        if direction == 'U':
            return cls.UP
        elif direction == 'D':
            return cls.DOWN
        elif direction == 'L':
            return cls.LEFT
        elif direction == 'R':
            return cls.RIGHT

    @classmethod
    def to_char(cls, direction):
        if direction == cls.UP:
            return 'U'
        elif direction == cls.DOWN:
            return 'D'
        elif direction == cls.LEFT:
            return 'L'
        elif direction == cls.RIGHT:
            return 'R'


class GridState:

    def __init__(self, n, m, grid, tiles, targets):
        self.n, self.m = n, m
        self.grid = grid
        self.tiles = tiles
        self.targets = targets

    def __copy__(self):
        return GridState(self.n, self.m, np.copy(self.grid),
                         copy.copy(self.tiles), copy.copy(self.targets))

    @classmethod
    def load(cls, input_file):
        file = open(input_file, 'r')
        # Get the Grid
        n, m = list(map(int, file.readline().strip().split()))
        grid = np.array(
            [[cell == "." for cell in file.readline().strip()] for _ in range(n)])
        k = int(file.readline().strip())
        tiles, targets = [], []
        for _ in range(k):
            line = list(map(int, file.readline().strip().split()))
            tiles.append((line[0], line[1]))
            targets.append((line[2], line[3]))
            # TODO: Add the 5th parameter for color, or make 2 modes default
        state = GridState(n, m, grid, tiles, targets)
        moves = list(map(Move.from_char, file.readline().strip('\n')))
        file.close()
        return state, moves

    def move(self, move: Move) -> bool:
        delta_r, delta_c = Move.to_tuple(move)
        flag = False
        for _ in range(max(self.grid.shape)):
            for idx, tile in enumerate(self.tiles):
                next_r, next_c = tile[0] + delta_r, tile[1] + delta_c
                if 0 <= next_r < self.n and 0 <= next_c < self.m and \
                        self.grid[next_r, next_c] and (next_r, next_c) not in self.tiles:
                    flag = True
                    tile = tile[0] + delta_r, tile[1] + delta_c
                self.tiles[idx] = tile
        return flag

    def available_actions(self) -> ty.Dict[Move, bool]:
        actions = {move: False for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]}
        for action in actions.keys():
            for tile in self.tiles:
                next_r, next_c = tile + actions
                if 0 <= next_r < self.n and 0 <= next_c < self.m and \
                        self.grid[next_r, next_c] and (next_r, next_c) not in self.tiles:
                    actions[action] = True
                    break
        return actions

    def done(self) -> bool:
        return self.tiles == self.targets

    @property
    def shape(self) -> ty.Tuple[int, int]:
        return self.grid.shape

    def __str__(self) -> str:
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
