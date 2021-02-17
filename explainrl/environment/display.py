import sys
import typing

import pygame

from explainrl.environment import config
from explainrl.environment.state import GridState


class GridRender(GridState):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pygame.init()
        window_width: int = self.m * config.BLOCK_SIZE
        window_height: int = self.n * config.BLOCK_SIZE
        window_title: str = "Tiler Slider"
        self.screen = pygame.display.set_mode((window_height, window_width))
        pygame.display.set_caption(window_title)

    def render(self):
        for x in range(self.m):
            for y in range(self.n):
                rect = pygame.Rect(y * config.BLOCK_SIZE, x * config.BLOCK_SIZE, config.BLOCK_SIZE, config.BLOCK_SIZE)
                # assigning colour based on grid value
                if (x, y) in self.tiles and (x, y) in self.targets:
                    color = config.COLORS["COMBINED"]
                elif (x, y) in self.tiles:
                    color = config.COLORS["TILE"]
                elif (x, y) in self.targets:
                    color = config.COLORS["TARGET"]
                elif self.grid[x, y]:
                    color = config.COLORS["SPACE"]
                else:
                    color = config.COLORS["OBSTACLE"]
                pygame.draw.rect(self.screen, color, rect, 0, border_radius=0)

    def update(self):
        self.screen.fill(config.COLORS["SCREEN"])
        self.render()
        pygame.display.update()
        pygame.time.wait(config.WAIT_TIME)

    @staticmethod
    def respond():
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                pygame.quit()
                sys.exit()

    @classmethod
    def load(cls, input_file):
        state: GridState
        moves: typing.List[typing.Tuple[int, int]]
        state, moves = super().load(input_file)

        state: GridRender = GridRender(state.n, state.m, state.grid, state.tiles, state.targets)
        state.render()
        print(state, "\n")

        for move in moves:
            print(state, "\n")
            state.move(move)
            state.update()
            state.respond()
