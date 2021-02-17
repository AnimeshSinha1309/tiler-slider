import sys
import pygame
import argparse
import numpy as np
from config import *

global N, M, SCREEN


def read_input(input_file):
    file = open(input_file, 'r')

    line = file.readline()
    N, M = int(line[0]), int(line[2])

    grid = []
    for i in range(N):
        line = file.readline().strip('\n')
        grid.append(list(line))
    moves = list(file.readline().strip('\n'))
    return N, M, np.asarray(grid), moves


def drawGrid(grid):
    sources = []
    for x in range(M):
        for y in range(N):
            rect = pygame.Rect(x*BLOCK_SIZE, y*BLOCK_SIZE,
                               BLOCK_SIZE, BLOCK_SIZE)
            # assigning colour based on grid value
            if grid[y][x] == OBSTACLE:
                color = OBSTACLE_COLOUR
            elif grid[y][x] == SOURCE:
                sources.append([x, y])
                color = SOURCE_COLOUR
            elif grid[y][x] == DESTINATION:
                color = DESTINATION_COLOUR
            elif grid[y][x] == COMBINED:
                sources.append([x, y])
                color = COMBINED_COLOUR
            elif grid[y][x] == SPACE:
                color = SPACE_COLOUR
            pygame.draw.rect(SCREEN, color, rect, 0, BLOCK_SIZE//10)
    return np.asarray(sources)


def update_cnt(grid):
    """
    store count of max no of cells source can move to up, left,
    right, down after one move
    """

    N, M = grid.shape
    # update up count
    up_count = np.zeros((N, M), dtype=int)
    for x in range(M):
        last_obstacle = -1
        for y in range(N):
            if grid[y][x] == OBSTACLE:
                last_obstacle = y
            up_count[y][x] = abs(last_obstacle - y) - 1

    # update down count
    down_count = np.zeros((N, M), dtype=int)
    for x in range(M):
        last_obstacle = N
        for y in range(N-1, -1, -1):
            if grid[y][x] == OBSTACLE:
                last_obstacle = y
            down_count[y][x] = abs(last_obstacle - y) - 1

    # update left count
    left_count = np.zeros((N, M), dtype=int)
    for y in range(N):
        last_obstacle = -1
        for x in range(M):
            if grid[y][x] == OBSTACLE:
                last_obstacle = x
            left_count[y][x] = abs(last_obstacle - x) - 1

    # update right count
    right_count = np.zeros((N, M), dtype=int)
    for y in range(N):
        last_obstacle = M
        for x in range(M-1, -1, -1):
            if grid[y][x] == OBSTACLE:
                last_obstacle = x
            right_count[y][x] = abs(last_obstacle - x) - 1

    return up_count, down_count, left_count, right_count


def make_move(sources, move, grid, up_count, down_count, left_count, right_count):
    for cord in sources:
        x = cord[0]
        y = cord[1]
        final_x = x
        final_y = y

        if grid[y][x] == COMBINED:
            grid[y][x] = DESTINATION
        else:
            grid[y][x] = SPACE

        if move == 'L':
            final_x = x - left_count[y][x]
        elif move == 'R':
            final_x = x + right_count[y][x]
        elif move == 'U':
            final_y = y - up_count[y][x]
        elif move == 'D':
            final_y = y + down_count[y][x]

        if grid[final_y][final_x] == SPACE:
            grid[final_y][final_x] = SOURCE
        elif grid[final_y][final_x] == DESTINATION:
            grid[final_y][final_x] = COMBINED

    return grid


def update_grid(grid):
    global SCREEN
    SCREEN.fill(SCREEN_COLOUR)
    sources = drawGrid(grid)
    pygame.display.update()
    pygame.time.wait(WAIT_TIME)
    return sources


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_file', required=True, help='Input File Path')
    args = parser.parse_args()

    # read input
    N, M, grid, moves = read_input(args.input_file)

    up_count, down_count, left_count, right_count = update_cnt(grid)

    # creating window
    global SCREEN
    pygame.init()
    WINDOW_WIDTH = M*BLOCK_SIZE
    WINDOW_HEIGHT = N*BLOCK_SIZE
    SCREEN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption(WINDOW_TITLE)

    # render initial grid
    sources = update_grid(grid)

    for move in moves:
        # update and render grid after each move
        grid = make_move(sources, move, grid, up_count,
                         down_count, left_count, right_count)
        sources = update_grid(grid)

        # for quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                pygame.quit()
                sys.exit()
