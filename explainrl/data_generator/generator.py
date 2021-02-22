import random
from data_generator import config
import numpy as np

random.seed(config.RANDOM_SEED)

def move_value(move):
    if(move == "U"):
        return (-1,0)
    elif(move == "D"):
        return (1,0)
    elif(move == "L"):
        return (0,-1)
    elif(move == "R"):
        return (0,1)

def make_move(N, M, move, tiles, grid):
    delta_r, delta_c = move_value(move)
    flag = 0
    for idx, tile in enumerate(tiles):
        next_r, next_c = tile[0] + delta_r, tile[1] + delta_c
        if 0 <= next_r < N and 0 <= next_c < M and grid[next_r][next_c] == "." and (next_r, next_c) not in tiles:
            tile = tile[0] + delta_r, tile[1] + delta_c
            flag = 1
        tiles[idx] = tile
    return flag, tiles

def generate(output_file):
    fout = open(output_file, "w")

    N = random.randint(config.N_MIN, config.N_MAX)
    M = random.randint(config.M_MIN, config.M_MAX)
    fout.write("{} {}\n".format(N, M))

    PROB = random.choices(config.PROB_CHOICES, weights = config.PROB_WEIGHTS)[0]
    arr = np.full(((N, M)), ['.'], dtype=str)
    for i in range(N):
        for j in range(M):
            if random.random() < PROB:
                arr[i][j] = '#'

    np.savetxt(fout, arr, fmt='%s', delimiter='')

    no_sources = random.choices(config.SOURCES_CHOICES, weights = config.SOURCES_WEIGHTS)[0]
    sources = set()  # set taken so that there are no two sources at same coordinates
    while(len(sources) != no_sources):
        row = random.randint(0, N-1)
        col = random.randint(0, M-1)
        if arr[row][col] == '.':  # no source at obstacle
            sources.add((row, col))

    fout.write("{}\n".format(no_sources))

    no_moves = random.randint(config.MOVES_MIN, config.MOVES_MAX)

    seq = ['U', 'D', 'L', 'R']
    moves = ''.join(random.choices(
        seq, weights=None, cum_weights=None, k=no_moves))

    destinations = list(sources)
    for move in moves:
        for _ in range(max(N, M) + 1):
            flag, destinations = make_move(N, M, move, destinations, arr)
            if flag == 0:
                break

    for src, dest in zip(sources, destinations):
        fout.write("{} {} {} {}\n".format(
            src[0], src[1], dest[0], dest[1]))
    fout.close()
