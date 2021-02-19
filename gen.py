import random
import numpy as np
N = 5
M = 8
PROB = 0.3  # probability of creating obstacle
MAX_SOURCES = 3
MAX_MOVES = 10
MAX_COLORS = 3


def make_moves(source, grid, moves):
    # TODO: Animesh
    destination = source  # temporarily done
    return destination


if __name__ == "__main__":
    fout = open("test/output.txt", "w")
    fout.write("{} {}\n".format(N, M))

    arr = np.full(((N, M)), ['.'], dtype=str)
    for i in range(N):
        for j in range(M):
            if random.random() < PROB:
                arr[i][j] = '#'

    np.savetxt(fout, arr, fmt='%s', delimiter='')

    no_sources = random.randint(1, MAX_SOURCES)
    sources = set()  # set taken so that there are no two sources at same coordinates
    while(len(sources) != no_sources):
        y = random.randint(0, N-1)
        x = random.randint(0, M-1)
        if arr[y][x] == '.':  # no source at obstacle
            sources.add((x, y))

    fout.write("{}\n".format(no_sources))

    no_moves = random.randint(1, MAX_MOVES)

    seq = ['U', 'D', 'L', 'R']
    moves = ''.join(random.choices(
        seq, weights=None, cum_weights=None, k=no_moves))

    destination = []
    for src in sources:
        destination.append(make_moves(src, arr, moves))

    # print(destination)
    for src, dest in zip(sources, destination):
        color = random.randint(0, MAX_COLORS - 1)
        fout.write("{} {} {} {} {}\n".format(
            src[0], src[1], dest[0], dest[1], color))
    fout.close()
