import copy

from .state import GridState, Move


def step(state: GridState, action: Move):
    next_state: GridState = copy.copy(state)
    next_state.move(action)
    done = next_state.done()
    reward = 10000 if done else 0
    return next_state, reward, done, None
