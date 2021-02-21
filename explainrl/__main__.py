import argparse
import tqdm

from .environment.display import GridRender
from .environment.state import GridState, Move
from .environment.env import step
from .models.dense import FeedForwardEvaluator
from .agents.mcts import MCTSAgent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', required=True, help='Input File Path')
    args = parser.parse_args()

    # GridRender.load(args.input_file)
    model = FeedForwardEvaluator((3, 3)) # state.shape

    for _trial in range(10):
        state, moves = GridState.load(args.input_file)
        result = []
        with tqdm.trange(1, 101) as progress:
            for time in progress:
                agent = MCTSAgent(state, model)
                action = agent.act()
                result.append(action)
                if time % 10 == 9:
                    agent.train()
                next_state, reward, done, debug = step(state, action)
                if done:
                    progress.set_postfix(steps=time,
                                         moves="".join(list(map(Move.to_char, result))))
                    break
                state = next_state
                agent.update(action)
