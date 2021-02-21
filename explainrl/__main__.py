import argparse
import tqdm

from .environment.display import GridRender
from .environment.state import GridState
from .environment.env import step
from .models.dense import FeedForwardEvaluator
from .agents.mcts import MCTSAgent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', required=True, help='Input File Path')
    args = parser.parse_args()

    # GridRender.load(args.input_file)
    state, moves = GridState.load(args.input_file)
    model = FeedForwardEvaluator()

    for _trial in range(10):
        with tqdm.trange(100) as progress:
            for time in progress:
                agent = MCTSAgent(state, model)
                action = agent.act()
                next_state, reward, done, debug = step(state, action)
                if done:
                    print(f"Solved in {time} timesteps.")
                    break
                agent.update(action)
