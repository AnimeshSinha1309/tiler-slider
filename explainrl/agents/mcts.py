import typing as ty

import numpy as np
import torch

from ..environment.state import GridState, Move
from ..environment.env import step


class MCTSAgent:

    class MCTSState:
        """
        State object representing the solution (boolean vector of swaps) as a MCTS node
        """

        def __init__(self, state, r_previous=0, parent_state=None, parent_action=None):
            """
            Initialize a new state
            """
            self.state: GridState = state
            self.parent_state, self.parent_action = parent_state, parent_action
            self.r_previous = r_previous
            self.n_value = torch.zeros(4)
            self.q_value = torch.zeros(4)
            self.child_states: ty.List[ty.Optional[MCTSAgent.MCTSState]] = [None for _ in range(4)]

        def update_q(self, reward, index):
            """
            Updates the q-value for the state
            :param reward: The obtained total reward from this state
            :param index: the index of the action chosen for which the reward was provided
            n_value is the number of times a node visited
            q_value is the q function
            n += 1, w += reward, q = w / n -> this is being implicitly computed using the weighted average
            """
            self.q_value[index] = (self.q_value[index] * self.n_value[index] + reward) / (self.n_value[index] + 1)
            self.n_value[index] += 1

        def select(self, model, c=1000) -> int:
            """
            Select one of the child actions based on UCT rule.

            :param model: The neural network used to make this approximation
            :param c: Exploration Exploitation tradeoff constant
            :return: int, the index of the chosen action
            """
            _value, priors = model(self.state)
            n_visits = torch.sum(self.n_value).item()
            uct = self.q_value + (priors * c * np.sqrt(n_visits + 1) / (self.n_value + 1))
            best_val = torch.max(uct)
            best_move_indices: torch.Tensor = torch.where(torch.eq(best_val, uct))[0]
            winner: int = np.random.choice(best_move_indices.numpy())
            return winner

        def expand(self, action: Move):
            """
            Expands out one child of the current state which has not been explored yet
            :param action: Move, the action from the current state to expand
            :return: GridState, the resulting state
            """
            next_state, reward, _done, _debug = step(self.state, action)
            self.child_states[action.value] = MCTSAgent.MCTSState(
                            next_state, r_previous=reward, parent_state=self, parent_action=action)
            return self.child_states[action.value]

        def rollout(self, model):
            """
            Uses a neural network to approximate the results from rolling out the current state,
            i.e. estimates the value of the state.
            :param model: The neural network used to make this approximation
            :return: float, value function of the current state
            """
            value, _policy = model(self.state)
            return value

        def backup(self, future_reward, gamma=0.95):
            """
            Backs up the value of each state (q-value and n-visits) for the entire ancestry
            of the currently visited node
            :param future_reward: float, reward expected in the future
            :param gamma: float, the discount factor
            :return: None
            """
            if self.parent_state is None:
                return
            else:
                total_reward = self.r_previous + gamma * future_reward
                self.parent_state.update_q(total_reward, self.parent_action.value)
                self.parent_state.backup(total_reward)

    """
    Monte Carlo Tree Search combiner object for evaluating the combination of moves
    that will form one step of the simulation.
    This at the moment does not look into the future steps, just calls an evaluator
    """

    def __init__(self, state, model):
        """
        Set's up the MCTS search agent object
        :param state: State to start searching from
        :param model: Function approximator for both value and policy
        """
        self.model = model
        self.root = MCTSAgent.MCTSState(state, self.model)

    def search(self, n_mcts):
        """
        Perform the MCTS search from the root
        TODO: Add exploration diagnostics, tree depth, etc.
        :param n_mcts: number of times to run for
        :return: None
        """
        for _ in range(n_mcts):
            mcts_state: MCTSAgent.MCTSState = self.root  # reset to root for new trace
            while True:
                action_index: int = mcts_state.select(self.model)
                if mcts_state.child_states[action_index] is not None:
                    mcts_state = mcts_state.child_states[action_index]
                    continue
                else:
                    mcts_state = mcts_state.expand(Move(action_index))
                    break
            total_reward = mcts_state.rollout(self.model)
            mcts_state.backup(total_reward)

    def act(self):
        """
        Get the best action
        :return: Move, the best action in current state
        """
        self.search(1000)
        action_idx = self.root.select(self.model)
        return Move(action_idx)

    def update(self, action):
        action_idx = action.value
        self.root = self.root.child_states[action_idx]
        return True
