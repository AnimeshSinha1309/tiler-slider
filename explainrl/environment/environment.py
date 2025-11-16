"""
Environment wrapper for Tiler-Slider puzzle game.

This module provides a clean interface for interacting with the game state.
Rendering and reward calculation are handled separately.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
from .state import GameState
from .dataloader import ImageLoader


class TilerSliderEnv:
    """
    Environment for Tiler-Slider puzzle.

    Provides:
    - reset(): Initialize a new episode
    - step(move): Execute a move and get next state
    - close(): Cleanup resources

    Action Space: GameState.Move enum
        - GameState.Move.UP
        - GameState.Move.DOWN
        - GameState.Move.LEFT
        - GameState.Move.RIGHT

    Observation Space: Box(size, size, 3)
        3-channel image representing blocked cells, tiles, and targets
    """

    def __init__(self, size: int = None, blocked_locations: list = None,
                 initial_locations: list = None, target_locations: list = None,
                 multi_color: bool = False, max_steps: int = 100):
        """
        Initialize the environment.

        Args:
            size: Board size (creates size x size grid)
            blocked_locations: List of (row, col) blocked positions
            initial_locations: List of (row, col) initial tile positions
            target_locations: List of (row, col) target positions
            multi_color: Whether tile order matters for matching
            max_steps: Maximum steps before episode terminates
        """
        self.size = size
        self.blocked_locations = blocked_locations or []
        self.initial_locations = initial_locations or []
        self.target_locations = target_locations or []
        self.multi_color = multi_color
        self.max_steps = max_steps

        self.state: Optional[GameState] = None
        self.step_count = 0
        self.done = False

        # Action and observation spaces
        self.observation_shape = (size, size, 3) if size else None

    @classmethod
    def from_level(cls, level: ImageLoader.ImageProcessed, max_steps: int = 100):
        """
        Create environment from a parsed level.

        Args:
            level: ImageProcessed object from ImageLoader
            max_steps: Maximum steps before episode terminates

        Returns:
            TilerSliderEnv instance
        """
        return cls(
            size=level.size,
            blocked_locations=level.blocked_locations,
            initial_locations=level.initial_locations,
            target_locations=level.target_locations,
            multi_color=level.multiple_colors,
            max_steps=max_steps
        )

    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.

        Returns:
            Initial observation (state array)
        """
        self.state = GameState(
            self.size,
            self.blocked_locations,
            self.initial_locations,
            self.target_locations,
            self.multi_color
        )
        self.step_count = 0
        self.done = False
        return self.state.get_state_array()

    def step(self, move: GameState.Move) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            move: GameState.Move enum value (UP, DOWN, LEFT, RIGHT)

        Returns:
            Tuple of (observation, done, info):
                - observation: Current state as numpy array
                - done: Whether episode has terminated
                - info: Dictionary with additional information
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        if not isinstance(move, GameState.Move):
            raise TypeError(f"Action must be a GameState.Move enum, got {type(move)}")

        # Store previous state to detect invalid moves
        prev_locations = self.state.current_locations.copy()

        # Execute move
        is_won = self.state.move(move)

        # Build info dictionary
        info = {
            'is_won': is_won,
            'step_count': self.step_count,
            'invalid_move': prev_locations == self.state.current_locations
        }

        # Check win condition
        if is_won:
            self.done = True
            info['success'] = True

        # Check step limit
        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.done = True
            info['timeout'] = True

        return self.state.get_state_array(), self.done, info

    def close(self):
        """Clean up resources."""
        self.state = None

    def get_valid_moves(self) -> list[GameState.Move]:
        """
        Get list of moves that would change the state.

        Returns:
            List of valid Move enum values
        """
        if self.state is None:
            return []

        valid_moves = []
        current_locations = self.state.current_locations.copy()

        for move in GameState.Move:
            # Create temporary state to test move
            temp_state = self.state.copy()
            temp_state.move(move)

            # If state changed, move is valid
            if temp_state.current_locations != current_locations:
                valid_moves.append(move)

        return valid_moves

    def get_info(self) -> Dict[str, Any]:
        """
        Get current environment information.

        Returns:
            Dictionary with environment state information
        """
        if self.state is None:
            return {'initialized': False}

        return {
            'initialized': True,
            'size': self.size,
            'step_count': self.step_count,
            'max_steps': self.max_steps,
            'done': self.done,
            'is_won': self.state.is_won(),
            'num_tiles': len(self.state.current_locations),
            'num_targets': len(self.state.target_locations),
            'multi_color': self.multi_color,
            'valid_moves': self.get_valid_moves()
        }


class TilerSliderEnvFactory:
    """
    Factory for creating Tiler-Slider environments from various sources.
    """

    @staticmethod
    def create_simple_env(size: int = 5, num_tiles: int = 2,
                         num_obstacles: int = 3, seed: int = None) -> TilerSliderEnv:
        """
        Create a simple random environment for testing.

        Args:
            size: Board size
            num_tiles: Number of tiles/targets
            num_obstacles: Number of blocked cells
            seed: Random seed for reproducibility

        Returns:
            TilerSliderEnv instance
        """
        if seed is not None:
            np.random.seed(seed)

        # Generate random positions
        all_positions = [(i, j) for i in range(size) for j in range(size)]
        np.random.shuffle(all_positions)

        blocked = all_positions[:num_obstacles]
        initial = all_positions[num_obstacles:num_obstacles + num_tiles]
        targets = all_positions[num_obstacles + num_tiles:num_obstacles + 2 * num_tiles]

        return TilerSliderEnv(
            size=size,
            blocked_locations=blocked,
            initial_locations=initial,
            target_locations=targets,
            multi_color=False
        )

    @staticmethod
    def create_from_string(board_str: str, multi_color: bool = False) -> TilerSliderEnv:
        """
        Create environment from string representation.

        Format:
            'X' = blocked cell
            '.' = empty cell
            'a', 'b', 'c', ... = tiles
            'A', 'B', 'C', ... = targets

        Args:
            board_str: String representation of the board
            multi_color: Whether to use multi-color mode

        Returns:
            TilerSliderEnv instance
        """
        lines = [line for line in board_str.strip().split('\n') if line.strip()]
        size = len(lines)

        blocked = []
        initial = []
        targets = []

        for i, line in enumerate(lines):
            for j, char in enumerate(line.strip()):
                if char == 'X':
                    blocked.append((i, j))
                elif char.islower():
                    # Tile position
                    tile_idx = ord(char) - ord('a')
                    while len(initial) <= tile_idx:
                        initial.append(None)
                    initial[tile_idx] = (i, j)
                elif char.isupper() and char != 'X':
                    # Target position
                    target_idx = ord(char) - ord('A')
                    while len(targets) <= target_idx:
                        targets.append(None)
                    targets[target_idx] = (i, j)

        # Remove None placeholders
        initial = [pos for pos in initial if pos is not None]
        targets = [pos for pos in targets if pos is not None]

        return TilerSliderEnv(
            size=size,
            blocked_locations=blocked,
            initial_locations=initial,
            target_locations=targets,
            multi_color=multi_color
        )
