#!/usr/bin/env python3
"""
Breadth-First Search (BFS) agent for solving Tiler-Slider puzzles.

This module implements a BFS solver that:
- Explores all possible game states systematically
- Uses state deduplication to avoid revisiting states
- Provides progress logging
- Returns optimal solution path
"""

import argparse
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

from explainrl.environment import GameState, TilerSliderEnv, ImageLoader


@dataclass
class StateNode:
    """
    Node in the BFS search tree.

    Attributes:
        state_hash: Hash of current tile positions for deduplication
        depth: Distance from initial state
        move: Move that led to this state (None for initial state)
        parent: Reference to parent node for path reconstruction
    """
    state_hash: Tuple[Tuple[int, int], ...]
    depth: int
    move: Optional[GameState.Move]
    parent: Optional['StateNode']


class BFSSolver:
    """
    Breadth-First Search solver for Tiler-Slider puzzles.

    This solver uses BFS to find the optimal (shortest) solution path.
    It maintains a cache of visited states based on tile positions to
    avoid exploring duplicate states.
    """

    def __init__(self, env: TilerSliderEnv, verbose: bool = True):
        """
        Initialize BFS solver.

        Args:
            env: TilerSliderEnv instance to solve
            verbose: If True, print progress logs
        """
        self.env = env
        self.verbose = verbose
        self.stats = {
            'states_explored': 0,
            'max_depth': 0,
            'queue_size': 0,
            'solution_length': 0,
            'time_elapsed': 0.0
        }

    def dump_state(self, game_state: GameState) -> Tuple[Tuple[int, int], ...]:
        """
        Serialize game state to a hashable representation.

        For state deduplication, we only care about tile positions.
        If multi_color is True, order matters; if False, we can use a set.

        Args:
            game_state: GameState to serialize

        Returns:
            Tuple of tile positions (hashable)
        """
        if game_state.multi_color:
            # Order matters - use tuple of positions
            return tuple(game_state.current_locations)
        else:
            # Order doesn't matter - use sorted tuple for consistent hashing
            return tuple(sorted(game_state.current_locations))

    def load_state(self, state_hash: Tuple[Tuple[int, int], ...]) -> List[Tuple[int, int]]:
        """
        Deserialize state hash back to tile positions.

        Args:
            state_hash: Serialized state

        Returns:
            List of tile positions
        """
        return list(state_hash)

    def solve(self) -> Optional[List[GameState.Move]]:
        """
        Solve the puzzle using BFS.

        Returns:
            List of moves to solve the puzzle, or None if no solution exists
        """
        start_time = time.time()

        # Reset environment to initial state
        self.env.reset()
        initial_state = self.env.state

        # Check if already solved
        if initial_state.is_won():
            if self.verbose:
                print("Puzzle already solved!")
            return []

        # Initialize BFS
        initial_hash = self.dump_state(initial_state)
        initial_node = StateNode(
            state_hash=initial_hash,
            depth=0,
            move=None,
            parent=None
        )

        queue = deque([initial_node])
        visited = {initial_hash: initial_node}  # Maps state_hash -> node

        # All possible moves
        all_moves = [GameState.Move.UP, GameState.Move.DOWN,
                     GameState.Move.LEFT, GameState.Move.RIGHT]

        if self.verbose:
            print(f"Starting BFS search...")
            print(f"Initial state: {initial_hash}")
            print(f"Multi-color mode: {initial_state.multi_color}")
            print(f"Number of tiles: {len(initial_state.current_locations)}")
            print(f"Target positions: {tuple(initial_state.target_locations)}")
            print("-" * 60)

        # BFS loop
        while queue:
            current_node = queue.popleft()
            self.stats['states_explored'] += 1
            self.stats['max_depth'] = max(self.stats['max_depth'], current_node.depth)
            self.stats['queue_size'] = len(queue)

            # Progress logging
            if self.verbose and self.stats['states_explored'] % 1000 == 0:
                elapsed = time.time() - start_time
                print(f"States explored: {self.stats['states_explored']:6d} | "
                      f"Max depth: {self.stats['max_depth']:3d} | "
                      f"Queue size: {self.stats['queue_size']:6d} | "
                      f"Time: {elapsed:.2f}s")

            # Reconstruct state from hash
            tile_positions = self.load_state(current_node.state_hash)

            # Try all 4 possible moves
            for move in all_moves:
                # Create a copy of the state and apply move
                new_state = initial_state.copy()
                new_state.current_locations = tile_positions.copy()
                is_won = new_state.move(move)

                # Get new state hash
                new_hash = self.dump_state(new_state)

                # Skip if state already visited
                if new_hash in visited:
                    continue

                # Create new node
                new_node = StateNode(
                    state_hash=new_hash,
                    depth=current_node.depth + 1,
                    move=move,
                    parent=current_node
                )

                # Add to visited set
                visited[new_hash] = new_node

                # Check if solved
                if is_won:
                    self.stats['time_elapsed'] = time.time() - start_time
                    self.stats['solution_length'] = new_node.depth
                    self.stats['max_depth'] = max(self.stats['max_depth'], new_node.depth)

                    if self.verbose:
                        print("-" * 60)
                        print("Solution found!")
                        print(f"Total states explored: {self.stats['states_explored']}")
                        print(f"Maximum depth reached: {self.stats['max_depth']}")
                        print(f"Solution length: {self.stats['solution_length']}")
                        print(f"Time elapsed: {self.stats['time_elapsed']:.2f}s")
                        print(f"States per second: {self.stats['states_explored'] / self.stats['time_elapsed']:.0f}")

                    # Reconstruct path
                    return self._reconstruct_path(new_node)

                # Add to queue for exploration
                queue.append(new_node)

        # No solution found
        self.stats['time_elapsed'] = time.time() - start_time
        if self.verbose:
            print("-" * 60)
            print("No solution found!")
            print(f"Total states explored: {self.stats['states_explored']}")
            print(f"Maximum depth reached: {self.stats['max_depth']}")
            print(f"Time elapsed: {self.stats['time_elapsed']:.2f}s")

        return None

    def _reconstruct_path(self, goal_node: StateNode) -> List[GameState.Move]:
        """
        Reconstruct solution path from goal node back to start.

        Args:
            goal_node: Final node in solution path

        Returns:
            List of moves from start to goal
        """
        path = []
        current = goal_node

        while current.parent is not None:
            path.append(current.move)
            current = current.parent

        # Reverse to get start -> goal order
        path.reverse()
        return path


def load_puzzle(puzzle_name: str) -> TilerSliderEnv:
    """
    Load a puzzle by name.

    Args:
        puzzle_name: Name of puzzle (e.g., 'puzzle_multi_003')

    Returns:
        TilerSliderEnv instance
    """
    # Save current directory
    original_dir = os.getcwd()

    try:
        # Change to data directory
        data_dir = os.path.join(os.path.dirname(__file__), '../../data')
        os.chdir(data_dir)

        # Load puzzle
        loader = ImageLoader()

        # Find puzzle index
        puzzle_idx = None
        for idx, item in enumerate(loader):
            if puzzle_name in item.name:
                puzzle_idx = idx
                break

        if puzzle_idx is None:
            raise ValueError(f"Puzzle '{puzzle_name}' not found")

        # Parse puzzle
        raw_data = loader[puzzle_idx]
        multi_color = 'multi' in puzzle_name.lower()
        level = ImageLoader.parse_puzzle_image(raw_data.puzzle_image, multiple_colors=multi_color)

        # Create environment
        env = TilerSliderEnv.from_level(level)

        return env

    finally:
        # Restore original directory
        os.chdir(original_dir)


def visualize_solution(env: TilerSliderEnv, moves: List[GameState.Move], delay: float = 0.5):
    """
    Visualize the solution by applying moves and printing states.

    Args:
        env: Environment to visualize
        moves: List of moves to apply
        delay: Delay between moves in seconds
    """
    from explainrl.environment import TextRender

    env.reset()
    renderer = TextRender(env.state)

    print("\nInitial state:")
    print(renderer.render())
    time.sleep(delay)

    for i, move in enumerate(moves, 1):
        env.step(move)
        print(f"\nAfter move {i}: {move.name}")
        print(renderer.render())
        time.sleep(delay)

    if env.state.is_won():
        print("\nPuzzle solved! ✓")


def main():
    """Main entry point for BFS solver."""
    parser = argparse.ArgumentParser(description='Solve Tiler-Slider puzzles using BFS')
    parser.add_argument('--level', type=str, required=True,
                       help='Puzzle name (e.g., puzzle_multi_003)')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize solution step-by-step')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress logs')

    args = parser.parse_args()

    # Load puzzle
    try:
        env = load_puzzle(args.level)
    except Exception as e:
        print(f"Error loading puzzle: {e}", file=sys.stderr)
        return 1

    # Solve puzzle
    solver = BFSSolver(env, verbose=not args.quiet)
    solution = solver.solve()

    if solution is None:
        print("No solution found!", file=sys.stderr)
        return 1

    # Print solution
    print("\nSolution:")
    print(" -> ".join(move.name for move in solution))

    # Visualize if requested
    if args.visualize:
        visualize_solution(env, solution)

    return 0


if __name__ == '__main__':
    sys.exit(main())
