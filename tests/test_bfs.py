"""
Unit tests for BFS solver.

Tests cover:
- State serialization (dump/load)
- BFS solving on simple puzzles
- State deduplication
- Solution correctness
- Edge cases (already solved, no solution)
"""

import pytest
import os
from explainrl.environment import TilerSliderEnv, TilerSliderEnvFactory, GameState
from explainrl.agents import BFSSolver


class TestStateSerialization:
    """Test state dump and load methods."""

    def test_dump_state_multi_color(self):
        """Test state dumping for multi-color puzzles."""
        env = TilerSliderEnv(
            size=3,
            blocked_locations=[],
            initial_locations=[(0, 0), (1, 1), (2, 2)],
            target_locations=[(2, 2), (1, 1), (0, 0)],
            multi_color=True
        )
        env.reset()
        solver = BFSSolver(env, verbose=False)

        state_hash = solver.dump_state(env.state)

        # Multi-color: order matters, should be a tuple
        assert isinstance(state_hash, tuple)
        assert state_hash == ((0, 0), (1, 1), (2, 2))

    def test_dump_state_single_color(self):
        """Test state dumping for single-color puzzles."""
        env = TilerSliderEnv(
            size=3,
            blocked_locations=[],
            initial_locations=[(0, 0), (2, 2), (1, 1)],
            target_locations=[(2, 2), (1, 1), (0, 0)],
            multi_color=False
        )
        env.reset()
        solver = BFSSolver(env, verbose=False)

        state_hash = solver.dump_state(env.state)

        # Single-color: order doesn't matter, should be sorted
        assert isinstance(state_hash, tuple)
        assert state_hash == ((0, 0), (1, 1), (2, 2))

    def test_load_state(self):
        """Test state loading."""
        env = TilerSliderEnv(size=3)
        solver = BFSSolver(env, verbose=False)

        state_hash = ((0, 0), (1, 1), (2, 2))
        positions = solver.load_state(state_hash)

        assert positions == [(0, 0), (1, 1), (2, 2)]


class TestBFSSolverSimple:
    """Test BFS solver on simple puzzles."""

    def test_already_solved(self):
        """Test puzzle that's already solved."""
        env = TilerSliderEnv(
            size=3,
            blocked_locations=[],
            initial_locations=[(0, 0)],
            target_locations=[(0, 0)],
            multi_color=False
        )
        solver = BFSSolver(env, verbose=False)

        solution = solver.solve()

        assert solution == []
        assert solver.stats['states_explored'] == 0
        assert solver.stats['solution_length'] == 0

    def test_one_move_solution(self):
        """Test puzzle solvable in one move."""
        # Tile at (1, 0), target at (0, 0), just move UP
        env = TilerSliderEnv(
            size=3,
            blocked_locations=[],
            initial_locations=[(1, 0)],
            target_locations=[(0, 0)],
            multi_color=False
        )
        solver = BFSSolver(env, verbose=False)

        solution = solver.solve()

        assert solution is not None
        assert len(solution) == 1
        assert solution[0] == GameState.Move.UP
        assert solver.stats['solution_length'] == 1

    def test_two_move_solution(self):
        """Test puzzle requiring two moves."""
        # Tile at (2, 2), target at (0, 0), move UP then LEFT
        env = TilerSliderEnv(
            size=3,
            blocked_locations=[],
            initial_locations=[(2, 2)],
            target_locations=[(0, 0)],
            multi_color=False
        )
        solver = BFSSolver(env, verbose=False)

        solution = solver.solve()

        assert solution is not None
        assert len(solution) == 2
        # Could be UP,LEFT or LEFT,UP
        assert set(solution) == {GameState.Move.UP, GameState.Move.LEFT}
        assert solver.stats['solution_length'] == 2

    def test_blocked_cell_navigation(self):
        """Test navigation around blocked cells."""
        # Test case:
        # a . .
        # X . .
        # . . A
        # Need to go RIGHT then DOWN then DOWN
        env = TilerSliderEnv(
            size=3,
            blocked_locations=[(1, 0)],
            initial_locations=[(0, 0)],
            target_locations=[(2, 2)],
            multi_color=False
        )
        solver = BFSSolver(env, verbose=False)

        solution = solver.solve()

        assert solution is not None
        assert len(solution) >= 2  # Optimal is 2 moves: RIGHT, DOWN

    def test_multi_tile_puzzle(self):
        """Test puzzle with multiple tiles."""
        # Simple 2-tile puzzle
        env = TilerSliderEnv(
            size=4,
            blocked_locations=[],
            initial_locations=[(0, 0), (0, 1)],
            target_locations=[(3, 0), (3, 1)],
            multi_color=False
        )
        solver = BFSSolver(env, verbose=False)

        solution = solver.solve()

        assert solution is not None
        # Both tiles need to move DOWN to row 3
        assert GameState.Move.DOWN in solution


class TestBFSSolverCorrectness:
    """Test that BFS solutions are correct."""

    def test_solution_validity(self):
        """Test that solution actually solves the puzzle."""
        # Use a simple, definitely solvable puzzle
        env = TilerSliderEnv(
            size=4,
            blocked_locations=[],
            initial_locations=[(0, 0), (0, 1)],
            target_locations=[(3, 0), (3, 1)],
            multi_color=False
        )
        solver = BFSSolver(env, verbose=False)

        solution = solver.solve()

        assert solution is not None

        # Verify solution by applying moves
        env.reset()
        for move in solution:
            env.step(move)

        assert env.state.is_won()

    def test_optimal_solution(self):
        """Test that BFS finds optimal (shortest) solution."""
        # Simple case where optimal solution is clear
        env = TilerSliderEnv(
            size=4,
            blocked_locations=[],
            initial_locations=[(3, 3)],
            target_locations=[(0, 0)],
            multi_color=False
        )
        solver = BFSSolver(env, verbose=False)

        solution = solver.solve()

        assert solution is not None
        # Optimal is 2 moves: UP and LEFT (or LEFT and UP)
        assert len(solution) == 2


class TestBFSSolverStatistics:
    """Test BFS statistics tracking."""

    def test_stats_tracking(self):
        """Test that solver tracks statistics correctly."""
        env = TilerSliderEnv(
            size=3,
            blocked_locations=[],
            initial_locations=[(2, 2)],
            target_locations=[(0, 0)],
            multi_color=False
        )
        solver = BFSSolver(env, verbose=False)

        solution = solver.solve()

        assert solution is not None
        assert solver.stats['states_explored'] > 0
        assert solver.stats['max_depth'] >= len(solution)
        assert solver.stats['solution_length'] == len(solution)
        assert solver.stats['time_elapsed'] > 0


class TestBFSSolverDeduplication:
    """Test state deduplication."""

    def test_state_deduplication_single_color(self):
        """Test that equivalent states are deduplicated in single-color mode."""
        # Two tiles that can swap positions - should recognize as same state
        env = TilerSliderEnv(
            size=3,
            blocked_locations=[],
            initial_locations=[(0, 0), (0, 1)],
            target_locations=[(2, 0), (2, 1)],
            multi_color=False
        )
        env.reset()
        solver = BFSSolver(env, verbose=False)

        # Create two states with tiles in different order
        state1 = env.state.copy()
        state1.current_locations = [(0, 0), (0, 1)]

        state2 = env.state.copy()
        state2.current_locations = [(0, 1), (0, 0)]

        hash1 = solver.dump_state(state1)
        hash2 = solver.dump_state(state2)

        # In single-color mode, these should hash to same value
        assert hash1 == hash2

    def test_state_deduplication_multi_color(self):
        """Test that different orderings are NOT deduplicated in multi-color mode."""
        env = TilerSliderEnv(
            size=3,
            blocked_locations=[],
            initial_locations=[(0, 0), (0, 1)],
            target_locations=[(2, 0), (2, 1)],
            multi_color=True
        )
        env.reset()
        solver = BFSSolver(env, verbose=False)

        # Create two states with tiles in different order
        state1 = env.state.copy()
        state1.current_locations = [(0, 0), (0, 1)]

        state2 = env.state.copy()
        state2.current_locations = [(0, 1), (0, 0)]

        hash1 = solver.dump_state(state1)
        hash2 = solver.dump_state(state2)

        # In multi-color mode, these should hash to different values
        assert hash1 != hash2


class TestBFSRealPuzzles:
    """Test BFS on real puzzle files."""

    @pytest.mark.skipif(not os.path.exists('data/puzzle_multi_003.jpg'),
                       reason="Puzzle file not found")
    def test_puzzle_multi_003(self):
        """Test BFS on puzzle_multi_003."""
        from explainrl.agents.bfs import load_puzzle

        env = load_puzzle('puzzle_multi_003')
        solver = BFSSolver(env, verbose=False)

        solution = solver.solve()

        # Should find a solution
        assert solution is not None
        assert len(solution) > 0

        # Verify solution works
        env.reset()
        for move in solution:
            env.step(move)

        assert env.state.is_won()

    @pytest.mark.skipif(not os.path.exists('data/puzzle_multi_001.jpg'),
                       reason="Puzzle file not found")
    def test_puzzle_multi_001(self):
        """Test BFS on puzzle_multi_001."""
        from explainrl.agents.bfs import load_puzzle

        env = load_puzzle('puzzle_multi_001')
        solver = BFSSolver(env, verbose=False)

        solution = solver.solve()

        # Should find a solution
        assert solution is not None

        # Verify solution works
        env.reset()
        for move in solution:
            env.step(move)

        assert env.state.is_won()

    @pytest.mark.skipif(not os.path.exists('data/puzzle_single_001.jpg'),
                       reason="Puzzle file not found")
    def test_puzzle_single_001(self):
        """Test BFS on puzzle_single_001."""
        from explainrl.agents.bfs import load_puzzle

        env = load_puzzle('puzzle_single_001')
        solver = BFSSolver(env, verbose=False)

        solution = solver.solve()

        # Should find a solution
        assert solution is not None

        # Verify solution works
        env.reset()
        for move in solution:
            env.step(move)

        assert env.state.is_won()
