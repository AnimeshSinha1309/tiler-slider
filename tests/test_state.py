"""
Comprehensive unit tests for GameState class.

Tests cover:
- Initialization and setup
- Move precomputation
- All movement directions
- Tile collision handling
- Win condition checking
- Multi-color mode
- Edge cases and error conditions
"""

import pytest
import numpy as np
from explainrl.environment.state import GameState


class TestGameStateInitialization:
    """Test GameState initialization and setup."""

    def test_basic_initialization(self):
        """Test basic state initialization."""
        state = GameState(
            size=5,
            blocked_locations=[(1, 1), (2, 2)],
            initial_locations=[(0, 0), (0, 1)],
            target_locations=[(4, 4), (4, 3)],
            multi_color=False
        )

        assert state.size == 5
        assert len(state.current_locations) == 2
        assert len(state.target_locations) == 2
        assert state.is_blocked[1, 1] == True
        assert state.is_blocked[2, 2] == True
        assert state.is_blocked[0, 0] == False
        assert state.multi_color == False

    def test_empty_board(self):
        """Test initialization with no tiles."""
        state = GameState(
            size=3,
            blocked_locations=[],
            initial_locations=[],
            target_locations=[],
            multi_color=False
        )

        assert state.size == 3
        assert len(state.current_locations) == 0
        assert state.is_won() == True  # No tiles means already won

    def test_multi_color_mode(self):
        """Test multi-color mode initialization."""
        state = GameState(
            size=4,
            blocked_locations=[],
            initial_locations=[(0, 0), (0, 1), (0, 2)],
            target_locations=[(3, 0), (3, 1), (3, 2)],
            multi_color=True
        )

        assert state.multi_color == True
        assert len(state.current_locations) == 3


class TestMovePrecomputation:
    """Test move precomputation logic."""

    def test_precompute_up_no_obstacles(self):
        """Test UP move precomputation without obstacles."""
        state = GameState(
            size=3,
            blocked_locations=[],
            initial_locations=[(2, 1)],
            target_locations=[(0, 1)],
            multi_color=False
        )

        # From (2, 1) moving UP should go to (0, 1)
        dest = state.move_to[2, 1, GameState.Move.UP.value]
        assert tuple(dest) == (0, 1)

    def test_precompute_down_no_obstacles(self):
        """Test DOWN move precomputation without obstacles."""
        state = GameState(
            size=3,
            blocked_locations=[],
            initial_locations=[(0, 1)],
            target_locations=[(2, 1)],
            multi_color=False
        )

        # From (0, 1) moving DOWN should go to (2, 1)
        dest = state.move_to[0, 1, GameState.Move.DOWN.value]
        assert tuple(dest) == (2, 1)

    def test_precompute_left_no_obstacles(self):
        """Test LEFT move precomputation without obstacles."""
        state = GameState(
            size=3,
            blocked_locations=[],
            initial_locations=[(1, 2)],
            target_locations=[(1, 0)],
            multi_color=False
        )

        # From (1, 2) moving LEFT should go to (1, 0)
        dest = state.move_to[1, 2, GameState.Move.LEFT.value]
        assert tuple(dest) == (1, 0)

    def test_precompute_right_no_obstacles(self):
        """Test RIGHT move precomputation without obstacles."""
        state = GameState(
            size=3,
            blocked_locations=[],
            initial_locations=[(1, 0)],
            target_locations=[(1, 2)],
            multi_color=False
        )

        # From (1, 0) moving RIGHT should go to (1, 2)
        dest = state.move_to[1, 0, GameState.Move.RIGHT.value]
        assert tuple(dest) == (1, 2)

    def test_precompute_with_obstacle(self):
        """Test move precomputation with obstacles."""
        state = GameState(
            size=5,
            blocked_locations=[(2, 2)],
            initial_locations=[(4, 2)],
            target_locations=[(0, 0)],
            multi_color=False
        )

        # From (4, 2) moving UP should stop at (3, 2) due to obstacle at (2, 2)
        dest = state.move_to[4, 2, GameState.Move.UP.value]
        assert tuple(dest) == (3, 2)

    def test_precompute_edge_cases(self):
        """Test move precomputation at board edges."""
        state = GameState(
            size=3,
            blocked_locations=[],
            initial_locations=[(0, 0)],
            target_locations=[(2, 2)],
            multi_color=False
        )

        # From (0, 0) moving UP should stay at (0, 0)
        dest = state.move_to[0, 0, GameState.Move.UP.value]
        assert tuple(dest) == (0, 0)

        # From (0, 0) moving LEFT should stay at (0, 0)
        dest = state.move_to[0, 0, GameState.Move.LEFT.value]
        assert tuple(dest) == (0, 0)


class TestMovementLogic:
    """Test tile movement logic."""

    def test_simple_move_up(self):
        """Test simple upward movement."""
        state = GameState(
            size=3,
            blocked_locations=[],
            initial_locations=[(2, 1)],
            target_locations=[(0, 1)],
            multi_color=False
        )

        state.move(GameState.Move.UP)
        assert state.current_locations[0] == (0, 1)

    def test_simple_move_down(self):
        """Test simple downward movement."""
        state = GameState(
            size=3,
            blocked_locations=[],
            initial_locations=[(0, 1)],
            target_locations=[(2, 1)],
            multi_color=False
        )

        state.move(GameState.Move.DOWN)
        assert state.current_locations[0] == (2, 1)

    def test_simple_move_left(self):
        """Test simple leftward movement."""
        state = GameState(
            size=3,
            blocked_locations=[],
            initial_locations=[(1, 2)],
            target_locations=[(1, 0)],
            multi_color=False
        )

        state.move(GameState.Move.LEFT)
        assert state.current_locations[0] == (1, 0)

    def test_simple_move_right(self):
        """Test simple rightward movement."""
        state = GameState(
            size=3,
            blocked_locations=[],
            initial_locations=[(1, 0)],
            target_locations=[(1, 2)],
            multi_color=False
        )

        state.move(GameState.Move.RIGHT)
        assert state.current_locations[0] == (1, 2)

    def test_move_blocked_by_wall(self):
        """Test movement blocked by obstacle."""
        state = GameState(
            size=5,
            blocked_locations=[(2, 2)],
            initial_locations=[(4, 2)],
            target_locations=[(0, 0)],
            multi_color=False
        )

        state.move(GameState.Move.UP)
        # Should stop at (3, 2) due to obstacle at (2, 2)
        assert state.current_locations[0] == (3, 2)

    def test_move_blocked_by_edge(self):
        """Test movement blocked by board edge."""
        state = GameState(
            size=3,
            blocked_locations=[],
            initial_locations=[(0, 0)],
            target_locations=[(2, 2)],
            multi_color=False
        )

        initial_pos = state.current_locations[0]
        state.move(GameState.Move.UP)
        # Should stay at (0, 0)
        assert state.current_locations[0] == initial_pos


class TestTileCollisions:
    """Test tile-to-tile collision handling."""

    def test_two_tiles_same_column_move_up(self):
        """Test two tiles in same column moving up."""
        state = GameState(
            size=5,
            blocked_locations=[],
            initial_locations=[(4, 2), (3, 2)],
            target_locations=[(0, 0), (1, 1)],
            multi_color=False
        )

        state.move(GameState.Move.UP)
        # Top tile should reach row 0, bottom tile should be at row 1
        positions = sorted(state.current_locations)
        assert positions[0] == (0, 2)
        assert positions[1] == (1, 2)

    def test_two_tiles_same_row_move_left(self):
        """Test two tiles in same row moving left."""
        state = GameState(
            size=5,
            blocked_locations=[],
            initial_locations=[(2, 4), (2, 3)],
            target_locations=[(0, 0), (1, 1)],
            multi_color=False
        )

        state.move(GameState.Move.LEFT)
        # Left tile should reach col 0, right tile should be at col 1
        positions = sorted(state.current_locations, key=lambda x: x[1])
        assert positions[0] == (2, 0)
        assert positions[1] == (2, 1)

    def test_three_tiles_collision(self):
        """Test three tiles colliding during movement."""
        state = GameState(
            size=6,
            blocked_locations=[],
            initial_locations=[(5, 1), (4, 1), (3, 1)],
            target_locations=[(0, 0), (1, 1), (2, 2)],
            multi_color=False
        )

        state.move(GameState.Move.UP)
        # Should stack at top: (0, 1), (1, 1), (2, 1)
        positions = sorted(state.current_locations)
        assert positions[0] == (0, 1)
        assert positions[1] == (1, 1)
        assert positions[2] == (2, 1)

    def test_tiles_collision_with_obstacle(self):
        """Test tiles colliding with obstacle between them."""
        state = GameState(
            size=5,
            blocked_locations=[(2, 1)],
            initial_locations=[(4, 1), (3, 1)],
            target_locations=[(0, 0), (1, 1)],
            multi_color=False
        )

        state.move(GameState.Move.UP)
        # Both should stop below obstacle
        positions = sorted(state.current_locations)
        assert positions[0] == (3, 1)
        assert positions[1] == (4, 1)


class TestWinCondition:
    """Test win condition checking."""

    def test_win_single_color_mode(self):
        """Test win condition in single color mode."""
        state = GameState(
            size=3,
            blocked_locations=[],
            initial_locations=[(0, 0), (0, 1)],
            target_locations=[(2, 2), (2, 1)],
            multi_color=False
        )

        assert state.is_won() == False

        # Move tiles to targets (order doesn't matter)
        state.current_locations = [(2, 1), (2, 2)]
        assert state.is_won() == True

    def test_win_multi_color_mode(self):
        """Test win condition in multi-color mode."""
        state = GameState(
            size=3,
            blocked_locations=[],
            initial_locations=[(0, 0), (0, 1)],
            target_locations=[(2, 2), (2, 1)],
            multi_color=True
        )

        # Wrong order shouldn't win
        state.current_locations = [(2, 1), (2, 2)]
        assert state.is_won() == False

        # Correct order should win
        state.current_locations = [(2, 2), (2, 1)]
        assert state.is_won() == True

    def test_partial_completion_no_win(self):
        """Test that partial completion doesn't trigger win."""
        state = GameState(
            size=3,
            blocked_locations=[],
            initial_locations=[(0, 0), (0, 1)],
            target_locations=[(2, 2), (2, 1)],
            multi_color=False
        )

        # Only one tile at target
        state.current_locations = [(2, 2), (1, 1)]
        assert state.is_won() == False

    def test_move_returns_win_status(self):
        """Test that move() returns win status."""
        state = GameState(
            size=3,
            blocked_locations=[],
            initial_locations=[(2, 0)],
            target_locations=[(0, 0)],
            multi_color=False
        )

        is_won = state.move(GameState.Move.UP)
        assert is_won == True
        assert state.is_won() == True


class TestRendering:
    """Test rendering and visualization."""

    def test_render_basic(self):
        """Test basic rendering."""
        state = GameState(
            size=3,
            blocked_locations=[(1, 1)],
            initial_locations=[(0, 0)],
            target_locations=[(2, 2)],
            multi_color=False
        )

        render = state.render()
        lines = render.strip().split('\n')

        assert len(lines) == 3
        assert 'a' in render  # Tile
        assert 'A' in render  # Target
        assert 'X' in render  # Blocked

    def test_render_multi_color(self):
        """Test rendering in multi-color mode."""
        state = GameState(
            size=3,
            blocked_locations=[],
            initial_locations=[(0, 0), (0, 1), (0, 2)],
            target_locations=[(2, 0), (2, 1), (2, 2)],
            multi_color=True
        )

        render = state.render()
        # Should have different letters for different tiles
        assert 'a' in render
        assert 'b' in render
        assert 'c' in render
        assert 'A' in render
        assert 'B' in render
        assert 'C' in render

    def test_str_method(self):
        """Test __str__ method."""
        state = GameState(
            size=2,
            blocked_locations=[],
            initial_locations=[(0, 0)],
            target_locations=[(1, 1)],
            multi_color=False
        )

        str_repr = str(state)
        assert isinstance(str_repr, str)
        assert len(str_repr) > 0


class TestStateArray:
    """Test numerical state representation for ML."""

    def test_get_state_array_shape(self):
        """Test state array has correct shape."""
        state = GameState(
            size=5,
            blocked_locations=[(1, 1)],
            initial_locations=[(0, 0)],
            target_locations=[(4, 4)],
            multi_color=False
        )

        arr = state.get_state_array()
        assert arr.shape == (5, 5, 3)
        assert arr.dtype == np.float32

    def test_get_state_array_blocked_channel(self):
        """Test blocked cells in state array."""
        state = GameState(
            size=3,
            blocked_locations=[(1, 1), (2, 2)],
            initial_locations=[(0, 0)],
            target_locations=[(0, 2)],
            multi_color=False
        )

        arr = state.get_state_array()
        # Channel 0 should have blocked cells
        assert arr[1, 1, 0] == 1.0
        assert arr[2, 2, 0] == 1.0
        assert arr[0, 0, 0] == 0.0

    def test_get_state_array_tile_channel(self):
        """Test tile positions in state array."""
        state = GameState(
            size=3,
            blocked_locations=[],
            initial_locations=[(0, 0), (1, 1)],
            target_locations=[(2, 2), (2, 1)],
            multi_color=False
        )

        arr = state.get_state_array()
        # Channel 1 should have tiles
        assert arr[0, 0, 1] > 0
        assert arr[1, 1, 1] > 0
        assert arr[2, 2, 1] == 0

    def test_get_state_array_target_channel(self):
        """Test target positions in state array."""
        state = GameState(
            size=3,
            blocked_locations=[],
            initial_locations=[(0, 0), (1, 1)],
            target_locations=[(2, 2), (2, 1)],
            multi_color=False
        )

        arr = state.get_state_array()
        # Channel 2 should have targets
        assert arr[2, 2, 2] > 0
        assert arr[2, 1, 2] > 0
        assert arr[0, 0, 2] == 0

    def test_get_state_array_multi_color(self):
        """Test state array in multi-color mode."""
        state = GameState(
            size=3,
            blocked_locations=[],
            initial_locations=[(0, 0), (0, 1)],
            target_locations=[(2, 0), (2, 1)],
            multi_color=True
        )

        arr = state.get_state_array()
        # Tiles should have different indices
        assert arr[0, 0, 1] == 1.0  # First tile
        assert arr[0, 1, 1] == 2.0  # Second tile


class TestCopyState:
    """Test state copying."""

    def test_copy_creates_independent_state(self):
        """Test that copy creates independent state."""
        state1 = GameState(
            size=3,
            blocked_locations=[(1, 1)],
            initial_locations=[(0, 0)],
            target_locations=[(2, 2)],
            multi_color=False
        )

        state2 = state1.copy()

        # Modify state1
        state1.move(GameState.Move.DOWN)

        # state2 should be unchanged
        assert state2.current_locations != state1.current_locations

    def test_copy_preserves_attributes(self):
        """Test that copy preserves all attributes."""
        state1 = GameState(
            size=5,
            blocked_locations=[(1, 1), (2, 2)],
            initial_locations=[(0, 0), (0, 1)],
            target_locations=[(4, 4), (4, 3)],
            multi_color=True
        )

        state2 = state1.copy()

        assert state2.size == state1.size
        assert state2.multi_color == state1.multi_color
        assert np.array_equal(state2.is_blocked, state1.is_blocked)
        assert state2.target_locations == state1.target_locations


class TestMoveEnumHelpers:
    """Test Move enum helper methods."""

    def test_from_char(self):
        """Test Move.from_char conversion."""
        assert GameState.Move.from_char('U') == GameState.Move.UP
        assert GameState.Move.from_char('D') == GameState.Move.DOWN
        assert GameState.Move.from_char('L') == GameState.Move.LEFT
        assert GameState.Move.from_char('R') == GameState.Move.RIGHT

        # Test lowercase
        assert GameState.Move.from_char('u') == GameState.Move.UP
        assert GameState.Move.from_char('d') == GameState.Move.DOWN

    def test_from_int(self):
        """Test Move.from_int conversion."""
        assert GameState.Move.from_int(0) == GameState.Move.UP
        assert GameState.Move.from_int(1) == GameState.Move.DOWN
        assert GameState.Move.from_int(2) == GameState.Move.LEFT
        assert GameState.Move.from_int(3) == GameState.Move.RIGHT

    def test_move_enum_values(self):
        """Test Move enum has correct integer values."""
        assert GameState.Move.UP.value == 0
        assert GameState.Move.DOWN.value == 1
        assert GameState.Move.LEFT.value == 2
        assert GameState.Move.RIGHT.value == 3


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_cell_board(self):
        """Test 1x1 board."""
        state = GameState(
            size=1,
            blocked_locations=[],
            initial_locations=[(0, 0)],
            target_locations=[(0, 0)],
            multi_color=False
        )

        assert state.is_won() == True

    def test_all_cells_blocked_except_tiles(self):
        """Test board with maximum obstacles."""
        state = GameState(
            size=3,
            blocked_locations=[(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)],
            initial_locations=[(0, 0)],
            target_locations=[(1, 1)],
            multi_color=False
        )

        # Tile should not be able to move
        initial_pos = state.current_locations[0]
        state.move(GameState.Move.RIGHT)
        assert state.current_locations[0] == initial_pos

    def test_large_board(self):
        """Test larger board size."""
        state = GameState(
            size=20,
            blocked_locations=[(10, 10)],
            initial_locations=[(0, 0)],
            target_locations=[(19, 19)],
            multi_color=False
        )

        assert state.size == 20
        assert state.is_blocked.shape == (20, 20)

    def test_many_tiles(self):
        """Test board with many tiles."""
        size = 10
        num_tiles = 20
        initial = [(i // size, i % size) for i in range(num_tiles)]
        target = [(size - 1 - i // size, size - 1 - i % size) for i in range(num_tiles)]

        state = GameState(
            size=size,
            blocked_locations=[],
            initial_locations=initial,
            target_locations=target,
            multi_color=False
        )

        assert len(state.current_locations) == num_tiles
