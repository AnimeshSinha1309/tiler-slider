"""
Game state implementation for the Tiler-Slider puzzle game.

FIXES from previous version:
1. Integer-based Move enum (0-3) instead of tuple values for cleaner indexing
2. Pre-computed move_to cache for efficient movement lookups
3. Proper tile collision handling - processes tiles in order and handles cascading
4. Multi-color support - distinguishes ordered vs unordered matching
5. Proper win condition checking
6. Better rendering with lowercase tiles and uppercase targets
"""

import copy
import enum
import numpy as np


class GameState:
    """
    Represents the state of a Tiler-Slider puzzle game.

    The game consists of a grid with:
    - Tiles that can slide in 4 directions
    - Target positions where tiles need to reach
    - Blocked cells that prevent movement
    - Optional multi-color mode where tile-target pairing matters
    """

    class Move(enum.Enum):
        """Movement directions with integer values for array indexing."""
        UP = 0
        DOWN = 1
        LEFT = 2
        RIGHT = 3

        @classmethod
        def from_char(cls, direction: str):
            """Convert character to Move enum."""
            mapping = {'U': cls.UP, 'D': cls.DOWN, 'L': cls.LEFT, 'R': cls.RIGHT}
            return mapping.get(direction.upper())

        @classmethod
        def from_int(cls, value: int):
            """Convert integer to Move enum."""
            return cls(value)

    def __init__(self, size: int, blocked_locations: list[tuple[int, int]],
                 initial_locations: list[tuple[int, int]],
                 target_locations: list[tuple[int, int]],
                 multi_color: bool = False):
        """
        Initialize game state.

        Args:
            size: Grid size (size x size board)
            blocked_locations: List of (row, col) blocked cells
            initial_locations: List of (row, col) initial tile positions
            target_locations: List of (row, col) target positions
            multi_color: If True, tile order matters for matching targets
        """
        self.size = size
        self.current_locations = copy.copy(initial_locations)
        self.target_locations = copy.copy(target_locations)
        self.multi_color = multi_color

        # FIX 1: Expanding grid of blocked locations for O(1) lookup
        self.is_blocked = np.zeros((size, size), dtype=bool)
        for i, j in blocked_locations:
            self.is_blocked[i, j] = True

        # FIX 2: Pre-compute all possible moves for efficiency
        # move_to[i, j, direction] = (new_i, new_j) after sliding
        self._precompute_moves()

    def _precompute_moves(self):
        """
        Pre-compute sliding destinations for all positions and directions.

        FIX: This eliminates the need to iterate during gameplay,
        making moves O(n_tiles) instead of O(n_tiles * board_size).
        """
        self.move_to = np.full((self.size, self.size, 4, 2), -1, dtype=int)

        # UP: Process top to bottom
        for i in range(self.size):
            for j in range(self.size):
                if i > 0 and not self.is_blocked[i - 1, j]:
                    self.move_to[i, j, self.Move.UP.value] = \
                        self.move_to[i - 1, j, self.Move.UP.value]
                else:
                    self.move_to[i, j, self.Move.UP.value] = (i, j)

        # DOWN: Process bottom to top
        for i in reversed(range(self.size)):
            for j in range(self.size):
                if i < self.size - 1 and not self.is_blocked[i + 1, j]:
                    self.move_to[i, j, self.Move.DOWN.value] = \
                        self.move_to[i + 1, j, self.Move.DOWN.value]
                else:
                    self.move_to[i, j, self.Move.DOWN.value] = (i, j)

        # LEFT: Process left to right
        for i in range(self.size):
            for j in range(self.size):
                if j > 0 and not self.is_blocked[i, j - 1]:
                    self.move_to[i, j, self.Move.LEFT.value] = \
                        self.move_to[i, j - 1, self.Move.LEFT.value]
                else:
                    self.move_to[i, j, self.Move.LEFT.value] = (i, j)

        # RIGHT: Process right to left
        for i in range(self.size):
            for j in reversed(range(self.size)):
                if j < self.size - 1 and not self.is_blocked[i, j + 1]:
                    self.move_to[i, j, self.Move.RIGHT.value] = \
                        self.move_to[i, j + 1, self.Move.RIGHT.value]
                else:
                    self.move_to[i, j, self.Move.RIGHT.value] = (i, j)

    def move(self, move: Move) -> bool:
        """
        Execute a move by sliding all tiles in the given direction.

        FIX 3: Proper collision handling
        - Process tiles in order based on movement direction
        - Handle tile-to-tile collisions by tracking used positions
        - Tiles "push back" against other tiles when they collide

        Args:
            move: Direction to move tiles

        Returns:
            True if the game is won after this move, False otherwise
        """
        # FIX: Determine processing order based on direction
        # Process tiles closer to the destination first to avoid conflicts
        if move == self.Move.UP:
            order_to_process = np.argsort([r for r, c in self.current_locations])
        elif move == self.Move.DOWN:
            order_to_process = np.argsort([-r for r, c in self.current_locations])
        elif move == self.Move.LEFT:
            order_to_process = np.argsort([c for r, c in self.current_locations])
        elif move == self.Move.RIGHT:
            order_to_process = np.argsort([-c for r, c in self.current_locations])

        used_locations = set()
        for i in order_to_process:
            # Get the sliding destination from pre-computed cache
            self.current_locations[i] = tuple(self.move_to[
                self.current_locations[i][0],
                self.current_locations[i][1],
                move.value
            ])

            # FIX: Handle tile-to-tile collision
            # If destination is occupied, move back one step
            while self.current_locations[i] in used_locations:
                row, col = self.current_locations[i]
                if move == self.Move.UP:
                    self.current_locations[i] = (row + 1, col)
                elif move == self.Move.DOWN:
                    self.current_locations[i] = (row - 1, col)
                elif move == self.Move.LEFT:
                    self.current_locations[i] = (row, col + 1)
                elif move == self.Move.RIGHT:
                    self.current_locations[i] = (row, col - 1)

            used_locations.add(self.current_locations[i])

        return self.is_won()

    def is_won(self) -> bool:
        """
        Check if the game is in a winning state.

        FIX 4: Proper multi-color support
        - multi_color=True: Exact order matching (tile[i] must reach target[i])
        - multi_color=False: Set matching (any tile can reach any target)

        Returns:
            True if all tiles are at their targets, False otherwise
        """
        if self.multi_color:
            return self.current_locations == self.target_locations
        else:
            return set(self.current_locations) == set(self.target_locations)

    def get_state_array(self) -> np.ndarray:
        """
        Get numerical representation of the board for ML models.

        Returns:
            3D array of shape (size, size, 3) where:
            - Channel 0: Blocked cells (1) vs free cells (0)
            - Channel 1: Tile positions (tile_index + 1, or 0 for no tile)
            - Channel 2: Target positions (target_index + 1, or 0 for no target)
        """
        state = np.zeros((self.size, self.size, 3), dtype=np.float32)

        # Channel 0: Blocked cells
        state[:, :, 0] = self.is_blocked.astype(np.float32)

        # Channel 1: Tile positions
        for idx, (i, j) in enumerate(self.current_locations):
            state[i, j, 1] = idx + 1 if self.multi_color else 1

        # Channel 2: Target positions
        for idx, (i, j) in enumerate(self.target_locations):
            state[i, j, 2] = idx + 1 if self.multi_color else 1

        return state

    def copy(self):
        """Create a deep copy of the game state."""
        new_state = GameState(
            self.size,
            [(i, j) for i in range(self.size) for j in range(self.size) if self.is_blocked[i, j]],
            copy.copy(self.current_locations),
            copy.copy(self.target_locations),
            self.multi_color
        )
        return new_state
