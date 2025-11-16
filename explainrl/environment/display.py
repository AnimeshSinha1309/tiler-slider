"""
Rendering classes for Tiler-Slider environment.

This module provides different renderers for visualizing the game state:
- TextRender: ASCII text rendering
- PygameRender: Graphical rendering using pygame
"""

import sys
from typing import Optional
import pygame
from .config import COLORS, BLOCK_SIZE, RADIUS, LINE_WIDTH, WAIT_TIME, ANIMATION_TIME
from .environment import TilerSliderEnv
from .state import GameState


class TextRender:
    """
    Text-based renderer for Tiler-Slider environment.

    Renders the game state as ASCII art with:
    - Lowercase letters (a, b, c...) for tiles
    - Uppercase letters (A, B, C...) for targets
    - 'X' for blocked cells
    - '.' for empty cells
    """

    def __init__(self, env: TilerSliderEnv):
        """
        Initialize text renderer.

        Args:
            env: TilerSliderEnv instance to render
        """
        self.env = env

    def render(self, show_info: bool = True) -> str:
        """
        Render the current environment state as text.

        Args:
            show_info: Whether to include step count and status

        Returns:
            String representation of the board
        """
        if self.env.state is None:
            return "Environment not initialized. Call reset() first."

        output = []

        if show_info:
            output.append(f"Step: {self.env.step_count}/{self.env.max_steps}")
            output.append(f"Done: {self.env.done}")
            output.append("")

        # Render the board
        board = []
        for i in range(self.env.size):
            row = []
            for j in range(self.env.size):
                # Check if there's a target at this position
                if (i, j) in self.env.state.target_locations:
                    idx = self.env.state.target_locations.index((i, j)) if self.env.multi_color else 0
                    row.append(chr(idx + ord('A')))
                # Check if there's a tile at this position
                elif (i, j) in self.env.state.current_locations:
                    idx = self.env.state.current_locations.index((i, j)) if self.env.multi_color else 0
                    row.append(chr(idx + ord('a')))
                # Check if blocked
                elif self.env.state.is_blocked[i, j]:
                    row.append('X')
                # Empty cell
                else:
                    row.append('.')
            board.append(''.join(row))

        output.extend(board)
        return '\n'.join(output)

    def __str__(self) -> str:
        """String representation using render method."""
        return self.render()

    @classmethod
    def simulate(cls, env: TilerSliderEnv, moves: list[GameState.Move],
                 print_each_step: bool = True) -> bool:
        """
        Simulate a sequence of moves and render each step.

        Args:
            env: Environment to simulate
            moves: List of moves to execute
            print_each_step: Whether to print state after each move

        Returns:
            True if puzzle was solved, False otherwise
        """
        renderer = cls(env)
        env.reset()

        if print_each_step:
            print("Initial state:")
            print(renderer.render())
            print()

        for i, move in enumerate(moves):
            obs, done, info = env.step(move)

            if print_each_step:
                print(f"Move {i+1}: {move.name}")
                print(renderer.render())
                print()

            if done:
                if info.get('is_won'):
                    if print_each_step:
                        print("Puzzle solved!")
                    return True
                elif info.get('timeout'):
                    if print_each_step:
                        print("Timeout!")
                    return False

        return env.state.is_won()


class PygameRender:
    """
    Pygame-based graphical renderer for Tiler-Slider environment.

    Provides animated visualization with:
    - Color-coded tiles and targets
    - Smooth animations
    - Interactive display
    """

    def __init__(self, env: TilerSliderEnv):
        """
        Initialize pygame renderer.

        Args:
            env: TilerSliderEnv instance to render
        """
        self.env = env
        pygame.init()

        window_width = self.env.size * BLOCK_SIZE
        window_height = self.env.size * BLOCK_SIZE
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Tiler-Slider")

    def render(self):
        """Render the current state to the pygame window."""
        if self.env.state is None:
            return

        for i in range(self.env.size):
            for j in range(self.env.size):
                rect = pygame.Rect(
                    j * BLOCK_SIZE,
                    i * BLOCK_SIZE,
                    BLOCK_SIZE,
                    BLOCK_SIZE
                )

                circle_color = COLORS["SPACE"]

                # Determine cell color based on state
                if ((i, j) in self.env.state.current_locations and
                    (i, j) in self.env.state.target_locations):
                    color = COLORS["COMBINED"]
                elif (i, j) in self.env.state.current_locations:
                    color = COLORS["TILE"]
                    circle_color = COLORS["SPACE"]
                elif (i, j) in self.env.state.target_locations:
                    color = COLORS["SPACE"]
                    circle_color = COLORS["TARGET"]
                elif self.env.state.is_blocked[i, j]:
                    color = COLORS["OBSTACLE"]
                else:
                    color = COLORS["SPACE"]

                pygame.draw.rect(self.screen, color, rect, 0, border_radius=0)

                # Draw circles for tiles or targets (but not both)
                if (((i, j) in self.env.state.current_locations) ^
                    ((i, j) in self.env.state.target_locations)):
                    pygame.draw.circle(
                        self.screen,
                        circle_color,
                        (j * BLOCK_SIZE + BLOCK_SIZE // 2,
                         i * BLOCK_SIZE + BLOCK_SIZE // 2),
                        radius=RADIUS
                    )

        # Draw grid lines
        for i in range(self.env.size + 1):
            # Horizontal lines
            pygame.draw.line(
                self.screen,
                COLORS['LINE'],
                (0, i * BLOCK_SIZE),
                (self.env.size * BLOCK_SIZE, i * BLOCK_SIZE),
                LINE_WIDTH
            )
            # Vertical lines
            pygame.draw.line(
                self.screen,
                COLORS['LINE'],
                (i * BLOCK_SIZE, 0),
                (i * BLOCK_SIZE, self.env.size * BLOCK_SIZE),
                LINE_WIDTH
            )

    def update(self, wait_time: int = WAIT_TIME):
        """
        Update the display and wait.

        Args:
            wait_time: Time to wait in milliseconds
        """
        self.screen.fill(COLORS["SCREEN"])
        self.render()
        pygame.display.update()
        pygame.time.wait(wait_time)

    @staticmethod
    def check_quit() -> bool:
        """
        Check for quit events.

        Returns:
            True if user wants to quit, False otherwise
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_q
            ):
                return True
        return False

    def close(self):
        """Close the pygame window."""
        pygame.quit()

    @classmethod
    def simulate(cls, env: TilerSliderEnv, moves: list[GameState.Move],
                 animate: bool = True) -> bool:
        """
        Simulate a sequence of moves with pygame visualization.

        Args:
            env: Environment to simulate
            moves: List of moves to execute
            animate: Whether to animate the movements

        Returns:
            True if puzzle was solved, False otherwise
        """
        renderer = cls(env)
        env.reset()

        # Show initial state
        renderer.update(WAIT_TIME)

        for move in moves:
            if renderer.check_quit():
                renderer.close()
                sys.exit()

            # Execute move
            obs, done, info = env.step(move)

            # Animate if requested
            if animate and not info.get('invalid_move'):
                for _ in range(max(env.size // 2, 3)):
                    renderer.update(ANIMATION_TIME)
                    if renderer.check_quit():
                        renderer.close()
                        sys.exit()
            else:
                renderer.update(WAIT_TIME)

            if done:
                # Show final state
                renderer.update(WAIT_TIME * 2)
                won = info.get('is_won', False)
                renderer.close()
                return won

        # Final check
        won = env.state.is_won()
        renderer.update(WAIT_TIME * 2)
        renderer.close()
        return won
