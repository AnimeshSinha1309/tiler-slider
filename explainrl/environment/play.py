"""
Interactive gameplay script for Tiler-Slider puzzle.

Usage:
    python explainrl/environment/play.py --level puzzle_single_175
    python -m explainrl.environment.play --level puzzle_single_175

Controls:
    Arrow Keys: Move tiles
    Q or ESC: Quit
"""

import argparse
import sys
import os
import pygame
from pathlib import Path

from explainrl.environment import TilerSliderEnv, GameState
from explainrl.environment.display import PygameRender
from explainrl.environment.dataloader import ImageLoader
from explainrl.environment.config import COLORS, BLOCK_SIZE, WAIT_TIME


class InteractivePygameRender(PygameRender):
    """Extended PygameRender with keyboard controls."""

    def __init__(self, env: TilerSliderEnv):
        """Initialize interactive renderer."""
        super().__init__(env)
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

    def render_with_info(self):
        """Render the game with additional information overlay."""
        # Draw the board
        self.render()

        # Draw info overlay
        info_y = 10

        # Step count
        step_text = self.small_font.render(
            f"Steps: {self.env.step_count}/{self.env.max_steps}",
            True,
            (0, 0, 0)
        )
        self.screen.blit(step_text, (10, info_y))

        # Win status
        if self.env.state.is_won():
            win_text = self.font.render("SOLVED!", True, (0, 200, 0))
            text_rect = win_text.get_rect(center=(
                self.env.size * BLOCK_SIZE // 2,
                self.env.size * BLOCK_SIZE // 2
            ))
            # Draw semi-transparent background
            s = pygame.Surface((text_rect.width + 20, text_rect.height + 20))
            s.set_alpha(200)
            s.fill((255, 255, 255))
            self.screen.blit(s, (text_rect.x - 10, text_rect.y - 10))
            self.screen.blit(win_text, text_rect)
        elif self.env.done:
            timeout_text = self.font.render("Out of moves!", True, (200, 0, 0))
            text_rect = timeout_text.get_rect(center=(
                self.env.size * BLOCK_SIZE // 2,
                self.env.size * BLOCK_SIZE // 2
            ))
            # Draw semi-transparent background
            s = pygame.Surface((text_rect.width + 20, text_rect.height + 20))
            s.set_alpha(200)
            s.fill((255, 255, 255))
            self.screen.blit(s, (text_rect.x - 10, text_rect.y - 10))
            self.screen.blit(timeout_text, text_rect)

        pygame.display.update()

    def wait_for_move(self) -> GameState.Move | None:
        """
        Wait for player to press an arrow key.

        Returns:
            GameState.Move if valid key pressed, None if quit
        """
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                        return None
                    elif event.key == pygame.K_UP:
                        return GameState.Move.UP
                    elif event.key == pygame.K_DOWN:
                        return GameState.Move.DOWN
                    elif event.key == pygame.K_LEFT:
                        return GameState.Move.LEFT
                    elif event.key == pygame.K_RIGHT:
                        return GameState.Move.RIGHT

            pygame.time.wait(50)

    def play(self):
        """
        Main game loop for interactive play.

        Returns:
            True if puzzle was solved, False if quit or timeout
        """
        self.env.reset()

        # Initial render
        self.render_with_info()

        print("\n" + "="*50)
        print("TILER-SLIDER - Interactive Play")
        print("="*50)
        print("\nControls:")
        print("  Arrow Keys: Move tiles")
        print("  Q or ESC: Quit")
        print("\nStarting game...")

        while not self.env.done:
            # Wait for player input
            move = self.wait_for_move()

            if move is None:
                print("\nGame quit by player.")
                self.close()
                return False

            # Execute move
            obs, done, info = self.env.step(move)

            # Update display
            self.render_with_info()

            # Print feedback
            if info.get('invalid_move'):
                print(f"Move {move.name}: No change (invalid move)")
            else:
                print(f"Move {move.name}: Step {self.env.step_count}")

            # Check win/lose conditions
            if info.get('is_won'):
                print(f"\n{'='*50}")
                print(f"CONGRATULATIONS! Puzzle solved in {self.env.step_count} steps!")
                print(f"{'='*50}\n")
                pygame.time.wait(3000)  # Show win screen for 3 seconds
                self.close()
                return True
            elif info.get('timeout'):
                print(f"\n{'='*50}")
                print(f"Out of moves! Maximum steps ({self.env.max_steps}) reached.")
                print(f"{'='*50}\n")
                pygame.time.wait(3000)  # Show timeout screen for 3 seconds
                self.close()
                return False

        self.close()
        return self.env.state.is_won()


def load_level_from_image(level_name: str, data_dir: Path) -> ImageLoader.ImageProcessed:
    """
    Load a level from image file.

    Args:
        level_name: Name of the level (e.g., 'puzzle_single_175')
        data_dir: Directory containing level images

    Returns:
        ImageProcessed object

    Raises:
        FileNotFoundError: If level image not found
    """
    # Look for image file
    possible_extensions = ['.jpg', '.jpeg', '.png']
    image_path = None

    for ext in possible_extensions:
        candidate = data_dir / f"{level_name}{ext}"
        if candidate.exists():
            image_path = candidate
            break

    if image_path is None:
        raise FileNotFoundError(
            f"Level '{level_name}' not found in {data_dir}\n"
            f"Looked for: {', '.join(level_name + ext for ext in possible_extensions)}"
        )

    # Change to data directory to load image
    original_dir = os.getcwd()
    os.chdir(data_dir)

    try:
        loader = ImageLoader()
        # Find the index of this image
        if image_path.name not in loader.files:
            raise FileNotFoundError(f"Image {image_path.name} not in loader files")

        idx = loader.files.index(image_path.name)
        raw_data = loader[idx]

        # Determine if multi-color based on filename
        multi_color = '_multi_' in level_name or '_multi.' in level_name

        # Parse the puzzle image
        level = ImageLoader.parse_puzzle_image(raw_data.puzzle_image, multi_color)

        return level
    finally:
        os.chdir(original_dir)


def main():
    """Main entry point for interactive play."""
    parser = argparse.ArgumentParser(
        description="Play Tiler-Slider puzzle interactively",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python explainrl/environment/play.py --level puzzle_single_175
  python -m explainrl.environment.play --level puzzle_single_175 --data-dir ./data
  python -m explainrl.environment.play --level puzzle_multi_042 --max-steps 50
        """
    )

    parser.add_argument(
        '--level',
        type=str,
        required=True,
        help='Level name (e.g., puzzle_single_175)'
    )

    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data'),
        help='Directory containing level images (default: ./data)'
    )

    parser.add_argument(
        '--max-steps',
        type=int,
        default=100,
        help='Maximum number of steps allowed (default: 100)'
    )

    args = parser.parse_args()

    # Validate data directory
    if not args.data_dir.exists():
        print(f"Error: Data directory '{args.data_dir}' does not exist")
        sys.exit(1)

    # Load level
    print(f"Loading level: {args.level}")
    try:
        level = load_level_from_image(args.level, args.data_dir)
        print(f"✓ Level loaded: {level.size}x{level.size} board, "
              f"{len(level.initial_locations)} tiles, "
              f"{'multi-color' if level.multiple_colors else 'single-color'} mode")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading level: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Create environment
    env = TilerSliderEnv.from_level(level, max_steps=args.max_steps)

    # Create interactive renderer and play
    renderer = InteractivePygameRender(env)

    try:
        won = renderer.play()
        sys.exit(0 if won else 1)
    except KeyboardInterrupt:
        print("\n\nGame interrupted by user.")
        renderer.close()
        sys.exit(1)
    except Exception as e:
        print(f"\nError during gameplay: {e}")
        import traceback
        traceback.print_exc()
        renderer.close()
        sys.exit(1)


if __name__ == '__main__':
    main()
