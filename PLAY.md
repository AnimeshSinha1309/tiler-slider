# Tiler-Slider Interactive Play

## Quick Start

Play a level interactively using keyboard controls:

```bash
# From project root
python explainrl/environment/play.py --level puzzle_single_175

# Or using module syntax
python -m explainrl.environment.play --level puzzle_single_175
```

## Controls

- **Arrow Keys**: Move tiles (UP, DOWN, LEFT, RIGHT)
- **Q or ESC**: Quit game

## Command Line Options

```bash
python explainrl/environment/play.py --help
```

### Arguments

- `--level` (required): Level name (e.g., `puzzle_single_175`, `puzzle_multi_042`)
- `--data-dir`: Directory containing level images (default: `./data`)
- `--max-steps`: Maximum number of steps allowed (default: `100`)

### Examples

```bash
# Play a single-color puzzle
python explainrl/environment/play.py --level puzzle_single_175

# Play a multi-color puzzle with custom step limit
python explainrl/environment/play.py --level puzzle_multi_042 --max-steps 50

# Specify custom data directory
python explainrl/environment/play.py --level my_puzzle --data-dir ./custom_data
```

## Level Naming Convention

The script automatically detects puzzle type from the filename:
- **Single-color mode**: Filenames containing `_single_` or no color indicator
- **Multi-color mode**: Filenames containing `_multi_`

Supported image formats: `.jpg`, `.jpeg`, `.png`

## Gameplay

1. The game window opens showing the initial puzzle state
2. Use arrow keys to move all tiles in that direction
3. Tiles slide until they hit a wall, obstacle, or another tile
4. Goal: Move all tiles to their target positions
5. Win condition:
   - **Single-color**: Any tile can go to any target
   - **Multi-color**: Each tile must reach its matching target (same letter)

## Visual Guide

- **Lowercase letters (a, b, c)**: Tiles
- **Uppercase letters (A, B, C)**: Targets
- **Gray blocks**: Obstacles
- **Purple + circle**: Tile on empty space
- **Purple filled**: Tile on its target
- **Green "SOLVED!"**: Puzzle completed
- **Red "Out of moves!"**: Maximum steps reached

## Exit Codes

- `0`: Puzzle solved successfully
- `1`: Quit, timeout, or error

## Troubleshooting

### "Data directory does not exist"
Ensure you're running from the project root and the `data/` directory exists.

### "Level not found"
Check that:
1. The level image exists in the data directory
2. The filename matches the `--level` argument (without extension)
3. The image has a supported extension (`.jpg`, `.jpeg`, or `.png`)

### Import errors
Make sure you're running from the project root, or the package is properly installed.
