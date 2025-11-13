# Environment Scripts Update - Changelog

## Overview
This update introduces a complete refactor of the game environment with improved movement logic, proper collision handling, and a reinforcement learning-ready interface.

## Key Improvements

### 1. **Fixed Movement Logic** (state.py)

#### Before:
- Move enum used tuple values `(delta_r, delta_c)` which were inefficient for array indexing
- No precomputation of moves - had to iterate to find destinations
- Tile-to-tile collisions were not properly handled
- No distinction between single-color and multi-color modes

#### After:
- **Integer-based Move enum** (0-3) for efficient array indexing
- **Pre-computed move cache**: `move_to[i, j, direction]` stores all possible destinations
  - Reduces move complexity from O(n_tiles * board_size) to O(n_tiles)
- **Proper collision handling**: Tiles are processed in order based on movement direction
  - UP/DOWN: Process by row (top-to-bottom or bottom-to-top)
  - LEFT/RIGHT: Process by column (left-to-right or right-to-left)
  - Tiles "push back" when colliding with other tiles
- **Multi-color support**:
  - `multi_color=True`: Exact order matching (tile[i] must reach target[i])
  - `multi_color=False`: Set matching (any tile can reach any target)

### 2. **New Features in GameState**

- `get_state_array()`: Returns 3D numpy array for ML models
  - Channel 0: Blocked cells
  - Channel 1: Tile positions
  - Channel 2: Target positions
- `copy()`: Deep copy of game state for tree search algorithms
- `render()`: Improved visualization with:
  - Lowercase letters for tiles (a, b, c...)
  - Uppercase letters for targets (A, B, C...)
  - 'X' for blocked cells
  - '.' for empty cells

### 3. **New RL Environment** (environment.py)

Created `TilerSliderEnv` following OpenAI Gym interface:

#### Action Space
- Discrete(4): UP (0), DOWN (1), LEFT (2), RIGHT (3)

#### Observation Space
- Box(size, size, 3): 3-channel state representation

#### Reward Structure
- `WIN_REWARD = 100.0`: Achieved when puzzle is solved
- `STEP_PENALTY = -0.1`: Small penalty per step to encourage efficiency
- `INVALID_MOVE_PENALTY = -1.0`: Penalty for moves that don't change state

#### Key Methods
- `reset()`: Initialize new episode
- `step(action)`: Execute action and return (observation, reward, done, info)
- `render(mode)`: Visualize state ('human' for text, 'rgb_array' for array)
- `get_valid_actions()`: Get list of actions that would change state
- `get_info()`: Get detailed environment information

### 4. **Environment Factory**

`TilerSliderEnvFactory` provides convenient creation methods:

- `create_simple_env()`: Generate random environments with seed support
- `create_from_string()`: Create from ASCII art representation
- `from_level()`: Create from ImageLoader.ImageProcessed objects

### 5. **Comprehensive Test Suite**

Created extensive unit tests with near 100% coverage:

#### test_state.py (350+ lines, 60+ tests)
- Initialization and setup
- Move precomputation (all directions, with/without obstacles)
- Movement logic (simple moves, edge cases)
- Tile collision handling (2 tiles, 3 tiles, with obstacles)
- Win condition checking (single/multi-color modes)
- Rendering and visualization
- State array generation for ML
- State copying
- Move enum helpers
- Edge cases (1x1 board, all blocked, large boards)

#### test_environment.py (500+ lines, 80+ tests)
- Environment initialization
- Reset functionality
- Step function and rewards
- Done conditions (win, timeout)
- Rendering (human, rgb_array modes)
- Valid action detection
- Environment info
- Factory methods (simple, from_string, from_level)
- Complete episode workflows
- Edge cases (zero max_steps, already won, empty env)

#### test_dataloader.py (150+ lines, 15+ tests)
- Data class creation
- Constants validation
- Edge cases (single tile, many tiles, all blocked)

### 6. **Project Structure Updates**

```
explainrl/
├── __init__.py                    # Package exports
└── environment/
    ├── __init__.py               # Module exports
    ├── state.py                  # Core game state (NEW VERSION)
    ├── environment.py            # RL environment (NEW)
    ├── dataloader.py             # Image loading (MOVED)
    ├── config.py                 # Constants
    └── display.py                # Visualization

tests/
├── __init__.py
├── test_state.py                 # State tests (NEW)
├── test_environment.py           # Environment tests (NEW)
└── test_dataloader.py            # Dataloader tests (NEW)

pytest.ini                         # Pytest configuration (NEW)
requirements-test.txt              # Test dependencies (NEW)
```

## Testing

Install test dependencies:
```bash
pip install -r requirements-test.txt
```

Run tests:
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=explainrl --cov-report=html

# Run specific test file
pytest tests/test_state.py -v

# Run specific test
pytest tests/test_state.py::TestMovementLogic::test_simple_move_up -v
```

## Usage Examples

### Basic Usage
```python
from explainrl.environment import TilerSliderEnv, GameState

# Create environment
env = TilerSliderEnv(
    size=5,
    blocked_locations=[(2, 2)],
    initial_locations=[(0, 0)],
    target_locations=[(4, 4)],
    max_steps=100
)

# Reset environment
obs = env.reset()

# Take a step
action = GameState.Move.DOWN.value  # 1
obs, reward, done, info = env.step(action)

# Render
print(env.render(mode='human'))
```

### Factory Methods
```python
from explainrl.environment import TilerSliderEnvFactory

# Create simple random environment
env = TilerSliderEnvFactory.create_simple_env(
    size=5,
    num_tiles=2,
    num_obstacles=3,
    seed=42  # For reproducibility
)

# Create from string
board = """
a..
.X.
..A
"""
env = TilerSliderEnvFactory.create_from_string(board)
```

### RL Training Loop
```python
env = TilerSliderEnv(size=5, ...)
obs = env.reset()

for episode in range(100):
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Get valid actions
        valid_actions = env.get_valid_actions()

        # Choose action (replace with your policy)
        action = random.choice(valid_actions)

        # Step
        obs, reward, done, info = env.step(action)
        total_reward += reward

        if info.get('is_won'):
            print(f"Solved in {env.step_count} steps!")
```

## Migration Guide

### Old Code:
```python
from explainrl.environment.state import GridState

state = GridState(n, m, grid, tiles, targets)
state.move(GridState.Move.UP)
```

### New Code:
```python
from explainrl.environment import GameState

state = GameState(
    size=n,
    blocked_locations=[(i,j) for i,j in blocked],
    initial_locations=tiles,
    target_locations=targets
)
state.move(GameState.Move.UP)
```

## Performance Improvements

1. **Pre-computed moves**: O(1) lookup instead of O(board_size) iteration
2. **Efficient collision handling**: Single pass through tiles
3. **Numpy arrays**: Faster numerical operations for ML
4. **Cached blocked cells**: O(1) lookup instead of list search

## Breaking Changes

- `GridState` renamed to `GameState`
- Move enum values changed from tuples to integers
- Constructor parameters changed (see Migration Guide)
- `move()` now returns boolean (win status)

## Backward Compatibility

The old `GridState.load()` classmethod is not present in the new version. To load from files, you'll need to:

1. Parse the file yourself
2. Create `GameState` with the parsed data

Or use the factory methods for common use cases.
