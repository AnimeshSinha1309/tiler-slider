# Test Suite for Tiler-Slider

This directory contains comprehensive unit tests for the Tiler-Slider game environment.

## Test Coverage

### test_state.py (60+ tests)
Tests for the core `GameState` class:

- ✅ **Initialization**: Basic setup, empty boards, multi-color mode
- ✅ **Move Precomputation**: All 4 directions, with/without obstacles, edge cases
- ✅ **Movement Logic**: Simple moves, blocked by walls, blocked by edges
- ✅ **Tile Collisions**: 2-tile, 3-tile, with obstacles
- ✅ **Win Conditions**: Single-color, multi-color, partial completion
- ✅ **Rendering**: Basic, multi-color, string representation
- ✅ **State Arrays**: Shape, channels, multi-color indices
- ✅ **State Copying**: Independence, attribute preservation
- ✅ **Move Enum**: Character/integer conversions, values
- ✅ **Edge Cases**: 1x1 boards, all blocked, large boards, many tiles

### test_environment.py (80+ tests)
Tests for the `TilerSliderEnv` RL interface:

- ✅ **Initialization**: Basic, defaults, multi-color
- ✅ **Reset**: Returns observation, initializes state, after steps
- ✅ **Step Function**: Return tuple, counter increment, observation updates
- ✅ **Rewards**: Step penalty, win reward, invalid move penalty
- ✅ **Done Conditions**: Win, timeout, normal continuation
- ✅ **Rendering**: Human mode, RGB array mode, before reset
- ✅ **Valid Actions**: Corner, center, with obstacles
- ✅ **Environment Info**: Initialized, not initialized, after steps
- ✅ **Factory - Simple**: Creation, reproducibility, no overlaps
- ✅ **Factory - String**: Basic, multi-tile, complex, playable
- ✅ **From Level**: Basic, custom max_steps
- ✅ **Complete Episodes**: Winning, timeout, multiple episodes
- ✅ **Edge Cases**: Zero max_steps, already won, empty environment

### test_dataloader.py (15+ tests)
Tests for the `ImageLoader` data structures:

- ✅ **Data Classes**: ImageRawData, ImageProcessed creation
- ✅ **Constants**: Existence, types, shapes, valid ranges
- ✅ **Edge Cases**: Single tile, many tiles, all blocked, valid coordinates

## Running Tests

### Prerequisites
```bash
pip install -r requirements-test.txt
```

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test File
```bash
pytest tests/test_state.py -v
pytest tests/test_environment.py -v
pytest tests/test_dataloader.py -v
```

### Run Specific Test Class
```bash
pytest tests/test_state.py::TestMovementLogic -v
```

### Run Specific Test
```bash
pytest tests/test_state.py::TestMovementLogic::test_simple_move_up -v
```

### Run with Coverage
```bash
# Generate coverage report
pytest tests/ --cov=explainrl --cov-report=html

# View report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Run Tests Matching Pattern
```bash
# Run all tests with "collision" in name
pytest tests/ -k collision -v

# Run all tests with "win" in name
pytest tests/ -k win -v
```

## Test Statistics

- **Total Tests**: 155+
- **Total Lines**: 1000+
- **Expected Coverage**: Near 100% for state.py and environment.py

## Test Design Principles

1. **Comprehensive**: Cover all methods, branches, and edge cases
2. **Isolated**: Each test is independent
3. **Clear**: Descriptive names and docstrings
4. **Fast**: No external dependencies or slow operations
5. **Deterministic**: Reproducible with seeds

## Coverage Goals

| Module | Target Coverage |
|--------|----------------|
| state.py | > 95% |
| environment.py | > 95% |
| dataloader.py | > 90% |

## Continuous Integration

Add to your CI pipeline:

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.8'
      - run: pip install -r requirements-test.txt
      - run: pytest tests/ --cov=explainrl --cov-report=xml
      - uses: codecov/codecov-action@v2
```

## Adding New Tests

When adding new functionality:

1. Create test class: `class TestNewFeature:`
2. Add test methods: `def test_specific_behavior(self):`
3. Use descriptive names
4. Add docstrings
5. Test edge cases
6. Aim for 100% coverage of new code

Example:
```python
class TestNewFeature:
    """Test the new feature."""

    def test_basic_behavior(self):
        """Test basic functionality works."""
        # Arrange
        env = TilerSliderEnv(size=3, ...)

        # Act
        result = env.new_method()

        # Assert
        assert result == expected_value
```
