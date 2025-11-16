"""
Unit tests for ImageLoader class.

Tests cover:
- Initialization
- Data structures
- Edge cases
"""

import pytest
import numpy as np
from explainrl.environment.dataloader import ImageLoader


class TestImageLoaderDataClasses:
    """Test ImageLoader data classes."""

    def test_image_raw_data_creation(self):
        """Test ImageRawData dataclass creation."""
        puzzle_img = np.zeros((100, 100, 3))
        level_label = np.zeros((50, 50, 3))
        target_moves = np.zeros((50, 50, 3))

        raw_data = ImageLoader.ImageRawData(
            name="test.jpg",
            puzzle_image=puzzle_img,
            level_label=level_label,
            target_moves=target_moves
        )

        assert raw_data.name == "test.jpg"
        assert raw_data.puzzle_image.shape == (100, 100, 3)
        assert raw_data.level_label.shape == (50, 50, 3)
        assert raw_data.target_moves.shape == (50, 50, 3)

    def test_image_processed_creation(self):
        """Test ImageProcessed dataclass creation."""
        processed = ImageLoader.ImageProcessed(
            size=5,
            blocked_locations=[(1, 1), (2, 2)],
            initial_locations=[(0, 0)],
            target_locations=[(4, 4)]
        )

        assert processed.size == 5
        assert len(processed.blocked_locations) == 2
        assert len(processed.initial_locations) == 1
        assert len(processed.target_locations) == 1

    def test_image_processed_empty_lists(self):
        """Test ImageProcessed with empty location lists."""
        processed = ImageLoader.ImageProcessed(
            size=3,
            blocked_locations=[],
            initial_locations=[],
            target_locations=[]
        )

        assert processed.size == 3
        assert processed.blocked_locations == []
        assert processed.initial_locations == []
        assert processed.target_locations == []


class TestImageLoaderConstants:
    """Test ImageLoader class constants."""

    def test_color_constants_exist(self):
        """Test that color constants are defined."""
        assert hasattr(ImageLoader, 'BACKGROUND_COLOR')
        assert hasattr(ImageLoader, 'EMPTY_TILE_COLOR')
        assert hasattr(ImageLoader, 'COLOR_TOLERANCE')

    def test_color_constants_are_arrays(self):
        """Test that color constants are numpy arrays."""
        assert isinstance(ImageLoader.BACKGROUND_COLOR, np.ndarray)
        assert isinstance(ImageLoader.EMPTY_TILE_COLOR, np.ndarray)
        assert isinstance(ImageLoader.COLOR_TOLERANCE, np.ndarray)

    def test_color_constants_have_correct_shape(self):
        """Test that color constants have 3 components (RGB)."""
        assert ImageLoader.BACKGROUND_COLOR.shape == (3,)
        assert ImageLoader.EMPTY_TILE_COLOR.shape == (3,)
        assert ImageLoader.COLOR_TOLERANCE.shape == (3,)

    def test_color_values_in_valid_range(self):
        """Test that color values are in valid RGB range."""
        for color in [ImageLoader.BACKGROUND_COLOR,
                     ImageLoader.EMPTY_TILE_COLOR,
                     ImageLoader.COLOR_TOLERANCE]:
            assert np.all(color >= 0)
            assert np.all(color <= 255)


class TestImageLoaderEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_processed_data_with_single_tile(self):
        """Test processed data with single tile."""
        processed = ImageLoader.ImageProcessed(
            size=5,
            blocked_locations=[],
            initial_locations=[(2, 2)],
            target_locations=[(3, 3)]
        )

        assert len(processed.initial_locations) == 1
        assert len(processed.target_locations) == 1

    def test_processed_data_with_many_tiles(self):
        """Test processed data with many tiles."""
        size = 10
        num_tiles = 50

        initial = [(i // size, i % size) for i in range(num_tiles)]
        target = [(size - 1 - i // size, i % size) for i in range(num_tiles)]

        processed = ImageLoader.ImageProcessed(
            size=size,
            blocked_locations=[],
            initial_locations=initial,
            target_locations=target
        )

        assert len(processed.initial_locations) == num_tiles
        assert len(processed.target_locations) == num_tiles

    def test_processed_data_with_all_blocked(self):
        """Test processed data with many blocked cells."""
        size = 5
        blocked = [(i, j) for i in range(size) for j in range(size)]

        processed = ImageLoader.ImageProcessed(
            size=size,
            blocked_locations=blocked[:-2],  # Leave 2 cells free
            initial_locations=[blocked[-2]],
            target_locations=[blocked[-1]]
        )

        assert len(processed.blocked_locations) == size * size - 2

    def test_tuple_coordinates_valid(self):
        """Test that coordinates are valid tuples."""
        processed = ImageLoader.ImageProcessed(
            size=5,
            blocked_locations=[(0, 0), (1, 1)],
            initial_locations=[(2, 2)],
            target_locations=[(3, 3)]
        )

        # Check all coordinates are tuples of length 2
        for loc in processed.blocked_locations:
            assert isinstance(loc, tuple)
            assert len(loc) == 2

        for loc in processed.initial_locations:
            assert isinstance(loc, tuple)
            assert len(loc) == 2

        for loc in processed.target_locations:
            assert isinstance(loc, tuple)
            assert len(loc) == 2
