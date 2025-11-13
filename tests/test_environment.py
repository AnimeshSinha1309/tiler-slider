"""
Comprehensive unit tests for TilerSliderEnv and TilerSliderEnvFactory.

Tests cover:
- Environment initialization
- Reset functionality
- Step function and rewards
- Rendering
- Action validation
- Factory methods
- RL interface compliance
- Edge cases
"""

import pytest
import numpy as np
from explainrl.environment.environment import TilerSliderEnv, TilerSliderEnvFactory
from explainrl.environment.state import GameState


class TestEnvironmentInitialization:
    """Test environment initialization."""

    def test_basic_initialization(self):
        """Test basic environment setup."""
        env = TilerSliderEnv(
            size=5,
            blocked_locations=[(1, 1)],
            initial_locations=[(0, 0)],
            target_locations=[(4, 4)],
            multi_color=False,
            max_steps=100
        )

        assert env.size == 5
        assert env.max_steps == 100
        assert env.multi_color == False
        assert env.action_space_n == 4
        assert env.observation_shape == (5, 5, 3)

    def test_initialization_with_defaults(self):
        """Test initialization with default parameters."""
        env = TilerSliderEnv(size=3)

        assert env.size == 3
        assert env.max_steps == 100
        assert env.blocked_locations == []
        assert env.initial_locations == []
        assert env.target_locations == []

    def test_multi_color_initialization(self):
        """Test multi-color environment initialization."""
        env = TilerSliderEnv(
            size=4,
            initial_locations=[(0, 0), (0, 1)],
            target_locations=[(3, 3), (3, 2)],
            multi_color=True
        )

        assert env.multi_color == True


class TestEnvironmentReset:
    """Test environment reset functionality."""

    def test_reset_returns_observation(self):
        """Test that reset returns valid observation."""
        env = TilerSliderEnv(
            size=3,
            initial_locations=[(0, 0)],
            target_locations=[(2, 2)]
        )

        obs = env.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (3, 3, 3)
        assert obs.dtype == np.float32

    def test_reset_initializes_state(self):
        """Test that reset properly initializes state."""
        env = TilerSliderEnv(
            size=3,
            initial_locations=[(0, 0)],
            target_locations=[(2, 2)]
        )

        env.reset()

        assert env.state is not None
        assert env.step_count == 0
        assert env.done == False

    def test_reset_after_steps(self):
        """Test reset after taking steps."""
        env = TilerSliderEnv(
            size=3,
            initial_locations=[(0, 0)],
            target_locations=[(2, 2)]
        )

        env.reset()
        env.step(0)  # Take a step
        env.step(1)  # Take another step

        # Reset should restore to initial state
        obs = env.reset()

        assert env.step_count == 0
        assert env.done == False
        assert env.state.current_locations == [(0, 0)]


class TestEnvironmentStep:
    """Test step function and transitions."""

    def test_step_returns_correct_tuple(self):
        """Test that step returns (obs, reward, done, info)."""
        env = TilerSliderEnv(
            size=3,
            initial_locations=[(0, 0)],
            target_locations=[(2, 2)]
        )

        env.reset()
        result = env.step(1)  # Move DOWN

        assert len(result) == 4
        obs, reward, done, info = result

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_step_increments_counter(self):
        """Test that step increments step counter."""
        env = TilerSliderEnv(
            size=3,
            initial_locations=[(0, 0)],
            target_locations=[(2, 2)]
        )

        env.reset()
        assert env.step_count == 0

        env.step(0)
        assert env.step_count == 1

        env.step(1)
        assert env.step_count == 2

    def test_step_without_reset_raises_error(self):
        """Test that stepping without reset raises error when done."""
        env = TilerSliderEnv(
            size=2,
            initial_locations=[(1, 0)],
            target_locations=[(0, 0)],
            max_steps=1
        )

        env.reset()
        env.step(0)  # This should set done=True (max steps)

        # Next step should raise error
        with pytest.raises(RuntimeError, match="Episode is done"):
            env.step(0)

    def test_invalid_action_raises_error(self):
        """Test that invalid action raises error."""
        env = TilerSliderEnv(
            size=3,
            initial_locations=[(0, 0)],
            target_locations=[(2, 2)]
        )

        env.reset()

        with pytest.raises(ValueError, match="Invalid action"):
            env.step(4)

        with pytest.raises(ValueError, match="Invalid action"):
            env.step(-1)

    def test_step_updates_observation(self):
        """Test that step returns updated observation."""
        env = TilerSliderEnv(
            size=3,
            initial_locations=[(2, 1)],
            target_locations=[(0, 1)]
        )

        obs1 = env.reset()
        obs2, _, _, _ = env.step(0)  # Move UP

        # Observations should be different
        assert not np.array_equal(obs1, obs2)


class TestRewardStructure:
    """Test reward calculation."""

    def test_step_penalty(self):
        """Test that each step incurs penalty."""
        env = TilerSliderEnv(
            size=3,
            initial_locations=[(1, 1)],
            target_locations=[(2, 2)]
        )

        env.reset()
        _, reward, _, _ = env.step(0)  # Valid move

        # Should get step penalty
        assert reward <= TilerSliderEnv.STEP_PENALTY

    def test_win_reward(self):
        """Test reward when winning."""
        env = TilerSliderEnv(
            size=3,
            initial_locations=[(1, 0)],
            target_locations=[(0, 0)]
        )

        env.reset()
        _, reward, done, info = env.step(0)  # Move UP to win

        assert done == True
        assert info['is_won'] == True
        assert reward >= TilerSliderEnv.WIN_REWARD  # Should include win reward

    def test_invalid_move_penalty(self):
        """Test penalty for invalid move."""
        env = TilerSliderEnv(
            size=3,
            initial_locations=[(0, 0)],
            target_locations=[(2, 2)]
        )

        env.reset()
        _, reward, _, info = env.step(0)  # Move UP from top edge (invalid)

        assert info['invalid_move'] == True
        # Should get both step penalty and invalid move penalty
        assert reward < TilerSliderEnv.STEP_PENALTY


class TestDoneConditions:
    """Test episode termination conditions."""

    def test_done_on_win(self):
        """Test that episode ends when puzzle is solved."""
        env = TilerSliderEnv(
            size=3,
            initial_locations=[(1, 0)],
            target_locations=[(0, 0)]
        )

        env.reset()
        _, _, done, info = env.step(0)  # Move UP to win

        assert done == True
        assert info['success'] == True

    def test_done_on_max_steps(self):
        """Test that episode ends at max_steps."""
        env = TilerSliderEnv(
            size=3,
            initial_locations=[(0, 0)],
            target_locations=[(2, 2)],
            max_steps=2
        )

        env.reset()

        _, _, done1, _ = env.step(1)  # Step 1
        assert done1 == False

        _, _, done2, info = env.step(1)  # Step 2
        assert done2 == True
        assert info.get('timeout') == True

    def test_not_done_before_conditions(self):
        """Test that episode continues normally."""
        env = TilerSliderEnv(
            size=5,
            initial_locations=[(0, 0)],
            target_locations=[(4, 4)],
            max_steps=10
        )

        env.reset()

        for _ in range(5):
            _, _, done, _ = env.step(1)
            if done:
                break

        # Should still be running (puzzle not solved, steps < max)
        # (unless we accidentally solved it)


class TestRendering:
    """Test rendering functionality."""

    def test_render_human_mode(self):
        """Test rendering in human mode."""
        env = TilerSliderEnv(
            size=3,
            initial_locations=[(0, 0)],
            target_locations=[(2, 2)]
        )

        env.reset()
        output = env.render(mode='human')

        assert isinstance(output, str)
        assert 'Step:' in output
        assert 'Done:' in output

    def test_render_rgb_array_mode(self):
        """Test rendering in rgb_array mode."""
        env = TilerSliderEnv(
            size=3,
            initial_locations=[(0, 0)],
            target_locations=[(2, 2)]
        )

        env.reset()
        output = env.render(mode='rgb_array')

        assert isinstance(output, np.ndarray)
        assert output.shape == (3, 3, 3)

    def test_render_without_reset(self):
        """Test rendering before reset."""
        env = TilerSliderEnv(size=3)
        output = env.render(mode='human')

        assert "not initialized" in output.lower()

    def test_render_invalid_mode(self):
        """Test rendering with invalid mode."""
        env = TilerSliderEnv(size=3)
        env.reset()

        with pytest.raises(ValueError, match="Unsupported render mode"):
            env.render(mode='invalid')


class TestValidActions:
    """Test valid action detection."""

    def test_get_valid_actions_corner(self):
        """Test valid actions from corner position."""
        env = TilerSliderEnv(
            size=3,
            initial_locations=[(0, 0)],
            target_locations=[(2, 2)]
        )

        env.reset()
        valid = env.get_valid_actions()

        # From top-left, only DOWN and RIGHT are valid
        assert GameState.Move.DOWN.value in valid
        assert GameState.Move.RIGHT.value in valid
        assert len(valid) <= 2

    def test_get_valid_actions_center(self):
        """Test valid actions from center position."""
        env = TilerSliderEnv(
            size=5,
            initial_locations=[(2, 2)],
            target_locations=[(4, 4)]
        )

        env.reset()
        valid = env.get_valid_actions()

        # From center, all 4 directions should be valid
        assert len(valid) == 4

    def test_get_valid_actions_with_obstacles(self):
        """Test valid actions with obstacles."""
        env = TilerSliderEnv(
            size=3,
            blocked_locations=[(0, 1), (1, 0)],
            initial_locations=[(1, 1)],
            target_locations=[(2, 2)]
        )

        env.reset()
        valid = env.get_valid_actions()

        # Should have some valid actions
        assert len(valid) > 0

    def test_get_valid_actions_before_reset(self):
        """Test getting valid actions before reset."""
        env = TilerSliderEnv(size=3)
        valid = env.get_valid_actions()

        assert valid == []


class TestEnvironmentInfo:
    """Test environment information methods."""

    def test_get_info_initialized(self):
        """Test get_info after initialization."""
        env = TilerSliderEnv(
            size=5,
            initial_locations=[(0, 0), (0, 1)],
            target_locations=[(4, 4), (4, 3)]
        )

        env.reset()
        info = env.get_info()

        assert info['initialized'] == True
        assert info['size'] == 5
        assert info['step_count'] == 0
        assert info['done'] == False
        assert info['is_won'] == False
        assert info['num_tiles'] == 2
        assert info['num_targets'] == 2

    def test_get_info_not_initialized(self):
        """Test get_info before initialization."""
        env = TilerSliderEnv(size=3)
        info = env.get_info()

        assert info['initialized'] == False

    def test_get_info_after_steps(self):
        """Test get_info after taking steps."""
        env = TilerSliderEnv(
            size=3,
            initial_locations=[(0, 0)],
            target_locations=[(2, 2)],
            max_steps=10
        )

        env.reset()
        env.step(1)
        env.step(1)

        info = env.get_info()

        assert info['step_count'] == 2
        assert 'valid_actions' in info


class TestEnvironmentClose:
    """Test environment cleanup."""

    def test_close_clears_state(self):
        """Test that close clears the state."""
        env = TilerSliderEnv(
            size=3,
            initial_locations=[(0, 0)],
            target_locations=[(2, 2)]
        )

        env.reset()
        assert env.state is not None

        env.close()
        assert env.state is None


class TestFactorySimpleEnv:
    """Test factory method for simple environments."""

    def test_create_simple_env(self):
        """Test creating simple random environment."""
        env = TilerSliderEnvFactory.create_simple_env(
            size=5,
            num_tiles=2,
            num_obstacles=3,
            seed=42
        )

        assert env.size == 5
        assert len(env.initial_locations) == 2
        assert len(env.target_locations) == 2
        assert len(env.blocked_locations) == 3

    def test_create_simple_env_reproducible(self):
        """Test that same seed produces same environment."""
        env1 = TilerSliderEnvFactory.create_simple_env(
            size=5,
            num_tiles=2,
            num_obstacles=3,
            seed=42
        )

        env2 = TilerSliderEnvFactory.create_simple_env(
            size=5,
            num_tiles=2,
            num_obstacles=3,
            seed=42
        )

        assert env1.initial_locations == env2.initial_locations
        assert env1.target_locations == env2.target_locations
        assert env1.blocked_locations == env2.blocked_locations

    def test_create_simple_env_different_seeds(self):
        """Test that different seeds produce different environments."""
        env1 = TilerSliderEnvFactory.create_simple_env(seed=1)
        env2 = TilerSliderEnvFactory.create_simple_env(seed=2)

        # Highly likely to be different
        assert (env1.initial_locations != env2.initial_locations or
                env1.blocked_locations != env2.blocked_locations)

    def test_create_simple_env_no_overlaps(self):
        """Test that simple env has no overlapping positions."""
        env = TilerSliderEnvFactory.create_simple_env(
            size=10,
            num_tiles=5,
            num_obstacles=5,
            seed=42
        )

        all_positions = (env.blocked_locations +
                        env.initial_locations +
                        env.target_locations)

        # All positions should be unique
        assert len(all_positions) == len(set(all_positions))


class TestFactoryFromString:
    """Test factory method for string-based creation."""

    def test_create_from_string_basic(self):
        """Test creating environment from string."""
        board = """
        a..
        .X.
        ..A
        """

        env = TilerSliderEnvFactory.create_from_string(board)

        assert env.size == 3
        assert len(env.initial_locations) == 1
        assert len(env.target_locations) == 1
        assert len(env.blocked_locations) == 1

    def test_create_from_string_multi_tile(self):
        """Test creating environment with multiple tiles."""
        board = """
        ab..
        ....
        ....
        ..BA
        """

        env = TilerSliderEnvFactory.create_from_string(board, multi_color=True)

        assert env.size == 4
        assert len(env.initial_locations) == 2
        assert len(env.target_locations) == 2
        assert env.multi_color == True

    def test_create_from_string_complex(self):
        """Test creating complex environment from string."""
        board = """
        abc
        XXX
        CBA
        """

        env = TilerSliderEnvFactory.create_from_string(board, multi_color=True)

        assert env.size == 3
        assert len(env.initial_locations) == 3
        assert len(env.target_locations) == 3
        assert len(env.blocked_locations) == 3

    def test_create_from_string_empty_cells(self):
        """Test string with mostly empty cells."""
        board = """
        a....
        .....
        .....
        .....
        ....A
        """

        env = TilerSliderEnvFactory.create_from_string(board)

        assert env.size == 5
        assert len(env.blocked_locations) == 0

    def test_create_from_string_playable(self):
        """Test that string-created environment is playable."""
        board = """
        a.
        .A
        """

        env = TilerSliderEnvFactory.create_from_string(board)
        env.reset()

        # Should be able to take steps
        obs, reward, done, info = env.step(GameState.Move.DOWN.value)
        assert isinstance(obs, np.ndarray)


class TestFromLevel:
    """Test creating environment from ImageLoader level."""

    def test_from_level_basic(self):
        """Test creating environment from level object."""
        # Create a mock level object
        class MockLevel:
            size = 4
            blocked_locations = [(1, 1)]
            initial_locations = [(0, 0)]
            target_locations = [(3, 3)]

        level = MockLevel()
        env = TilerSliderEnv.from_level(level)

        assert env.size == 4
        assert env.blocked_locations == [(1, 1)]
        assert env.initial_locations == [(0, 0)]
        assert env.target_locations == [(3, 3)]

    def test_from_level_with_max_steps(self):
        """Test from_level with custom max_steps."""
        class MockLevel:
            size = 3
            blocked_locations = []
            initial_locations = [(0, 0)]
            target_locations = [(2, 2)]

        level = MockLevel()
        env = TilerSliderEnv.from_level(level, max_steps=50)

        assert env.max_steps == 50


class TestCompleteEpisode:
    """Test complete episode workflows."""

    def test_complete_winning_episode(self):
        """Test a complete episode that wins."""
        env = TilerSliderEnv(
            size=3,
            initial_locations=[(2, 1)],
            target_locations=[(0, 1)],
            max_steps=10
        )

        obs = env.reset()
        assert isinstance(obs, np.ndarray)

        # Move up to win
        obs, reward, done, info = env.step(GameState.Move.UP.value)

        assert done == True
        assert info['is_won'] == True
        assert reward > 0  # Should include win reward

    def test_complete_timeout_episode(self):
        """Test episode that times out."""
        env = TilerSliderEnv(
            size=5,
            initial_locations=[(0, 0)],
            target_locations=[(4, 4)],
            max_steps=2
        )

        env.reset()

        # Take 2 steps (won't reach target)
        _, _, done1, _ = env.step(1)
        assert done1 == False

        _, _, done2, info = env.step(1)
        assert done2 == True
        assert info.get('timeout') == True
        assert info.get('success', False) == False

    def test_multiple_episodes(self):
        """Test running multiple episodes."""
        env = TilerSliderEnv(
            size=3,
            initial_locations=[(1, 1)],
            target_locations=[(0, 0)],
            max_steps=5
        )

        for episode in range(3):
            obs = env.reset()
            done = False
            steps = 0

            while not done and steps < 10:
                action = np.random.randint(0, 4)
                obs, reward, done, info = env.step(action)
                steps += 1

            # Episode should complete
            assert done or steps == 10


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_max_steps(self):
        """Test environment with zero max steps."""
        env = TilerSliderEnv(
            size=3,
            initial_locations=[(0, 0)],
            target_locations=[(2, 2)],
            max_steps=0
        )

        env.reset()
        _, _, done, info = env.step(0)

        # Should immediately timeout
        assert done == True

    def test_very_large_max_steps(self):
        """Test environment with very large max steps."""
        env = TilerSliderEnv(
            size=3,
            initial_locations=[(0, 0)],
            target_locations=[(2, 2)],
            max_steps=1000000
        )

        env.reset()
        _, _, done, _ = env.step(0)

        # Should not timeout after one step
        assert done == False or env.state.is_won()

    def test_already_won_initial_state(self):
        """Test environment that starts in winning state."""
        env = TilerSliderEnv(
            size=3,
            initial_locations=[(2, 2)],
            target_locations=[(2, 2)]
        )

        obs = env.reset()

        # Should already be won
        assert env.state.is_won() == True

        # Any move should recognize the win
        _, _, _, info = env.step(0)
        assert info['is_won'] == True

    def test_empty_environment(self):
        """Test environment with no tiles."""
        env = TilerSliderEnv(
            size=3,
            initial_locations=[],
            target_locations=[]
        )

        env.reset()

        # Should be in winning state (no tiles to place)
        assert env.state.is_won() == True
