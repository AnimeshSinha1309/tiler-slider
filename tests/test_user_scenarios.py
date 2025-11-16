"""
Integration tests migrated from user's original test cases.

These tests verify that the refactored GameState behaves identically
to the original GameEnvironment implementation.
"""

from explainrl.environment import GameState, ImageLoader, TilerSliderEnv, TextRender


def test_game_sequence_1():
    """
    Test case 1: Complex movement sequence with tile collisions.

    This test verifies:
    - Tile movement in all 4 directions
    - Tile collision handling
    - Multi-color mode (ordered tile-target matching)
    - Win detection
    - Rendering with tiles overlapping targets
    """
    test_level = ImageLoader.ImageProcessed(
        size=4,
        blocked_locations=[(1, 0), (2, 3)],
        initial_locations=[(0, 3), (3, 2)],
        target_locations=[(0, 0), (3, 0)],
        multiple_colors=True
    )

    # Create environment from level
    env = TilerSliderEnv.from_level(test_level)
    renderer = TextRender(env)
    env.reset()

    # Initial state: tile a at (0,3), tile b at (3,2)
    # Target A at (0,0), target B at (3,0)
    assert renderer.render(show_info=False) == "A..a\nX...\n...X\nB.b."

    # Move RIGHT: tile a stays at (0,3), tile b moves to (3,3)
    obs, done, info = env.step(GameState.Move.RIGHT)
    assert done is False
    assert renderer.render(show_info=False) == "A..a\nX...\n...X\nB..b"

    # Move DOWN: tile a moves to (1,3), tile b stays at (3,3)
    obs, done, info = env.step(GameState.Move.DOWN)
    assert done is False
    assert renderer.render(show_info=False) == "A...\nX..a\n...X\nB..b"

    # Move LEFT: tile a moves to (1,1), tile b moves to (3,0)
    obs, done, info = env.step(GameState.Move.LEFT)
    assert done is False
    assert renderer.render(show_info=False) == "A...\nXa..\n...X\nB..."

    # Move UP: tile a moves to (0,1), tile b moves to (2,0)
    obs, done, info = env.step(GameState.Move.UP)
    assert done is False
    assert renderer.render(show_info=False) == "Aa..\nX...\nb..X\nB..."

    # Move LEFT: tile a moves to (0,0), tile b stays at (2,0)
    obs, done, info = env.step(GameState.Move.LEFT)
    assert done is False
    assert renderer.render(show_info=False) == "A...\nX...\nb..X\nB..."

    # Move DOWN: tile a stays at (0,0), tile b moves to (3,0) - WIN!
    obs, done, info = env.step(GameState.Move.DOWN)
    assert done is True
    assert info['is_won'] is True
    assert renderer.render(show_info=False) == "A...\nX...\n...X\nB..."


def test_game_sequence_2():
    """
    Test case 2: Movement with repeated invalid moves.

    This test verifies:
    - Multiple tiles moving in same direction
    - Tiles reaching edge and staying there
    - Invalid moves (no state change) return info
    - Rendering accuracy throughout
    """
    test_level = ImageLoader.ImageProcessed(
        size=4,
        blocked_locations=[(1, 0), (2, 3)],
        initial_locations=[(0, 3), (3, 2)],
        target_locations=[(0, 0), (3, 0)],
        multiple_colors=True
    )

    env = TilerSliderEnv.from_level(test_level)
    renderer = TextRender(env)
    env.reset()

    # Initial state
    assert renderer.render(show_info=False) == "A..a\nX...\n...X\nB.b."

    # Move DOWN: tile a moves to (1,3), tile b stays at (3,2)
    obs, done, info = env.step(GameState.Move.DOWN)
    assert done is False
    assert renderer.render(show_info=False) == "A...\nX..a\n...X\nB.b."

    # Move LEFT: tile a moves to (1,1), tile b moves to (3,0)
    obs, done, info = env.step(GameState.Move.LEFT)
    assert done is False
    assert renderer.render(show_info=False) == "A...\nXa..\n...X\nB..."

    # Move DOWN: tile a moves to (3,1), tile b stays at (3,0)
    obs, done, info = env.step(GameState.Move.DOWN)
    assert done is False
    assert renderer.render(show_info=False) == "A...\nX...\n...X\nBa.."

    # Move RIGHT: tile a moves to (3,2), tile b stays at (3,0)
    obs, done, info = env.step(GameState.Move.RIGHT)
    assert done is False
    # After moving right, tile 'a' should be at (3,3), tile 'b' at (3,2)?
    # Let me check: from (3,1), moving RIGHT with tile b at (3,0)
    # Actually tile a at (3,1) slides right, and based on original test expects "B.ba"
    # which means B at (3,0), then '.', then 'b' at (3,2), then 'a' at (3,3)
    assert renderer.render(show_info=False) == "A...\nX...\n...X\nB.ba"

    # Move RIGHT again: tile a tries to move but is at edge, stays at (3,3)
    obs, done, info = env.step(GameState.Move.RIGHT)
    assert done is False
    assert renderer.render(show_info=False) == "A...\nX...\n...X\nB.ba"

    # Move LEFT: tile b at (3,0) stays, tile a at (3,3) moves to (3,1)
    obs, done, info = env.step(GameState.Move.LEFT)
    assert done is False
    assert renderer.render(show_info=False) == "A...\nX...\n...X\nBa.."


if __name__ == "__main__":
    print("Running test_game_sequence_1...")
    test_game_sequence_1()
    print("✓ test_game_sequence_1 passed!")

    print("\nRunning test_game_sequence_2...")
    test_game_sequence_2()
    print("✓ test_game_sequence_2 passed!")

    print("\n✓ All tests passed!")
