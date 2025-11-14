"""
Integration tests migrated from user's original test cases.

These tests verify that the refactored GameState behaves identically
to the original GameEnvironment implementation.
"""

from explainrl.environment import GameState, ImageLoader


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

    # Create state from level
    state = GameState(
        size=test_level.size,
        blocked_locations=test_level.blocked_locations,
        initial_locations=test_level.initial_locations,
        target_locations=test_level.target_locations,
        multi_color=test_level.multiple_colors
    )

    # Initial state: tile a at (0,3), tile b at (3,2)
    # Target A at (0,0), target B at (3,0)
    assert state.render() == "A..a\nX...\n...X\nB.b.\n"

    # Move RIGHT: tile a stays at (0,3), tile b moves to (3,3)
    assert state.move(GameState.Move.RIGHT) is False
    assert state.render() == "A..a\nX...\n...X\nB..b\n"

    # Move DOWN: tile a moves to (1,3), tile b stays at (3,3)
    assert state.move(GameState.Move.DOWN) is False
    assert state.render() == "A...\nX..a\n...X\nB..b\n"

    # Move LEFT: tile a moves to (1,1), tile b moves to (3,0)
    # Note: tile b is now at target B, so only 'B' shows (target displayed over tile)
    assert state.move(GameState.Move.LEFT) is False
    assert state.render() == "A...\nXa..\n...X\nB...\n"

    # Move UP: tile a moves to (0,1), tile b moves to (2,0)
    assert state.move(GameState.Move.UP) is False
    assert state.render() == "Aa..\nX...\nb..X\nB...\n"

    # Move LEFT: tile a moves to (0,0), tile b stays at (2,0)
    # Note: tile a is now at target A, so only 'A' shows
    assert state.move(GameState.Move.LEFT) is False
    assert state.render() == "A...\nX...\nb..X\nB...\n"

    # Move DOWN: tile a stays at (0,0), tile b moves to (3,0) - WIN!
    # Both tiles are now at their targets
    assert state.move(GameState.Move.DOWN) is True
    assert state.render() == "A...\nX...\n...X\nB...\n"


def test_game_sequence_2():
    """
    Test case 2: Movement with repeated invalid moves.

    This test verifies:
    - Multiple tiles moving in same direction
    - Tiles reaching edge and staying there
    - Invalid moves (no state change) return False
    - Rendering accuracy throughout
    """
    test_level = ImageLoader.ImageProcessed(
        size=4,
        blocked_locations=[(1, 0), (2, 3)],
        initial_locations=[(0, 3), (3, 2)],
        target_locations=[(0, 0), (3, 0)],
        multiple_colors=True
    )

    state = GameState(
        size=test_level.size,
        blocked_locations=test_level.blocked_locations,
        initial_locations=test_level.initial_locations,
        target_locations=test_level.target_locations,
        multi_color=test_level.multiple_colors
    )

    # Initial state
    assert state.render() == "A..a\nX...\n...X\nB.b.\n"

    # Move DOWN: tile a moves to (1,3), tile b stays at (3,2)
    assert state.move(GameState.Move.DOWN) is False
    assert state.render() == "A...\nX..a\n...X\nB.b.\n"

    # Move LEFT: tile a moves to (1,1), tile b moves to (3,0)
    assert state.move(GameState.Move.LEFT) is False
    assert state.render() == "A...\nXa..\n...X\nB...\n"

    # Move DOWN: tile a moves to (3,1), tile b stays at (3,0)
    assert state.move(GameState.Move.DOWN) is False
    assert state.render() == "A...\nX...\n...X\nBa..\n"

    # Move RIGHT: tile a moves to (3,2), tile b stays at (3,0)
    assert state.move(GameState.Move.RIGHT) is False
    assert state.render() == "A...\nX...\n...X\nB.ba\n"

    # Move RIGHT again: tile a tries to move but hits blocked cell at (2,3), stays at (3,2)
    # Actually, tile a moves to (3,3) first, then gets pushed back? Let me trace:
    # Wait, the expected output is the same, so tile a must stay at (3,2)
    # Actually looking at the precomputation logic, from (3,2) moving RIGHT:
    # - Check (3,3): not blocked, so can go there... unless blocked at (2,3) affects it?
    # No wait, (2,3) is blocked, but that's a different row.
    # From (3,2), moving RIGHT should go to (3,3).
    # But the test expects no change. Let me check if there's tile collision.
    # Tile b is at (3,0), so no collision at (3,3).
    # Hmm, this is odd. Let me re-read the test.
    # Oh wait, looking at the previous move, after moving RIGHT, the state is "B.ba\n"
    # This means tile a is at (3,3) already! Not (3,2).
    # So the second RIGHT move from (3,3) should indeed not change anything (already at edge).
    assert state.move(GameState.Move.RIGHT) is False
    assert state.render() == "A...\nX...\n...X\nB.ba\n"

    # Move LEFT: tile a moves to (3,0)? No, tile b is there. So moves to (3,1)?
    # Let's think: processing order for LEFT is left-to-right.
    # Tile b at (3,0) moves LEFT -> stays at (3,0) (edge)
    # Tile a at (3,3) moves LEFT -> should go to (3,0) but blocked by tile b, so goes to (3,1)
    # Wait, that's not right either. Let me re-check the move logic.
    # Actually, looking at the move() method, it processes tiles in a specific order.
    # For LEFT, it sorts by column (ascending), so processes leftmost tiles first.
    # - Tile b at (3,0) is processed first: moves LEFT -> stays at (3,0)
    # - Tile a at (3,3) is processed second: moves LEFT -> destination is (3,0), but occupied, so backs up to (3,1)
    # Expected: "A...\nX...\n...X\nBa..\n"
    # This shows tile b at (3,0) and tile a at (3,1). Correct!
    assert state.move(GameState.Move.LEFT) is False
    assert state.render() == "A...\nX...\n...X\nBa..\n"


if __name__ == "__main__":
    print("Running test_game_sequence_1...")
    test_game_sequence_1()
    print("✓ test_game_sequence_1 passed!")

    print("\nRunning test_game_sequence_2...")
    test_game_sequence_2()
    print("✓ test_game_sequence_2 passed!")

    print("\n✓ All tests passed!")
