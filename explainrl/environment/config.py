COLORS = {
    "SCREEN": (250, 250, 130),
    "SPACE": (230, 230, 230),
    "OBSTACLE": (105, 105, 105),
    "TILE": (138, 43, 226),
    "TARGET": (138, 43, 226),
    "COMBINED": (138, 43, 226),
    "LINE": (0, 0, 0),
}

# Color palette for multi-color mode (distinct colors for different tiles)
# Each tile index gets a unique color, shared with its matching target
TILE_COLORS = [
    (255, 99, 71),    # Tomato red
    (30, 144, 255),   # Dodger blue
    (50, 205, 50),    # Lime green
    (255, 165, 0),    # Orange
    (138, 43, 226),   # Blue violet
    (255, 20, 147),   # Deep pink
    (0, 206, 209),    # Dark turquoise
    (255, 215, 0),    # Gold
    (147, 112, 219),  # Medium purple
    (34, 139, 34),    # Forest green
]

WINDOW_TITLE = 'Tiler-Slider'
BLOCK_SIZE = 150
RADIUS = BLOCK_SIZE // 5  # radius of circle drawn inside rectangles
ANIMATION_TIME = 50  # wait time for animation
WAIT_TIME = 1000  # wait for this time after each move
LINE_WIDTH = 2  # horizontal & vertical lines drawn on board
