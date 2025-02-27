# the width and height of a tile in squares
TILE_WIDTH: int = 9
TILE_HEIGHT: int = 9

# dictionary mapping action numbers to (x,y) acceleration
ACTIONS_TO_ACCELERATION: dict[int, tuple[int, int]] = {
    0: (-1, -1),  # left-up
    1: (-1, 0),  # left
    2: (-1, 1),  # left-down
    3: (0, -1),  # up
    4: (0, 0),  # nothing
    5: (0, 1),  # down
    6: (1, -1),  # right-up
    7: (1, 0),  # right
    8: (1, 1),  # right-down
}

OBSTACLE_NAMES: list[str] = ["ice", "broken road", "sand", "traffic_light"]
OBSTACLE_MASK_NAMES: list[str] = [
    "blob",
    "small_blob",
    "chess_field",
    "reverse_chess_field",
    "top_half",
    "bottom_half",
    "left_half",
    "right_half",
]

# a dictionary mapping the names of cardinal directions to integers
DIRECTIONS_TO_INTS: dict[str, int] = {
    "north": 0,
    "east": 1,
    "south": 2,
    "west": 3,
}
