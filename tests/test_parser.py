import pytest

from pgtg.constants import OBSTACLE_NAMES, TILE_HEIGHT, TILE_WIDTH
from pgtg.map_generator import MapPlan
from pgtg.parser import find_direction, parse_map_object, parse_tile_map_to_graph


@pytest.mark.parametrize(
    "map_plan",
    [
        "map_plan_3x1_with_traffic",
        "map_plan_with_traffic_and_all_deadends",
        "map_plan_2x2_with_obstacles",
    ],
)
class TestParseMap:
    def test_dimensions(self, map_plan, request):
        map_plan = request.getfixturevalue(map_plan)
        width, height, map, _, _ = parse_map_object(map_plan)

        assert len(map) == map_plan.width * TILE_WIDTH
        assert len(map[0]) == map_plan.height * TILE_HEIGHT

        assert len(map) == width
        assert len(map[0]) == height

    def test_subgoals(self, map_plan, request):
        map_plan = request.getfixturevalue(map_plan)
        width, height, map, num_subgoals, _ = parse_map_object(map_plan)

        num_subgoal_features_in_map = sum(
            [
                "subgoal" in map[x][y] or "final goal" in map[x][y]
                for x in range(width)
                for y in range(height)
            ]
        )

        assert (
            num_subgoals == num_subgoal_features_in_map / 3
        )  # divide by 3 because there are 3 subgoal features for each subgoal


class TestWalls:
    def test_walls_on_straight_tiles(self, map_plan_3x1_with_traffic):
        _, _, map, _, _ = parse_map_object(map_plan_3x1_with_traffic)

        # walls on the top three tiles
        assert all(["wall" in map[x][y] for x in range(len(map)) for y in range(3)])
        # walls on the bottom three tiles
        assert all(["wall" in map[x][y] for x in range(len(map)) for y in range(6, 9)])

    @pytest.mark.parametrize(
        "map_plan",
        [
            "map_plan_3x1_with_traffic",
            "map_plan_with_traffic_and_all_deadends",
            "map_plan_2x2_with_obstacles",
        ],
    )
    def test_never_walls_on_the_center(self, map_plan, request):
        map_plan = request.getfixturevalue(map_plan)

        _, _, map, _, _ = parse_map_object(map_plan)

        center_square_offsets = [(3, 4), (4, 3), (4, 4), (4, 5), (5, 4)]

        for tile_x in range(map_plan.width):
            for tile_y in range(map_plan.height):
                for x_offset, y_offset in center_square_offsets:
                    assert not "wall" in map[tile_x * TILE_WIDTH + x_offset][
                        tile_y * TILE_HEIGHT + y_offset
                    ] or map_plan.tiles[tile_y][tile_x]["exits"] == [0, 0, 0, 0]

    @pytest.mark.parametrize(
        "map_plan",
        [
            "map_plan_3x1_with_traffic",
            "map_plan_with_traffic_and_all_deadends",
            "map_plan_2x2_with_obstacles",
        ],
    )
    def test_always_walls_on_the_corners(self, map_plan, request):
        map_plan = request.getfixturevalue(map_plan)

        _, _, map, _, _ = parse_map_object(map_plan)

        # the offsets to the corner squares that always have walls from the top left corner
        corner_square_offsets = [
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 1),
            (1, 2),
            (2, 0),
            (2, 1),
            (6, 0),
            (6, 1),
            (7, 0),
            (7, 1),
            (7, 2),
            (8, 0),
            (8, 1),
            (8, 2),
            (0, 6),
            (0, 7),
            (0, 8),
            (1, 6),
            (1, 7),
            (1, 8),
            (2, 7),
            (2, 8),
            (6, 7),
            (6, 8),
            (7, 6),
            (7, 7),
            (7, 8),
            (8, 6),
            (8, 7),
            (8, 8),
        ]

        for tile_x in range(map_plan.width):
            for tile_y in range(map_plan.height):
                for x_offset, y_offset in corner_square_offsets:
                    assert (
                        "wall"
                        in map[tile_x * TILE_WIDTH + x_offset][
                            tile_y * TILE_HEIGHT + y_offset
                        ]
                    )


class TestObstacles:
    def test_no_obstacles_on_walls(self, map_plan_2x2_with_obstacles):
        _, _, map, _, _ = parse_map_object(map_plan_2x2_with_obstacles)

        squares_with_walls = [
            square for colum in map for square in colum if "wall" in square
        ]

        assert not any(
            [
                any([obstacle in square for obstacle in OBSTACLE_NAMES])
                for square in squares_with_walls
            ]
        )

    def test_obstacles_in_the_correct_tiles(self, map_plan_2x2_with_obstacles):
        _, _, map, _, _ = parse_map_object(map_plan_2x2_with_obstacles)

        top_left_map_quarter = [colum[:TILE_HEIGHT] for colum in map[:TILE_WIDTH]]
        top_right_map_quarter = [colum[:TILE_HEIGHT] for colum in map[-TILE_WIDTH:]]
        bottom_left_map_quarter = [colum[-TILE_HEIGHT:] for colum in map[:TILE_WIDTH]]
        bottom_right_map_quarter = [colum[-TILE_HEIGHT:] for colum in map[-TILE_WIDTH:]]

        all_features_in_top_left_quarter = {
            feature
            for colum in top_left_map_quarter
            for square in colum
            for feature in square
        }
        all_features_in_top_right_quarter = {
            feature
            for colum in top_right_map_quarter
            for square in colum
            for feature in square
        }
        all_features_in_bottom_left_quarter = {
            feature
            for colum in bottom_left_map_quarter
            for square in colum
            for feature in square
        }
        all_features_in_bottom_right_quarter = {
            feature
            for colum in bottom_right_map_quarter
            for square in colum
            for feature in square
        }

        assert "ice" in all_features_in_top_left_quarter
        assert not any(
            obstacles in all_features_in_top_left_quarter
            for obstacles in set(OBSTACLE_NAMES) - {"ice"}
        )

        assert "broken road" in all_features_in_top_right_quarter
        assert not any(
            obstacles in all_features_in_top_right_quarter
            for obstacles in set(OBSTACLE_NAMES) - {"broken road"}
        )

        assert "sand" in all_features_in_bottom_left_quarter
        assert not any(
            obstacles in all_features_in_bottom_left_quarter
            for obstacles in set(OBSTACLE_NAMES) - {"sand"}
        )

        assert not any(
            obstacles in all_features_in_bottom_right_quarter
            for obstacles in OBSTACLE_NAMES
        )


class TestTrafficLanes:
    def test_deadend_lanes(self, map_plan_with_traffic_and_all_deadends):
        _, _, map, _, _ = parse_map_object(map_plan_with_traffic_and_all_deadends)

        # deadend north
        assert all(
            [
                "car_lane middle_to_north up" in tile
                for tile in map[36 + 5][9 : 9 + 5 + 1]
            ]
        )
        # deadend east
        assert all(
            [
                "car_lane middle_to_east right" in tile
                for tile in [row[18 + 5] for row in map[9 + 3 : 9 + 9]]
            ]
        )
        # deadend south
        assert all(
            [
                "car_lane middle_to_south down" in tile
                for tile in map[3][27 + 3 : 27 + 9]
            ]
        )
        # deadend west
        assert all(
            [
                "car_lane middle_to_west left" in tile
                for tile in [row[18 + 3] for row in map[18 + 9 : 27 + 6]]
            ]
        )

    def test_deadend_spawner(self, map_plan_with_traffic_and_all_deadends):
        _, _, map, _, _ = parse_map_object(map_plan_with_traffic_and_all_deadends)

        # deadend north
        assert "car_spawner" in map[36 + 5][9 + 5]
        # deadend east
        assert "car_spawner" in map[9 + 3][18 + 5]
        # deadend south
        assert "car_spawner" in map[3][27 + 3]
        # deadend west
        assert "car_spawner" in map[27 + 5][18 + 3]

    def test_border_spawners(self, map_plan_3x1_with_traffic):
        _, _, map, _, _ = parse_map_object(map_plan_3x1_with_traffic)

        # spawner at the start
        assert "car_spawner" in map[0][5]

        # spawner at the goal
        assert "car_spawner" in map[26][3]


class TestGraphFromMapPlan:
    def test_same_number_of_tiles_and_nodes(self, map_plan_3x3):
        graph = parse_tile_map_to_graph(map_plan_3x3)

        assert len(graph.nodes()) == map_plan_3x3.width * map_plan_3x3.height

    def test_same_number_of_exits_and_edges(self, map_plan_3x3):
        graph = parse_tile_map_to_graph(map_plan_3x3)

        num_exits = (
            sum([sum(tile["exits"]) for row in map_plan_3x3.tiles for tile in row])
            - sum(
                [tile["exits"][0] == 1 for tile in map_plan_3x3.tiles[0]]
            )  # remove the exits that are on the top border
            - sum(
                [tile["exits"][2] == 1 for tile in map_plan_3x3.tiles[-1]]
            )  # remove the exits that are on the bottom border
            - sum(
                [row[-1]["exits"][1] == 1 for row in map_plan_3x3.tiles]
            )  # remove the exits that are on the right border
            - sum(
                [row[0]["exits"][3] == 1 for row in map_plan_3x3.tiles]
            )  # remove the exits that are on the left border
        ) / 2  # divide by 2 because each exit appears on two tiles

        num_edges = (
            len(graph.edges()) / 2
        )  # divide by 2 because the graph is undirected and each edge appears twice

        assert num_exits == num_edges


class TestDirection:
    @pytest.mark.parametrize(
        "coordinates_1, coordinates_2, direction_letter",
        [
            ((0, 0), (0, -1), "north"),
            ((0, 0), (1, 0), "east"),
            ((0, 0), (0, 1), "south"),
            ((0, 0), (-1, 0), "west"),
            ((3, 4), (3, 3), "north"),
            ((2, 3), (3, 3), "east"),
            ((3, 2), (3, 3), "south"),
            ((4, 3), (3, 3), "west"),
            ((1, 2), (1, -1), "north"),
            ((3, 4), (6, 4), "east"),
            ((5, 6), (5, 9), "south"),
            ((7, 8), (4, 8), "west"),
        ],
    )
    def test_find_direction(self, coordinates_1, coordinates_2, direction_letter):
        assert find_direction(coordinates_1, coordinates_2) == direction_letter

    @pytest.mark.parametrize(
        "coordinates_1, coordinates_2",
        [
            ((0, 0), (0, 0)),
            ((1, 2), (1, 2)),
            ((0, 0), (1, 1)),
            ((5, 6), (7, 8)),
            ((3, 3), (-3, -3)),
        ],
    )
    def test_find_direction_raises(self, coordinates_1, coordinates_2):
        with pytest.raises(ValueError):
            find_direction(coordinates_1, coordinates_2)


@pytest.fixture
def map_plan_3x3():
    return MapPlan(
        width=3,
        height=3,
        tiles=[
            [
                {"exits": [0, 1, 1, 0]},
                {"exits": [0, 1, 0, 1]},
                {"exits": [0, 1, 0, 1]},
            ],
            [
                {"exits": [1, 0, 1, 0]},
                {"exits": [0, 0, 0, 0]},
                {"exits": [0, 0, 0, 0]},
            ],
            [
                {"exits": [1, 1, 0, 1]},
                {"exits": [0, 1, 0, 1]},
                {"exits": [0, 0, 0, 1]},
            ],
        ],
        start=(0, 2, "west"),
        goal=(2, 0, "east"),
    )


@pytest.fixture
def map_plan_2x2_with_obstacles():
    return MapPlan(
        width=2,
        height=2,
        tiles=[
            [
                {
                    "exits": [0, 1, 1, 0],
                    "obstacle_type": "ice",
                    "obstacle_mask": "bottom_half",
                },
                {
                    "exits": [0, 1, 1, 1],
                    "obstacle_type": "broken road",
                    "obstacle_mask": "top_half",
                },
            ],
            [
                {
                    "exits": [1, 1, 0, 1],
                    "obstacle_type": "sand",
                    "obstacle_mask": "blob",
                },
                {"exits": [1, 0, 0, 1]},
            ],
        ],
        start=(0, 1, "west"),
        goal=(1, 0, "east"),
    )


@pytest.fixture
def map_plan_with_traffic_and_all_deadends():
    """the map looks like this:

    ......######g
    ......#....#.
    ....#####....
    .#....#......
    s######......
    """
    return MapPlan(
        width=5,
        height=5,
        tiles=[
            [
                {"exits": [0, 0, 0, 0]},
                {"exits": [0, 0, 0, 0]},
                {"exits": [0, 1, 1, 0]},
                {"exits": [0, 1, 0, 1]},
                {"exits": [0, 1, 1, 1]},
            ],
            [
                {"exits": [0, 0, 0, 0]},
                {"exits": [0, 0, 0, 0]},
                {"exits": [1, 0, 1, 0]},
                {"exits": [0, 0, 0, 0]},
                {"exits": [1, 0, 0, 0]},
            ],
            [
                {"exits": [0, 0, 0, 0]},
                {"exits": [0, 1, 0, 0]},
                {"exits": [1, 1, 1, 1]},
                {"exits": [0, 0, 0, 1]},
                {"exits": [0, 0, 0, 0]},
            ],
            [
                {"exits": [0, 0, 1, 0]},
                {"exits": [0, 0, 0, 0]},
                {"exits": [1, 0, 1, 0]},
                {"exits": [0, 0, 0, 0]},
                {"exits": [0, 0, 0, 0]},
            ],
            [
                {"exits": [1, 1, 0, 1]},
                {"exits": [0, 1, 0, 1]},
                {"exits": [1, 0, 0, 1]},
                {"exits": [0, 0, 0, 0]},
                {"exits": [0, 0, 0, 0]},
            ],
        ],
        start=(0, 4, "west"),
        goal=(4, 0, "east"),
    )


@pytest.fixture
def map_plan_3x1_with_traffic():
    """the map looks like this:
    s#g
    """
    return MapPlan(
        width=3,
        height=1,
        tiles=[
            [
                {"exits": [0, 1, 0, 1]},
                {"exits": [0, 1, 0, 1]},
                {"exits": [0, 1, 0, 1]},
            ],
        ],
        start=(0, 0, "west"),
        goal=(2, 0, "east"),
    )
