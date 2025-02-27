import numpy as np
import pytest

from pgtg.constants import OBSTACLE_NAMES
from pgtg.map_generator import (
    MapPlan,
    add_connections_to_borders,
    add_obstacles_to_map,
    generate_map,
    generate_map_graph,
    map_graph_to_tile_map_object,
)


class TestGenerateMap:
    def test_generate_map(self, rng):
        width = 3
        height = 3
        percentage_of_connections = 1.0

        map_plan = generate_map(width, height, percentage_of_connections, rng)

        assert map_plan.width == width
        assert len(map_plan.tiles[0]) == width

        assert map_plan.height == height
        assert len(map_plan.tiles) == height

    locations = [
        (0, 0, "north"),
        (0, 0, "west"),
        (0, 1, "west"),
        (1, 0, "north"),
        (0, 2, "south"),
        (0, 2, "west"),
        (2, 0, "north"),
        (2, 0, "east"),
        (2, 1, "east"),
        (1, 2, "south"),
        (2, 2, "east"),
        (2, 2, "south"),
    ]

    @pytest.mark.parametrize("start_position", locations)
    @pytest.mark.parametrize("goal_position", locations)
    def test_movable_start_and_goal(self, rng, start_position, goal_position):
        width = 3
        height = 3
        percentage_of_connections = 0

        if start_position == goal_position:  # start and goal must be different
            with pytest.raises(ValueError):
                map_plan = generate_map(
                    width,
                    height,
                    percentage_of_connections,
                    rng,
                    start_position=start_position,
                    goal_position=goal_position,
                )
            return

        else:
            map_plan = generate_map(
                width,
                height,
                percentage_of_connections,
                rng,
                start_position=start_position,
                goal_position=goal_position,
            )

            assert map_plan.start == start_position
            assert map_plan.goal == goal_position

    @pytest.mark.parametrize(
        "position",
        [
            (1, 1, "north"),
            (0, 0, "east"),
            (0, 1, "north"),
            (0, 1, "east"),
            (0, 1, "south"),
            (1, 0, "east"),
            (2, 2, "north"),
            (2, 2, "west"),
        ],
    )
    def test_illegal_start_or_goal_positions(self, rng, position):
        width = 3
        height = 3
        percentage_of_connections = 0

        with pytest.raises(ValueError):
            generate_map(
                width, height, percentage_of_connections, rng, start_position=position
            )

        with pytest.raises(ValueError):
            generate_map(
                width, height, percentage_of_connections, rng, goal_position=position
            )


class TestGenerateMapGraph:
    def test_generate_map_graph_with_all_connections(self, rng):
        width = 3
        height = 3
        percentage_of_connections = 1.0

        map_graph = generate_map_graph(width, height, percentage_of_connections, rng)

        num_edges = (
            len(map_graph.edges()) / 2
        )  # divide by 2 because the graph is undirected and has two edges for each connection

        assert len(map_graph.nodes()) == width * height + 2
        assert num_edges == (width - 1) * height + width * (height - 1) + 2

    def test_generate_map_graph_with_minimal_connections(self, rng):
        width = 3
        height = 3
        percentage_of_connections = 0.0

        map_graph = generate_map_graph(width, height, percentage_of_connections, rng)

        num_edges = (
            len(map_graph.edges()) / 2
        )  # divide by 2 because the graph is undirected and has two edges for each connection

        assert len(map_graph.nodes()) == width * height + 2
        # even with percentage_of_connections=0, there still is the possibility of a single path that connects every node
        assert num_edges <= width * height + 1

    @pytest.mark.parametrize("percentage_of_connections", [0.3, 0.5, 0.8])
    def test_generate_map_graph_with_some_connections(
        self, rng, percentage_of_connections
    ):
        width = 3
        height = 3

        map_graph = generate_map_graph(width, height, percentage_of_connections, rng)

        num_edges = (
            len(map_graph.edges()) / 2
        )  # divide by 2 because the graph is undirected and has two edges for each connection

        assert len(map_graph.nodes()) == width * height + 2

        expected_num_edges = (
            int(
                ((width - 1) * height + width * (height - 1))
                * percentage_of_connections
            )
            + 2
        )
        assert (
            num_edges == expected_num_edges
            or expected_num_edges < num_edges <= width * height + 1
        )

    @pytest.mark.parametrize("start_position", [(0, 0), (0, 1), (1, 0), (2, 2)])
    @pytest.mark.parametrize("goal_position", [(0, 0), (0, 1), (1, 0), (2, 2)])
    def test_movable_start_and_goal(self, rng, start_position, goal_position):
        width = 3
        height = 3
        percentage_of_connections = 0

        map_graph = generate_map_graph(
            width, height, percentage_of_connections, rng, start_position, goal_position
        )

        assert map_graph.nodes(from_node="start") == [start_position]
        assert map_graph.nodes(from_node="end") == [goal_position]


class TestMapGraphToMapPlan:
    @pytest.mark.parametrize("width, height", [(1, 1), (3, 3), (7, 7), (2, 4)])
    @pytest.mark.parametrize("percentage_of_connections", [0.0, 0.3, 0.5, 0.1])
    def test_same_number_of_nodes_and_tiles(
        self, rng, width, height, percentage_of_connections
    ):

        graph = generate_map_graph(width, height, percentage_of_connections, rng)
        map_plan = map_graph_to_tile_map_object(width, height, graph)

        assert (
            map_plan.width * map_plan.height
            == len(map_plan.tiles) * len(map_plan.tiles[0])
            == len(graph.nodes()) - 2
        )  # -2 because we exclude the start and end nodes

    @pytest.mark.parametrize("width, height", [(1, 1), (3, 3), (7, 7), (2, 4)])
    @pytest.mark.parametrize("percentage_of_connections", [0.0, 0.3, 0.5, 0.1])
    def test_same_number_of_edges_and_exits(
        self, rng, width, height, percentage_of_connections
    ):

        graph = generate_map_graph(width, height, percentage_of_connections, rng)
        map_plan = map_graph_to_tile_map_object(width, height, graph)

        num_exits = (
            sum([sum(tile["exits"]) for row in map_plan.tiles for tile in row]) - 2
        ) / 2  # -2 because we exclude the start and end nodes and divided by 2 because each exit is on two tiles
        num_edges = (
            len(graph.edges()) - 4
        ) / 2  # -4 because we exclude the start and end nodes and divide by 2 because the graph is undirected and has two edges for each connection

        assert num_exits == num_edges

    def test_removed_edges_lead_to_no_exits(self, rng):
        width = 3
        height = 3
        percentage_of_connections = 1

        graph = generate_map_graph(width, height, percentage_of_connections, rng)

        # remove edges between the nodes that will become tiles at the bottom left and bottom center
        graph.del_edge((0, 2), (1, 2))
        graph.del_edge((1, 2), (0, 2))

        # remove edges between the nodes that will become tiles at the middle and top center
        graph.del_edge((1, 1), (1, 0))
        graph.del_edge((1, 0), (1, 1))

        # remove edges between the nodes that will become tiles at the top right and top center
        graph.del_edge((1, 0), (2, 0))
        graph.del_edge((2, 0), (1, 0))

        map_plan = map_graph_to_tile_map_object(width, height, graph)

        assert map_plan.tiles[2][0]["exits"][1] == 0
        assert map_plan.tiles[2][1]["exits"][3] == 0

        assert map_plan.tiles[1][1]["exits"][0] == 0
        assert map_plan.tiles[0][1]["exits"][2] == 0

        assert map_plan.tiles[0][1]["exits"][1] == 0
        assert map_plan.tiles[0][2]["exits"][3] == 0


class TestConnectionsToBorders:
    def test_add_all_connections_to_borders_on_1x1_map(self, map_plan_1x1, rng):
        map_plan = map_plan_1x1
        add_connections_to_borders(map_plan, 1.0, rng)

        assert map_plan.tiles[0][0]["exits"] == [1, 1, 1, 1]

    def test_add_all_connections_to_borders_on_3x3_map(self, map_plan_3x3, rng):
        map_plan = map_plan_3x3
        add_connections_to_borders(map_plan, 1.0, rng)

        # connections to top border
        assert map_plan.tiles[0][0]["exits"][0] == 1
        assert map_plan.tiles[0][1]["exits"][0] == 1
        assert map_plan.tiles[0][2]["exits"][0] == 1

        # connections to right border
        assert map_plan.tiles[0][2]["exits"][1] == 1
        assert map_plan.tiles[1][2]["exits"][1] == 1
        assert map_plan.tiles[2][2]["exits"][1] == 1

        # connections to bottom border
        assert map_plan.tiles[2][0]["exits"][2] == 1
        assert map_plan.tiles[2][1]["exits"][2] == 1
        assert map_plan.tiles[2][2]["exits"][2] == 1

        # connections to left border
        assert map_plan.tiles[0][0]["exits"][3] == 1
        assert map_plan.tiles[1][0]["exits"][3] == 1
        assert map_plan.tiles[2][0]["exits"][3] == 1

    def test_add_half_connections_to_borders_on_3x3_map(self, map_plan_3x3, rng):
        map_plan = map_plan_3x3
        add_connections_to_borders(map_plan, 0.5, rng)

        # count the number of connections to borders
        assert (
            sum(
                [
                    # connections to top border
                    map_plan.tiles[0][0]["exits"][0] == 1,
                    map_plan.tiles[0][1]["exits"][0] == 1,
                    map_plan.tiles[0][2]["exits"][0] == 1,
                    # connections to right border
                    # map_object["map"][0][2]["exits"][1] == 1, # exclude, because this is the exit
                    map_plan.tiles[1][2]["exits"][1] == 1,
                    map_plan.tiles[2][2]["exits"][1] == 1,
                    # connections to bottom border
                    map_plan.tiles[2][0]["exits"][2] == 1,
                    map_plan.tiles[2][1]["exits"][2] == 1,
                    map_plan.tiles[2][2]["exits"][2] == 1,
                    # connections to left border
                    map_plan.tiles[0][0]["exits"][3] == 1,
                    map_plan.tiles[1][0]["exits"][3] == 1,
                    # map_object["map"][2][0]["exits"][3] == 1, # exclude, because this is the start
                ]
            )
            == 5
        )


class TestAddObstacles:

    def test_add_no_obstacles(self, map_plan_3x3, rng):

        map_plan = map_plan_3x3
        add_obstacles_to_map(map_plan, 0, rng)

        for row in map_plan.tiles:
            for tile in row:
                assert "obstacle_type" not in tile
                assert "obstacle_mask" not in tile

    @pytest.mark.parametrize("obstacle_probability", [0.0, 0.5, 1.0])
    def test_add_obstacles(self, map_plan_9x9_all_crossings, rng, obstacle_probability):

        map_plan = map_plan_9x9_all_crossings
        add_obstacles_to_map(map_plan, obstacle_probability, rng)

        num_obstacles = sum(
            [1 for row in map_plan.tiles for tile in row if "obstacle_type" in tile]
        )

        # since the number of obstacles is random and we only set the probability, we can only check if it is within a certain range
        assert (
            round(obstacle_probability * map_plan.width * map_plan.height * 0.75)
            <= num_obstacles
            <= round(obstacle_probability * map_plan.width * map_plan.height * 1.25)
        )

        for row in map_plan.tiles:
            for tile in row:
                if "obstacle_type" in tile:
                    assert tile["obstacle_type"] in OBSTACLE_NAMES
                    assert "obstacle_mask" in tile

    @pytest.mark.parametrize(
        "probability_weights",
        [
            {
                "ice_probability_weight": 1.0,
                "broken_road_probability_weight": 1.0,
                "sand_probability_weight": 1.0,
                "traffic_light_probability_weight": 1.0,
            },
            {
                "ice_probability_weight": 1.0,
                "broken_road_probability_weight": 0.0,
                "sand_probability_weight": 0.0,
                "traffic_light_probability_weight": 0.0,
            },
            {
                "ice_probability_weight": 0.0,
                "broken_road_probability_weight": 1.0,
                "sand_probability_weight": 0.0,
                "traffic_light_probability_weight": 0.0,
            },
            {
                "ice_probability_weight": 0.0,
                "broken_road_probability_weight": 0.0,
                "sand_probability_weight": 1.0,
                "traffic_light_probability_weight": 0.0,
            },
            {
                "ice_probability_weight": 0.0,
                "broken_road_probability_weight": 0.0,
                "sand_probability_weight": 0.0,
                "traffic_light_probability_weight": 1.0,
            },
            {
                "ice_probability_weight": 2.0,
                "broken_road_probability_weight": 0.5,
                "sand_probability_weight": 1.0,
                "traffic_light_probability_weight": 0.0,
            },
            {
                "ice_probability_weight": 0.5,
                "broken_road_probability_weight": 1.0,
                "sand_probability_weight": 2.0,
                "traffic_light_probability_weight": 1.0,
            },
        ],
    )
    def test_add_obstacles_with_custom_weights(
        self, map_plan_9x9_all_crossings, rng, probability_weights
    ):

        map_plan = map_plan_9x9_all_crossings
        add_obstacles_to_map(map_plan, 1, rng, **probability_weights)

        num_obstacles = sum(
            [1 for row in map_plan.tiles for tile in row if "obstacle_type" in tile]
        )
        assert num_obstacles == map_plan.width * map_plan.height

        num_obstacle_types = {obstacle_type: 0 for obstacle_type in OBSTACLE_NAMES}
        for row in map_plan.tiles:
            for tile in row:
                if "obstacle_type" in tile:
                    num_obstacle_types[tile["obstacle_type"]] += 1

        sum_probability_weights = sum(probability_weights.values())
        for obstacle_type in OBSTACLE_NAMES:
            expected_num_obstacles = round(
                map_plan.width
                * map_plan.height
                * (
                    probability_weights[
                        f"{obstacle_type.replace(' ','_')}_probability_weight"
                    ]
                    / sum_probability_weights
                )
            )
            # since the choice of obstacles is random and we only set the probability weights, we can only check if it is within a certain range
            assert (
                round(expected_num_obstacles * 0.5)
                <= num_obstacle_types[obstacle_type]
                <= round(expected_num_obstacles * 1.5)
            )


@pytest.fixture
def rng():
    return np.random.default_rng(seed=0)


@pytest.fixture
def map_plan_1x1():
    return MapPlan(
        width=1,
        height=1,
        tiles=[[{"exits": [0, 1, 0, 1]}]],
        start=(0, 0, "west"),
        goal=(0, 0, "east"),
    )


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
def map_plan_9x9_all_crossings():
    return MapPlan(
        width=9,
        height=9,
        tiles=[
            [
                {"exits": [1, 1, 1, 1]},
                {"exits": [1, 1, 1, 1]},
                {"exits": [1, 1, 1, 1]},
                {"exits": [1, 1, 1, 1]},
                {"exits": [1, 1, 1, 1]},
                {"exits": [1, 1, 1, 1]},
                {"exits": [1, 1, 1, 1]},
                {"exits": [1, 1, 1, 1]},
                {"exits": [1, 1, 1, 1]},
            ]
            for _ in range(9)
        ],
        start=(0, 2, "west"),
        goal=(2, 0, "east"),
    )
