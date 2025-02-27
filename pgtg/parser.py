import copy
import json
import os
from pathlib import Path

import graph

from pgtg.constants import OBSTACLE_NAMES, TILE_HEIGHT, TILE_WIDTH
from pgtg.map_generator import MapPlan
from pgtg.map_tiles_data import OBSTACLE_MASKS, TILES, TRAFFIC_LANES


def parse_map_object(
    map_plan: MapPlan,
) -> tuple[int, int, list[list[set[str]]], int, dict[tuple[int, int], str]]:
    """Parses a map plan, to the important features of a episode map.

    Args:
        map_plan: The map to be parsed

    Returns:
        A tuple (width, height, map, num_subgoals, subgoal_directions), where width and height are the dimensions of the map,
        map represents the features of each tile, num_subgoals is the number of subgoals in the map, and subgoal_directions
        is a dictionary mapping tile coordinates to the direction of the subgoal on that tile.
    """

    graph = parse_tile_map_to_graph(map_plan)

    # finds the shortest path between the lower left tile and the upper right tile.
    shortest_path: list[tuple[int, int]] = graph.shortest_path(
        (map_plan.start[0], map_plan.start[1]), (map_plan.goal[0], map_plan.goal[1])
    )[1]

    # dict maps tile coordinate to the character of the exit the subgoal has to be on.
    subgoal_coordinates_to_direction: dict[tuple[int, int], str] = {}
    for i in range(len(shortest_path) - 1):
        subgoal_coordinates_to_direction.update(
            {shortest_path[i]: find_direction(shortest_path[i], shortest_path[i + 1])}
        )

    # create a empty map
    map: list[list[set[str]]] = [
        [set() for _ in range(map_plan.height * TILE_HEIGHT)]
        for _ in range(map_plan.width * TILE_WIDTH)
    ]

    for tile_x in range(map_plan.width):
        for tile_y in range(map_plan.height):

            # copy the tile to preserve the original
            current_tile = copy.deepcopy(
                TILES[tuple(map_plan.tiles[tile_y][tile_x]["exits"])]
            )

            # if the tile is on the shortest path and not the last tile replace exit markers with subgoals
            if (tile_x, tile_y) in shortest_path[:-1]:
                replace_features_in_tile(
                    current_tile,
                    "exit " + subgoal_coordinates_to_direction[(tile_x, tile_y)],
                    "subgoal",
                )

            # if the tile is the first one on the shortest past replace exit markers with start
            if (tile_x, tile_y) == shortest_path[0]:
                replace_features_in_tile(
                    current_tile,
                    "exit " + map_plan.start[2],
                    "start",
                )

            # if the tile is the last one on the shortest past replace exit markers with final goals
            if (tile_x, tile_y) == shortest_path[-1]:
                replace_features_in_tile(
                    current_tile,
                    "exit " + map_plan.goal[2],
                    "final goal",
                )

            # remove unused exit markers
            replace_features_in_tile(
                current_tile,
                "exit north",
                None,
            )
            replace_features_in_tile(
                current_tile,
                "exit east",
                None,
            )
            replace_features_in_tile(
                current_tile,
                "exit south",
                None,
            )
            replace_features_in_tile(
                current_tile,
                "exit west",
                None,
            )

            # if the tile has a obstacle type add obstacles
            if map_plan.tiles[tile_y][tile_x].get("obstacle_type") is not None:
                assert (
                    map_plan.tiles[tile_y][tile_x].get("obstacle_mask") is not None
                ), f"The tile at ({tile_x},{tile_y}) has a obstacle type without a obstacle mask"

                add_obstacles_to_tile(
                    current_tile,
                    OBSTACLE_MASKS[map_plan.tiles[tile_y][tile_x]["obstacle_mask"]],
                    map_plan.tiles[tile_y][tile_x]["obstacle_type"],
                )

            # if the tile has any exits add traffic lanes
            if map_plan.tiles[tile_y][tile_x]["exits"] != [0, 0, 0, 0]:
                add_traffic_lanes_to_tile(
                    current_tile,
                    TRAFFIC_LANES[tuple(map_plan.tiles[tile_y][tile_x]["exits"])],
                )

            # if the tile is on the border of the map add car spawners
            if tile_x == 0:
                replace_features_in_tile(
                    current_tile,
                    "car_lane all right",
                    "car_spawner",
                    keep_old_features=True,
                )
            if tile_x == map_plan.width - 1:
                replace_features_in_tile(
                    current_tile,
                    "car_lane all left",
                    "car_spawner",
                    keep_old_features=True,
                )
            if tile_y == 0:
                replace_features_in_tile(
                    current_tile,
                    "car_lane all down",
                    "car_spawner",
                    keep_old_features=True,
                )
            if tile_y == map_plan.height - 1:
                replace_features_in_tile(
                    current_tile,
                    "car_lane all up",
                    "car_spawner",
                    keep_old_features=True,
                )

            # add the tile to the map
            for square_x in range(TILE_WIDTH):
                for square_y in range(TILE_HEIGHT):
                    map[tile_x * TILE_WIDTH + square_x][
                        tile_y * TILE_HEIGHT + square_y
                    ] = current_tile[square_x][square_y]

    # add the direction of the final goal
    subgoal_coordinates_to_direction.update({shortest_path[-1]: map_plan.goal[2]})

    return (
        map_plan.width * TILE_WIDTH,
        map_plan.height * TILE_HEIGHT,
        map,
        len(subgoal_coordinates_to_direction),
        subgoal_coordinates_to_direction,
    )


def replace_features_in_tile(
    tile: list[list[set[str]]],
    old_feature: str,
    new_feature: str | None,
    keep_old_features: bool = False,
) -> None:
    """Replaces a feature in a tile with up to one new feature.

    Args:
        tile: The tile to be modified.
        old_feature: The feature to be replaced.
        new_feature: The new feature to replace the old feature with. If None, the old feature simply is removed.
    """

    for x in range(TILE_WIDTH):
        for y in range(TILE_HEIGHT):
            if old_feature in tile[x][y]:
                if not keep_old_features:
                    tile[x][y].remove(old_feature)

                if new_feature is not None:
                    tile[x][y].add(new_feature)


def add_obstacles_to_tile(
    tile: list[list[set[str]]], obstacle_mask: list[list[set[str]]], obstacle_type: str
) -> None:
    """Adds obstacles to a tile.

    Args:
        tile: The tile to be modified.
        obstacle_mask: The mask of the obstacles to be added.
        obstacle_type: The type of the obstacles to be added.
    """

    assert obstacle_type in OBSTACLE_NAMES, f"Unknown obstacle type: {obstacle_type}"

    for x in range(TILE_WIDTH):
        for y in range(TILE_HEIGHT):
            if "obstacle" in obstacle_mask[x][y] and "wall" not in tile[x][y]:
                tile[x][y].add(obstacle_type)


def add_traffic_lanes_to_tile(
    tile: list[list[set[str]]], traffic_lanes: list[list[set[str]]]
) -> None:
    """Adds traffic lanes to a tile.

    Args:
        tile: The tile to be modified.
        traffic_lanes: The traffic lanes to be added.
    """

    for x in range(TILE_WIDTH):
        for y in range(TILE_HEIGHT):
            tile[x][y].update(traffic_lanes[x][y])


def json_file_to_map_plan(path: str) -> MapPlan:
    """Creates a map plan from a JSON file.

    Args:
        path: The path to the JSON file, either absolute or relative to the current working directory.

    Returns:
        A map plan.
    """

    if not path.endswith(".json"):
        path = path + ".json"

    map_object = json.load(open(path))
    return MapPlan.from_dict(map_object)


def parse_tile_map_to_graph(tile_map: MapPlan) -> graph.Graph:
    """Generates a graph from a map plan representing the connections in the map.

    Args:
        tile_map: A map plan of the map the graph is generated for.

    Returns:
        graph: A graph representing the connections in the map.
    """

    result = graph.Graph()
    width = tile_map.width
    height = tile_map.height
    i = 0
    for row in tile_map.tiles:
        j = 0
        for node in row:
            result.add_node((j, i), node)

            if node["exits"][0] and i > 0:
                result.add_edge((j, i), (j, i - 1))

            if node["exits"][1] and j < width - 1:
                result.add_edge((j, i), (j + 1, i))

            if node["exits"][2] and i < height - 1:
                result.add_edge((j, i), (j, i + 1))

            if node["exits"][3] and j > 0:
                result.add_edge((j, i), (j - 1, i))
            j = j + 1
        i = i + 1
    return result


def find_direction(coordinates: tuple[int, int], other: tuple[int, int]) -> str:
    """Compares two tile coordinates to find out the cardinal direction they lie in relative to each other.

    Args:
        coordinates: Coordinates of the first tile.
        other: Coordinates of the second tile.

    Returns:
        The name of the cardinal direction the second tile lies in relative to the first tile.

    Raises:
        ValueError: If the two tiles are not horizontal or vertical to each other.
    """

    coordinates_x, coordinates_y = coordinates
    other_x, other_y = other

    if coordinates_y == other_y:
        if coordinates_x - other_x < 0:
            return "east"
        elif coordinates_x - other_x > 0:
            return "west"
    if coordinates_x == other_x:
        if coordinates_y - other_y < 0:
            return "south"
        elif coordinates_y - other_y > 0:
            return "north"
    raise ValueError("Not a cardinal Direction.")
