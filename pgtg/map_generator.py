from dataclasses import dataclass
from typing import Any

import graph

from constants import DIRECTIONS_TO_INTS, OBSTACLE_MASK_NAMES, OBSTACLE_NAMES


@dataclass
class MapPlan:
    """Dataclass representing a map that has been generated but can't be used for running an episode yet."""

    width: int
    height: int
    tiles: list[list[dict[str, Any]]]
    start: tuple[int, int, str]
    goal: tuple[int, int, str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MapPlan":
        """Creates a MapPlan object from a dictionary."""

        return cls(
            width=data["width"],
            height=data["height"],
            tiles=data["map"],
            start=data["start"],
            goal=data["goal"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Returns a dictionary representation of the object."""

        return {
            "width": self.width,
            "height": self.height,
            "map": self.tiles,
            "start": self.start,
            "goal": self.goal,
        }


def generate_map(
    width: int,
    height: int,
    percentage_of_connections: float,
    rng,
    *,
    start_position: tuple[int, int] | tuple[int, int, str] | str = "random",
    goal_position: tuple[int, int] | tuple[int, int, str] | str = "random",
    minimum_distance_between_start_and_goal: int | None = None,
    obstacle_probability: float = 0,
    ice_probability_weight: float = 1,
    broken_road_probability_weight: float = 1,
    sand_probability_weight: float = 1,
    traffic_light_probability_weight: float = 1,
) -> MapPlan:
    """Generates a map object based on the provided parameters.

    Args:
        width: The width of the generated map object in tiles.
        height: The height of the generated map object in tiles.
        percentage_of_connections: The percentage of connections that are generated compared to all possible connection that could exist. A value of 1 lead each tile being connected to all neighbors. The start and end of the map are always connected, even if a value of 0 is chosen.
        rng: (np rng) A rng that is used for all randomness.
        start_position: The position of the start. Can be a tuple (x, y) or a tuple (x, y, direction) where direction is one of "north", "east", "south", or "west",
            if no direction is provided, a random direction is chosen. The position and diction have to match a border of the map.
            Can also be the string "random" to choose a random position and direction.
        goal_position: The position of the goal. Can be a tuple (x, y) or a tuple (x, y, direction) where direction is one of "north", "east", "south" or "west",
            if no direction is provided, a random direction is chosen. The position and diction have to match a border of the map.
            Can also be the string "random" to choose a random position and direction.
        minimum_distance_between_start_and_goal: Default None. The minimum distance between the start and the goal measured as manhattan distance.
            Can only be used if start_position and goal_position are "random". The maximum possible distance is map width + map height - 2.
        obstacle_probability: Default 0. The probability of a tile having a obstacle.
        ice_probability_weight: Default 1. Relative weight of the ice obstacle when choosing a random obstacle.
        broken_road_probability_weight: Default 1. Relative weight of the broken road obstacle when choosing a random obstacle.
        sand_probability_weight: Default 1. Relative weight of the sand obstacle when choosing a random obstacle.
        traffic_light_probability_weight: Default 1. Relative weight of the traffic light obstacle when choosing a random obstacle.

    Returns:
        A object representing a map.

    Raises:
        ValueError: start_position specifies a tile or direction that is not a map border.
        ValueError: start_position is a string other than "random".
        ValueError: goal_position specifies a tile or direction that is not a map border.
        ValueError: goal_position is a string other than "random".
        ValueError: start_position and goal_position specify the same tile and direction (the same tile and no direction / different directions is allowed, if that tile has two map borders).
        ValueError: minimum_distance_between_start_and_goal has been used but start_position and goal_position are not "random".
        ValueError: minimum_distance_between_start_and_goal is too large.
    """

    for position, name in [
        (start_position, "start_position"),
        (goal_position, "goal_position"),
    ]:
        if isinstance(position, tuple) and not (
            position[0] == 0
            or position[0] == -1
            or position[0] == width - 1
            or position[1] == 0
            or position[1] == -1
            or position[1] == height - 1
        ):
            raise ValueError(f"{name} must specify a tile on the map border.")

        if (
            isinstance(position, tuple)
            and len(position) == 3
            and not (
                (
                    (not position[2] == "north") or (position[1] == 0)
                )  # equivalent to: position[2] == "north" => position[1] == 0
                and (
                    (not position[2] == "east")
                    or (position[0] == -1 or position[0] == width - 1)
                )  # equivalent to: position[2] == "east" => position[0] == width - 1
                and (
                    (not position[2] == "south")
                    or (position[1] == -1 or position[1] == height - 1)
                )  # equivalent to: position[2] == "south" => (position[1] == -1 or position[1] == height - 1)
                and (
                    (not position[2] == "west") or (position[0] == 0)
                )  # equivalent to: position[2] == "west" => position[0] == 0
            )
        ):
            raise ValueError(f"The direction in {name} is not a map border.")

    if (
        isinstance(start_position, tuple)
        and len(start_position) == 3
        and isinstance(goal_position, tuple)
        and len(goal_position) == 3
        and start_position == goal_position
    ):
        raise ValueError(
            "start_position and goal_position can't be the same tile and direction."
        )

    if (
        minimum_distance_between_start_and_goal is not None
        and start_position != "random"
        and goal_position != "random"
    ):
        raise ValueError(
            "minimum_distance_between_start_and_goal can only be used if start_position and goal_position are 'random'."
        )

    if (
        minimum_distance_between_start_and_goal is not None
        and minimum_distance_between_start_and_goal > width + height - 2
    ):
        raise ValueError(
            "minimum_distance_between_start_and_goal can't be larger than width + height - 2."
        )

    start_position, goal_position = chose_random_start_and_goal_position_and_direction(
        width,
        height,
        rng,
        start_position,
        goal_position,
        minimum_distance_between_start_and_goal,
    )

    map_graph = generate_map_graph(
        width,
        height,
        percentage_of_connections,
        rng,
        start_position=(start_position[0], start_position[1]),
        goal_position=(goal_position[0], goal_position[1]),
    )
    map_plan = map_graph_to_tile_map_object(
        width, height, map_graph, start_position[2], goal_position[2]
    )
    add_connections_to_borders(map_plan, percentage_of_connections, rng)

    if obstacle_probability > 0:
        add_obstacles_to_map(
            map_plan,
            obstacle_probability,
            rng,
            ice_probability_weight=ice_probability_weight,
            broken_road_probability_weight=broken_road_probability_weight,
            sand_probability_weight=sand_probability_weight,
            traffic_light_probability_weight=traffic_light_probability_weight,
        )

    return map_plan


def generate_map_graph(
    width: int,
    height: int,
    percentage_of_connections: float,
    rng,
    start_position: tuple[int, int] = (0, -1),
    goal_position: tuple[int, int] = (-1, 0),
) -> graph.Graph:
    """Generates a graph representing a map based on the provided parameters.

    Args:
        width: The width of the generated map graph in tiles.
        height: The height of the generated map graph in tiles.
        percentage_of_connections: The percentage of connections that are generated compared to all possible connection that could exist. A value of 1 lead each tile being connected to all neighbors. The start and end of the map are always connected, even if a value of 0 is chosen.
        rng: (np rng) A rng that is used for all randomness.
        start_position: Default (0, -1). The position of the start. Tuple (x, y).
        goal_position: Default (-1, 0). The position of the goal. Tuple (x, y).

    Returns:
        A graph representing a map.

    Raises:
        ValueError: minimum_distance_between_start_and_goal is not None and start_position and goal_position are not "random".
        ValueError: If the minimum_distance_between_start_and_goal is too large.
    """

    map_graph = graph.Graph()

    for x in range(width):
        for y in range(height):
            if x < width - 1:
                map_graph.add_edge((x, y), (x + 1, y), 1, True)
            if y < height - 1:
                map_graph.add_edge((x, y), (x, y + 1), 1, True)

    removable_edges = [(edge[0], edge[1]) for edge in map_graph.edges()]

    start_x = start_position[0] if start_position[0] >= 0 else start_position[0] + width
    start_y = (
        start_position[1] if start_position[1] >= 0 else start_position[1] + height
    )
    start_position = (start_x, start_y)

    goal_x = goal_position[0] if goal_position[0] >= 0 else goal_position[0] + width
    goal_y = goal_position[1] if goal_position[1] >= 0 else goal_position[1] + height
    goal_position = (goal_x, goal_y)

    map_graph.add_edge("start", start_position, 1, True)
    map_graph.add_edge("end", goal_position, 1, True)

    num_edges_to_keep = round(len(removable_edges) * percentage_of_connections)

    shortest_path = map_graph.breadth_first_search("start", "end")

    # remove edges until the desired number of edges is reached or no more edges can be removed (don't count the four edges that are always there: start to first node and reverse, end to last node and reverse)
    while len(map_graph.edges()) - 4 > num_edges_to_keep and len(removable_edges) > 0:

        chosen_edge = removable_edges[rng.choice(len(removable_edges))]
        chosen_edge_reverse = tuple(reversed(chosen_edge))

        removable_edges.remove(chosen_edge)
        removable_edges.remove(chosen_edge_reverse)

        map_graph.del_edge(*chosen_edge)
        map_graph.del_edge(*chosen_edge_reverse)

        if all(x in shortest_path for x in chosen_edge):
            # it is not necessary to check if the two nodes of the chosen edge appear behind each other in the shortest path, if they didn't it wouldn't be a shortest path
            if map_graph.is_connected("start", "end"):
                shortest_path = map_graph.breadth_first_search("start", "end")
            else:
                map_graph.add_edge(*chosen_edge)
                map_graph.add_edge(*chosen_edge_reverse)

    return map_graph


def map_graph_to_tile_map_object(
    width: int,
    height: int,
    graph: graph.Graph,
    start_direction: str = "west",
    goal_direction: str = "east",
) -> MapPlan:
    """Turns a graph representing a map into a dict object representing the same map.

    Args:
        width: The width of the map in tiles.
        height: The height of the map in tiles.
        graph: The graph representing the map.
        start_direction: Default "west". The direction the start is facing. Can be one of "north", "east", "south", or "west".
        goal_direction: Default "east". The direction the goal is facing. Can be one of "north", "east", "south", or "

    Returns:
        The map object.
    """

    start_position = graph.nodes(from_node="start")[0]
    goal_position = graph.nodes(from_node="end")[0]

    map_plan = MapPlan(
        width,
        height,
        [],
        (*start_position, start_direction),
        (*goal_position, goal_direction),
    )

    for y in range(height):
        row = []
        for x in range(width):
            tile = {}
            tile["exits"] = [0, 0, 0, 0]
            neighboring_nodes = graph.nodes(from_node=(x, y))

            if neighboring_nodes == None:
                continue

            if (x, y - 1) in neighboring_nodes:  # tile to the north
                tile["exits"][0] = 1

            if (x + 1, y) in neighboring_nodes:  # tile to the east
                tile["exits"][1] = 1

            if (x, y + 1) in neighboring_nodes:  # tile to the south
                tile["exits"][2] = 1

            if (x - 1, y) in neighboring_nodes:  # tile to the west
                tile["exits"][3] = 1

            row.append(tile)
        map_plan.tiles.append(row)

    start_x, start_y = start_position
    map_plan.tiles[start_y][start_x]["exits"][
        DIRECTIONS_TO_INTS[start_direction]
    ] = 1  # add the start exit
    goal_x, goal_y = goal_position
    map_plan.tiles[goal_y][goal_x]["exits"][
        DIRECTIONS_TO_INTS[goal_direction]
    ] = 1  # add the end exit

    return map_plan


def add_connections_to_borders(
    map: MapPlan, percentage_of_connections_to_edges: float, rng
) -> None:
    """Given a map object adds connections from the tiles next to the borders to the borders.

    Args:
        map_object: The map object that the connections will be added to.
        percentage_of_connections: The percentage of connections that are generated compared to all possible connection that could exist. A value of 1 lead each tile next to a border being connected to said border and a value of 0 leads to none being connected (except start and end).
        rng: (np rng) A rng that is used for all randomness.
    """

    width = map.width
    height = map.height

    # list of possible connections to edges as (tile_y, tile_x, direction) with direction: 0=north/top 1=east/right 2=south/bottom 3=west/left
    possible_connections_to_borders = (
        [(0, x, 0) for x in range(width)]  # connections to top edge
        + [(y, width - 1, 1) for y in range(height)]  # connections to right edge
        + [(height - 1, x, 2) for x in range(width)]  # connections to bottom edge
        + [(y, 0, 3) for y in range(height)]  # connection to left edge
    )

    possible_connections_to_borders.remove((height - 1, 0, 3))  # remove the start
    possible_connections_to_borders.remove((0, width - 1, 1))  # remove the goal

    num_connections_to_borders_to_add = round(
        len(possible_connections_to_borders) * percentage_of_connections_to_edges
    )

    for _ in range(num_connections_to_borders_to_add):
        connection_to_border_to_add = tuple(rng.choice(possible_connections_to_borders))
        possible_connections_to_borders.remove(connection_to_border_to_add)
        map.tiles[connection_to_border_to_add[0]][connection_to_border_to_add[1]][
            "exits"
        ][connection_to_border_to_add[2]] = 1


def add_obstacles_to_map(
    map: MapPlan,
    obstacle_probability: float,
    rng,
    *,
    ice_probability_weight: float = 1,
    broken_road_probability_weight: float = 1,
    sand_probability_weight: float = 1,
    traffic_light_probability_weight: float = 1,
) -> None:
    """Given a map object adds obstacles to the tiles.

    Args:
        map_object: The map object that the obstacles will be added to.
        obstacle_probability: The probability of adding a obstacle to a tile.
        rng: (np rng) A rng that is used for all randomness.
        ice_probability_weight: Default 1. Relative weight of the ice obstacle when choosing a random obstacle.
        broken_road_probability_weight: Default 1. Relative weight of the broken road obstacle when choosing a random obstacle.
        sand_probability_weight: Default 1. Relative weight of the sand obstacle when choosing a random obstacle.
        traffic_light_probability_weight: Default 1. Relative weight of the traffic light obstacle when choosing a random obstacle.
    """

    probability_weight_sum = (
        ice_probability_weight
        + broken_road_probability_weight
        + sand_probability_weight
        + traffic_light_probability_weight
    )

    ice_relative_probability_weight = ice_probability_weight / probability_weight_sum
    broken_road_relative_probability_weight = (
        broken_road_probability_weight / probability_weight_sum
    )
    sand_relative_probability_weight = sand_probability_weight / probability_weight_sum
    traffic_light_relative_probability_weight = (
        traffic_light_probability_weight / probability_weight_sum
    )

    for row in range(map.height):
        for column in range(map.width):
            if (
                rng.random() < obstacle_probability
                and not str(map.tiles[row][column]["exits"]) == "[0, 0, 0, 0]"
            ):
                obstacle_type = rng.choice(
                    OBSTACLE_NAMES,
                    p=[
                        ice_relative_probability_weight,
                        broken_road_relative_probability_weight,
                        sand_relative_probability_weight,
                        traffic_light_relative_probability_weight,
                    ],
                )
                map.tiles[row][column]["obstacle_type"] = obstacle_type

                if obstacle_type != "traffic_light":
                    map.tiles[row][column]["obstacle_mask"] = rng.choice(
                        OBSTACLE_MASK_NAMES
                    )

                else:
                    possible_traffic_light_obstacle_masks = []
                    if map.tiles[row][column]["exits"][0]:
                        possible_traffic_light_obstacle_masks.append(
                            "traffic_light_north"
                        )
                    if map.tiles[row][column]["exits"][1]:
                        possible_traffic_light_obstacle_masks.append(
                            "traffic_light_east"
                        )
                    if map.tiles[row][column]["exits"][2]:
                        possible_traffic_light_obstacle_masks.append(
                            "traffic_light_south"
                        )
                    if map.tiles[row][column]["exits"][3]:
                        possible_traffic_light_obstacle_masks.append(
                            "traffic_light_west"
                        )

                    if (
                        map.tiles[row][column]["exits"][0]
                        and map.tiles[row][column]["exits"][2]
                        and sum(map.tiles[row][column]["exits"]) >= 3
                    ):
                        possible_traffic_light_obstacle_masks.append(
                            "traffic_light_north_and_south"
                        )
                    if (
                        map.tiles[row][column]["exits"][1]
                        and map.tiles[row][column]["exits"][3]
                        and sum(map.tiles[row][column]["exits"]) >= 3
                    ):
                        possible_traffic_light_obstacle_masks.append(
                            "traffic_light_east_and_west"
                        )

                    map.tiles[row][column]["obstacle_mask"] = rng.choice(
                        possible_traffic_light_obstacle_masks
                    )


def chose_random_start_and_goal_position_and_direction(
    width: int,
    height: int,
    rng,
    start_position: tuple[int, int] | tuple[int, int, str] | str,
    goal_position: tuple[int, int] | tuple[int, int, str] | str,
    minimum_distance_between_start_and_goal: int | None,
) -> tuple[tuple[int, int, str], tuple[int, int, str]]:
    """Choses random start and goal positions and directions for a map.

    If the start or goal position is a tuple (x, y, direction), it is not changed. If it is a tuple (x, y) a random applicable direction is chosen and added.
    If it is "random" a random position and direction is chosen.

    Args:
        width: The width of the map.
        height: The height of the map.
        rng: (np rng) A rng that is used for all randomness.
        start_position: The position of the start. Tuple (x, y) or (x, y, direction) or the string "random".
        goal_position: The position of the goal. Tuple (x, y) or (x, y, direction) or the string "random".
        minimum_distance_between_start_and_goal: The minimum distance between the start and the goal. Can only be used if start_position and goal_position are "random".
            If None, no minimum distance is enforced.

    Returns:
        The start and goal positions and directions as tuples (x, y, direction).
    """

    if start_position == "random":
        chosen_start = chose_random_start_or_goal_position(width, height, rng)
    else:
        if len(start_position) == 2:
            chosen_start = (
                start_position[0] if start_position[0] != -1 else width - 1,
                start_position[1] if start_position[1] != -1 else height - 1,
            )
        else:
            chosen_start = (
                start_position[0] if start_position[0] != -1 else width - 1,
                start_position[1] if start_position[1] != -1 else height - 1,
                start_position[2],
            )

    if goal_position == "random":
        chosen_goal = chose_random_start_or_goal_position(width, height, rng)
    else:
        if len(goal_position) == 2:
            chosen_goal = (
                goal_position[0] if goal_position[0] != -1 else width - 1,
                goal_position[1] if goal_position[1] != -1 else height - 1,
            )
        else:
            chosen_goal = (
                goal_position[0] if goal_position[0] != -1 else width - 1,
                goal_position[1] if goal_position[1] != -1 else height - 1,
                goal_position[2],
            )

    if minimum_distance_between_start_and_goal is not None:
        while (
            abs(chosen_start[0] - chosen_goal[0])
            + abs(chosen_start[1] - chosen_goal[1])
            < minimum_distance_between_start_and_goal
        ):
            chosen_start = chose_random_start_or_goal_position(width, height, rng)
            chosen_goal = chose_random_start_or_goal_position(width, height, rng)

    if len(chosen_start) == 2:
        chosen_start = (
            *chosen_start,
            chose_random_start_or_goal_direction(width, height, rng, chosen_start),
        )

    if len(chosen_goal) == 2:
        chosen_goal = (
            *chosen_goal,
            chose_random_start_or_goal_direction(width, height, rng, chosen_goal),
        )

    while chosen_start == chosen_goal:
        if start_position == "random":
            chosen_start = chose_random_start_or_goal_position(width, height, rng)
        if start_position == "random" or len(start_position) == 2:
            chosen_start = (
                chosen_start[0],
                chosen_start[1],
                chose_random_start_or_goal_direction(width, height, rng, chosen_start),
            )

        if goal_position == "random":
            chosen_goal = chose_random_start_or_goal_position(width, height, rng)
        if goal_position == "random" or len(goal_position) == 2:
            chosen_goal = (
                chosen_goal[0],
                chosen_goal[1],
                chose_random_start_or_goal_direction(width, height, rng, chosen_goal),
            )

    return chosen_start, chosen_goal


def chose_random_start_or_goal_direction(
    width: int, height: int, rng, position: tuple[int, int]
) -> str:
    """Choses a random applicable direction for a start or goal position.

    Args:
        width: The width of the map.
        height: The height of the map.
        rng: (np rng) A rng that is used for all randomness.
        position: The position of a start or goal. Tuple (x, y).

    Returns:
        The direction as a string.
    """

    possible_directions = []
    if position[1] == 0:
        possible_directions.append("north")
    if position[0] == width - 1:
        possible_directions.append("east")
    if position[1] == height - 1:
        possible_directions.append("south")
    if position[0] == 0:
        possible_directions.append("west")

    return rng.choice(possible_directions)


def chose_random_start_or_goal_position(
    width: int,
    height: int,
    rng,
) -> tuple[int, int]:
    """Choses random start and goal positions for a map.

    Args:
        width: The width of the map.
        height: The height of the map.
        rng: (np rng) A rng that is used for all randomness.

    Returns:
        The start and goal positions as tuples (x, y, direction).
    """

    match rng.integers(0, 4):
        case 0:
            return (rng.integers(0, width), 0)
        case 1:
            return (width - 1, rng.integers(0, height))
        case 2:
            return (rng.integers(0, width), height - 1)
        case 3:
            return (0, rng.integers(0, height))
