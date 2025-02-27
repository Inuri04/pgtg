import json

from pgtg import parser
from pgtg.constants import DIRECTIONS_TO_INTS, TILE_HEIGHT, TILE_WIDTH
from pgtg.map_generator import MapPlan


class EpisodeMap:
    """Class representing a map for the environment that can be modified and used for running a episode."""

    def __init__(self, map_plan: MapPlan):
        # save the original map plan to be able to later save the map to a file
        self.map_plan = map_plan

        (
            self.width,
            self.height,
            self._map,
            self.num_subgoals,
            self.tile_coordinates_to_subgoal_directions,
        ) = parser.parse_map_object(self.map_plan)

        self.tile_width = int(self.width / TILE_WIDTH)
        self.tile_height = int(self.height / TILE_HEIGHT)

        self.starters = []
        self.goals = []
        self.traffic_spawnable_positions = []
        self.car_spawners = []

        for x in range(self.width):
            for y in range(self.height):
                if self.feature_at(x, y, "start"):
                    self.starters.append((x, y))
                if any(
                    ["car_lane" in feature for feature in self.get_features_at(x, y)]
                ):
                    self.traffic_spawnable_positions.append((x, y))
                if self.feature_at(x, y, "car_spawner"):
                    self.car_spawners.append((x, y))
                if self.feature_at(x, y, "final goal"):
                    self.goals.append((x, y))

    def inside_map(self, x: int, y: int) -> bool:
        """Returns true if the position specified by the x and y coordinates is inside the map and false otherwise."""

        return not (x < 0 or y < 0 or x >= self.width or y >= self.height)

    def get_features_at(self, x: int, y: int) -> set[str]:
        """Returns a list of all features at the position specified by the x and y coordinates."""

        if x < 0 or x > self.width - 1 or y < 0 or y > self.height - 1:
            raise ValueError("coordinates are outside the map")
        return self._map[x][y]

    def set_features_at(self, x: int, y: int, features: set[str]) -> None:
        """Replaces the features at the the position specified by the x and y coordinates with the provided ones."""

        if x < 0 or x > self.width - 1 or y < 0 or y > self.height - 1:
            raise ValueError("coordinates are outside the map")
        self._map[x][y] = features

    def feature_at(self, x: int, y: int, features: str | set[str] | list[str]) -> bool:
        """Returns true if one of the specified features is present at the the position specified by the x and y coordinates and false otherwise."""

        if isinstance(features, str):
            return features in self.get_features_at(x, y)
        else:
            return not self.get_features_at(x, y).isdisjoint(features)

    def add_feature_at(self, x: int, y: int, feature: str) -> None:
        """Adds the specified feature at the the position specified by the x and y coordinates."""
        self.get_features_at(x, y).add(feature)

    def remove_feature_at(self, x: int, y: int, feature: str) -> None:
        """Removes the specified feature at the the position specified by the x and y coordinates. Doesn't raise an error if feature is not present."""

        self.get_features_at(x, y).discard(feature)

    def get_map_cutout(
        self,
        top_left_x: int,
        top_left_y: int,
        bottom_right_x: int,
        bottom_right_y: int,
        fill_squares_outside_map_with: set[str] | None = None,
    ) -> list[list[set[str]]]:
        """Returns a rectangular cutout of the map. If parts of the cutout are outside the map, those squares are optionally filled with the specified features.

        Args:
            top_left_x: The x-value of the top left corner of the cutout.
            top_left_y: The y-value of the top left corner of the cutout.
            bottom_right_x: The x-value of the bottom right corner of the cutout.
            bottom_right_y: The y-value of the bottom right corner of the cutout.
            fill_squares_outside_map_with: Default None. Squares outside the map are filled with these features. If None, the squares are left empty.

        Returns:
            A cutout from the map.
        """

        cutout = [
            [set() for _ in range(bottom_right_y - top_left_y + 1)]
            for _ in range(bottom_right_x - top_left_x + 1)
        ]

        for x in range(top_left_x, bottom_right_x + 1):
            for y in range(top_left_y, bottom_right_y + 1):
                if self.inside_map(x, y):
                    cutout[x - top_left_x][y - top_left_y] = self.get_features_at(x, y)
                else:
                    if fill_squares_outside_map_with is not None:
                        cutout[x - top_left_x][
                            y - top_left_y
                        ] = fill_squares_outside_map_with
                    else:
                        cutout[x - top_left_x][y - top_left_y] = set()

        return cutout

    def get_next_subgoal_direction(self, x: int, y: int) -> int:
        """Returns the direction of the next subgoal from the specified position or -1 if no subgoal is on that tile.

        Args:
            x: x-value of the position.
            y: y-value of the position.

        Returns:
            The direction of the next subgoal. (0: north, 1: east, 2: south, 3: west, -1: no subgoal found)
        """

        tile_x = int(x / TILE_WIDTH)
        tile_y = int(y / TILE_HEIGHT)

        direction = self.tile_coordinates_to_subgoal_directions.get(
            (tile_x, tile_y), None
        )

        if direction is None:
            return -1

        return DIRECTIONS_TO_INTS[direction]

    def set_subgoals_to_used(self, x: int, y: int) -> None:
        """Marks the subgoal at the specified coordinates as used and recursively marks all directly adjacent subgoals as used as well.

        Args:
            x: x-value of the position
            y: y-value of the position
        """

        assert self.feature_at(x, y, "subgoal"), (
            "Subgoal expected but found "
            + str(self.get_features_at(x, y))
            + " instead."
        )

        self.remove_feature_at(x, y, "subgoal")
        self.add_feature_at(x, y, "used subgoal")

        # recursively replace directly adjacent subgoal parts
        if self.feature_at(x, y + 1, "subgoal"):
            self.set_subgoals_to_used(x, y + 1)

        if self.feature_at(x, y - 1, "subgoal"):
            self.set_subgoals_to_used(x, y - 1)

        if self.feature_at(x + 1, y, "subgoal"):
            self.set_subgoals_to_used(x + 1, y)

        if self.feature_at(x - 1, y, "subgoal"):
            self.set_subgoals_to_used(x - 1, y)

    def save_map(self, path: str) -> None:
        """Saves the map as a JSON file.

        Args:
            path (String): path to the file the map should be saved in.
        """

        if not path.endswith(".json"):
            path += ".json"

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.map_plan.to_dict(), f, ensure_ascii=False, indent=4)
