import copy
import warnings
from dataclasses import dataclass
from typing import Any, NamedTuple, SupportsFloat

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pygame
from gymnasium import spaces
from PIL.Image import Image

from pgtg import graphic
from pgtg.constants import (
    ACTIONS_TO_ACCELERATION,
    OBSTACLE_NAMES,
    TILE_HEIGHT,
    TILE_WIDTH,
)
from pgtg.map import EpisodeMap
from pgtg.map_generator import generate_map
from pgtg.parser import json_file_to_map_plan


def _round(x):
    return int(np.floor(x + 0.5))


class Position(NamedTuple):
    x: int
    y: int


@dataclass
class Car:
    id: int
    position: Position
    route: str


class PGTGEnv(gym.Env):
    """Class representing the modular racetrack environment."""

    metadata = {"render_modes": ["human", "rgb_array", "pil_image"], "render_fps": 4}

    def __init__(
        self,
        map_path: str | None = None,
        *,
        random_map_width: int = 4,
        random_map_height: int = 4,
        random_map_percentage_of_connections: float = 0.5,
        random_map_start_position: tuple[int, int] | tuple[int, int, str] | str = (
            0,
            -1,
            "west",
        ),
        random_map_goal_position: tuple[int, int] | tuple[int, int, str] | str = (
            -1,
            0,
            "east",
        ),
        random_map_minimum_distance_between_start_and_goal: int | None = None,
        random_map_obstacle_probability: float = 0.0,
        random_map_ice_probability_weight: float = 1,
        random_map_broken_road_probability_weight: float = 1,
        random_map_sand_probability_weight: float = 1,
        random_map_traffic_light_probability_weight: float = 1,
        render_mode: str | None = None,
        features_to_include_in_observation: list[str] = [
            "walls",
            "goals",
            "ice",
            "broken road",
            "sand",
            "traffic",
            "traffic_light_green",
            "traffic_light_yellow",
            "traffic_light_red",
        ],
        use_sliding_observation_window: bool = False,
        sliding_observation_window_size: int = 4,
        use_next_subgoal_direction: bool = False,
        sum_subgoals_reward: int = 100,
        final_goal_bonus: int = 0,
        crash_penalty: int = 100,
        traffic_light_violation_penalty: int = 50,
        standing_still_penalty: int = 0,
        already_visited_position_penalty: int = 0,
        ice_probability: float = 0.1,
        street_damage_probability: float = 0.1,
        sand_probability: float = 0.2,
        traffic_density: float = 0.0,
        traffic_light_phases_duration: tuple[int, int, int] = (10, 3, 10),
        ignore_traffic_collisions: bool = False,
    ):
        """initializes an object of the class with the given arguments.

        Args:
            map_name: Default None. Path to map file, either absolute or relative to the current working directory. If it is None a random map is generated instead.
            random_map_width: Default 4. Width of the random map.
            random_map_height: Default 4. Height of the random map.
            random_map_percentage_of_connections: Default 0.5. Percentage of connections between tiles that will be kept while generating the random map.
            random_map_start_position: Default (0, -1, "west"). Start position of the random map. Can be a tuple (x, y) or a tuple (x, y, direction)
                where direction is one of "north", "east", "south", or "west", if no direction is provided, a random applicable direction is chosen.
                The position and diction have to match a border of the map. Can also be the string "random" to choose a random position and direction.
            random_map_goal_position: Default (-1, 0, "east"). Goal position of the random map. Can be a tuple (x, y) or a tuple (x, y, direction)
                where direction is one of "north", "east", "south", or "west", if no direction is provided, a random applicable direction is chosen.
                The position and diction have to match a border of the map. Can also be the string "random" to choose a random position and direction.
            minimum_distance_between_start_and_goal: Default None. The minimum distance between the start and the goal measured as manhattan distance.
                Can only be used if start_position and goal_position are "random". The maximum possible distance is map width + map height - 2.
            random_map_obstacle_probability: Default 0.0. Probability that a tile receives an obstacle while generating the random map.
            random_map_ice_probability_weight: Default 1. Relative weight of the ice obstacle when generating the random map.
            random_map_broken_road_probability_weight: Default 1. Relative weight of the broken_road obstacle when generating the random map.
            random_map_sand_probability_weight: Default 1. Relative weight of the sand obstacle when generating the random map.
            random_map_traffic_light_probability_weight: Default 1. Relative weight of the traffic_light obstacle when generating the random map.
            render_mode: Default None. What kind of output render() generates. "human" results in no output by render() but automatic displaying during step().
            features_to_include_in_observation: Default ["wall", "goals", "ice", "broken road", "sand", "traffic", "traffic_light_green", "traffic_light_yellow", traffic_light_red"].
                The features that are included in the observation. Possible values are "wall", "goal", "ice", "broken road", "sand", "traffic", "traffic_light_green", "traffic_light_yellow", and traffic_light_red".
            use_sliding_observation_window: Default False. Whether or not to use a observation window that moves with the agent. If False instead the current tile is observed.
            sliding_observation_window_size: Default 4. The distance to the border of the moving observation window. A size of 1 results in a 3x3 window, a size of 4 in a 9x9 window and a size of 0 in a 1x1 window
                (only the currently occupied square is visible). No effect if use_sliding_observation_window is False.
            use_next_subgoal_direction: Default False. Whether or not to use the direction to the next subgoal as an additional feature in the observation.
            sum_subgoals_reward: Default 100. The sum of all subgoal rewards. Gets split evenly among all subgoals.
            final_goal_bonus: Default 0. Bonus for the final goal that gets added to the subgoal reward.
            crash_penalty: Default 100. Penalty for driving into a wall or out of the map. The value will get negated so a positive value should be used.
            traffic_light_violation_penalty: Default 50. Penalty for running a red light. The value will get negated so a positive value should be used.
            standing_still_penalty: Default 0. Penalty for not moving. Gets applied every step. The value will get negated so a positive value should be used.
            already_visited_position_penalty: Default 0. Penalty for moving to a position that was visited before. Gets applied every step for the final position of that step. Standing still does not result in penalty. The value will get negated so a positive value should be used.
            ice_probability: Default 0.1. Probability of the ice effect happening when the agent drives over it.
            street_damage_probability: Default  0.1. Probability of the broken road effect happening when the agent drives over it.
            sand_probability: Default 0.2. Probability of the sand effect happening when the agent drives over it.
            traffic_density: Default 0.0. The density of the traffic as fraction of the number of traffic cars and the number of positions where traffic could be. For a value of 0 no traffic is generated.
            traffic_light_phases_duration: Default (10, 3, 10). The duration of the green, yellow, and red phase of the traffic light in steps.
            ignore_traffic_collisions: Default False. If true collisions with traffic are ignored. For testing purposes.
        """

        if random_map_obstacle_probability > 0:
            if (
                random_map_ice_probability_weight > 0
                and "ice" not in features_to_include_in_observation
            ):
                warnings.warn(
                    "The ice obstacle is used in the map generation but not included in the observation. An agent will not be able to learn to avoid it."
                )
            if (
                random_map_broken_road_probability_weight > 0
                and "broken road" not in features_to_include_in_observation
            ):
                warnings.warn(
                    "The broken road obstacle is used in the map generation but not included in the observation. An agent will not be able to learn to avoid it."
                )
            if (
                random_map_sand_probability_weight > 0
                and "sand" not in features_to_include_in_observation
            ):
                warnings.warn(
                    "The sand obstacle is used in the map generation but not included in the observation. An agent will not be able to learn to avoid it."
                )
            if (
                random_map_traffic_light_probability_weight > 0
                and "traffic_light_green" not in features_to_include_in_observation
            ):
                warnings.warn(
                    "The traffic light obstacle is used in the map generation but green traffic lights are not included in the observation. An agent will not be able to learn to avoid it."
                )
            if (
                random_map_traffic_light_probability_weight > 0
                and "traffic_light_yellow" not in features_to_include_in_observation
            ):
                warnings.warn(
                    "The traffic light obstacle is used in the map generation but yellow traffic lights are not included in the observation. An agent will not be able to learn to avoid it."
                )
            if (
                random_map_traffic_light_probability_weight > 0
                and "traffic_light_red" not in features_to_include_in_observation
            ):
                warnings.warn(
                    "The traffic light obstacle is used in the map generation but red traffic lights are not included in the observation. An agent will not be able to learn to avoid it."
                )
        if traffic_density > 0 and "traffic" not in features_to_include_in_observation:
            warnings.warn(
                "Traffic is generated but not included in the observation. An agent will not be able to learn to avoid it."
            )

        # There are 8 different directions to accelerate into and the option to stand still.
        self.action_space = spaces.Discrete(9)

        observation_window_size = (
            (TILE_WIDTH, TILE_HEIGHT)
            if not use_sliding_observation_window
            else (
                1 + sliding_observation_window_size * 2,
                1 + sliding_observation_window_size * 2,
            )
        )

        # The agent sees the position in the current tile, its velocity and the features in that tile.
        observation_space_dict = {
            "position": spaces.MultiDiscrete([TILE_WIDTH, TILE_HEIGHT], dtype=np.int32),
            "velocity": spaces.Box(low=-99, high=99, shape=(2,), dtype=np.int32),
            "map": spaces.Dict(
                {
                    feature: spaces.MultiBinary(observation_window_size)
                    for feature in features_to_include_in_observation
                }
            ),
        }

        if use_next_subgoal_direction:
            observation_space_dict["next_subgoal_direction"] = spaces.Discrete(
                5, start=-1
            )

        self.observation_space = spaces.Dict(observation_space_dict)

        self.render_mode = render_mode

        self.features_to_include_in_observation = features_to_include_in_observation

        self.use_sliding_observation_window = use_sliding_observation_window
        self.sliding_observation_window_size = sliding_observation_window_size
        self.use_next_subgoal_direction = use_next_subgoal_direction

        self.reward_range = (-np.inf, np.inf)

        self.map_path = map_path
        self.map_plan = None

        self.random_map_width = random_map_width
        self.random_map_height = random_map_height
        self.random_map_percentage_of_connections = random_map_percentage_of_connections
        self.random_map_start_position = random_map_start_position
        self.random_map_goal_position = random_map_goal_position
        self.random_map_minimum_distance_between_start_and_goal = (
            random_map_minimum_distance_between_start_and_goal
        )
        self.random_map_obstacle_probability = random_map_obstacle_probability
        self.random_map_ice_probability_weight = random_map_ice_probability_weight
        self.random_map_broken_road_probability_weight = (
            random_map_broken_road_probability_weight
        )
        self.random_map_sand_probability_weight = random_map_sand_probability_weight
        self.random_map_traffic_light_probability_weight = (
            random_map_traffic_light_probability_weight
        )

        self.sum_subgoals_reward = sum_subgoals_reward  # the sum of all subgoal rewards, the individual subgoal reward is all_subgoals_reward / # of subgoals
        self.final_goal_bonus = final_goal_bonus
        self.crash_penalty = crash_penalty
        self.traffic_light_violation_penalty = traffic_light_violation_penalty
        self.standing_still_penalty = standing_still_penalty
        self.already_visited_position_penalty = already_visited_position_penalty

        self.ice_probability = ice_probability
        self.street_damage_probability = street_damage_probability
        self.sand_probability = sand_probability
        self.traffic_density = traffic_density

        self.traffic_light_phases_duration = traffic_light_phases_duration

        self.ignore_traffic_collisions = ignore_traffic_collisions

        self.window_size = 720
        self.window = None
        self.clock = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict, dict[str, Any]]:
        """Resets the environment. Must be used before starting a episode.

        Returns:
            observation: One element from within the observation space.
            info: Additional information about the state of the environment. Analogous to the info returned by step().
        """
        super().reset(seed=seed)

        # define one random number generator for each random element
        (
            self.map_rng,
            self.car_rng,
            self.ice_rng,
            self.broken_road_rng,
            self.sand_rng,
        ) = self.np_random.spawn(5)

        if (
            not self.map_path == None
        ):  # if a map_path is provided load the map from that path
            if self.map_plan == None:  # only load the file on the first reset
                self.map_plan = json_file_to_map_plan(self.map_path)
            self.map = EpisodeMap(self.map_plan)
        else:
            random_generated_map_plan = generate_map(
                self.random_map_width,
                self.random_map_height,
                self.random_map_percentage_of_connections,
                self.map_rng,
                start_position=self.random_map_start_position,
                goal_position=self.random_map_goal_position,
                minimum_distance_between_start_and_goal=self.random_map_minimum_distance_between_start_and_goal,
                obstacle_probability=self.random_map_obstacle_probability,
                ice_probability_weight=self.random_map_ice_probability_weight,
                broken_road_probability_weight=self.random_map_broken_road_probability_weight,
                sand_probability_weight=self.random_map_sand_probability_weight,
                traffic_light_probability_weight=self.random_map_traffic_light_probability_weight,
            )
            self.map = EpisodeMap(random_generated_map_plan)

        self.individual_subgoal_reward = (
            self.sum_subgoals_reward / self.map.num_subgoals
        )

        self.position = np.array(self.map_rng.choice(self.map.starters))

        self.velocity = np.array([0, 0])

        self.terminated = False
        self.truncated = False
        self.flat_tire = False

        self.positions_path = [list(self.position)]
        self.tile_path = [list(self.position)]
        self.noise_path = []

        self.cars: list[Car] = []
        self._next_car_id = 0

        self._traffic_light_phase_counter = 0

        if self.traffic_density > 0:
            self._create_initial_traffic()

        return (self.get_observation(), self.get_info())

    def _decompose_velocity(
        self, velocity: npt.NDArray | None = None
    ) -> list[npt.NDArray | None]:
        """Decomposes the velocity to all intermediate steps of length 1.

        Args:
            velocity: The velocity to decompose. If None, current velocity is used.

        Returns:
            The list of the individual steps.
        """

        if velocity is None:
            velocity = self.velocity

        dx = velocity[0]
        dy = velocity[1]

        # first compute how the complete velocity change accumulates over time steps
        # trivial case:
        if dx == 0 and dy == 0:
            return []

        res = []
        if dx == 0:
            # each possible y value
            m = np.sign(dy)  # evaluates to 1 or -1, dependent on dy>0 or dy<0
            for i in range(1, np.abs(dy) + 1):
                res.append((0, i * m))
        elif dy == 0:
            m = np.sign(dx)  # evaluates to 1 or -1, dependent of dx>0 or dx<0
            for i in range(1, np.abs(dx) + 1):
                res.append((i * m, 0))
        elif np.abs(dx) >= np.abs(dy):
            m_y = dy / np.abs(dx)
            m_x = np.sign(dx)
            for i in range(1, np.abs(dx) + 1):
                act_x = int(i * m_x)
                act_y = int(_round(i * m_y))
                res.append((act_x, act_y))
        elif np.abs(dx) < np.abs(dy):
            m_x = dx / np.abs(dy)
            m_y = np.sign(dy)
            for i in range(1, np.abs(dy) + 1):
                act_y = int(i * m_y)
                act_x = int(_round(i * m_x))
                res.append((act_x, act_y))

        pre = np.array([0, 0])

        # now deconstruct these changes into single steps by subtracting the velocity already applied
        for i, vel in enumerate(res):
            tmp = np.array(vel)
            vel = tmp - pre
            res[i] = vel
            pre = tmp

        return res

    def generate_frame(
        self,
        hide_positions: bool = False,
        show_observation_window: bool = True,
    ) -> Image:
        """Generates a image showing the current state of the environment.

        Args:
            hide_positions: Default False. Whether or not to hide the positions of the agent.
            show_observation_window: Default True. Whether or not to show the observation window.

        Returns:
            The generated image.
        """

        pic = graphic.create_map(
            self,
            show_path=(not hide_positions),
            show_observation_window=show_observation_window,
        )

        return pic

    def render(self) -> Image | npt.NDArray | None:
        """Potentially returns a rendered representation of the game according to the render mode.

        Returns:
            The rendered representation of the game. If the render mode is "human" or None, None is returned.
        """

        match self.render_mode:
            case None:
                return None
            case "human":
                return None
            case "rgb_array":
                return np.transpose(
                    np.asarray(self.generate_frame().convert("RGB")), axes=(1, 0, 2)
                )
            case "pil_image":
                return self.generate_frame()
            case _:
                raise Exception("the selected render_mode is not supported")

    def _render_frame_for_human(self) -> None:
        """Renders the current state of the environment in a window."""

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (
                    self.window_size * (self.map.tile_width / self.map.tile_height),
                    self.window_size,
                )
            )
            pygame.display.set_caption("PGTG")

        if self.clock is None:
            self.clock = pygame.time.Clock()

        pil_image = self.generate_frame()

        pygame_image = pygame.image.fromstring(
            pil_image.tobytes(), pil_image.size, pil_image.mode  # type: ignore
        ).convert()

        pygame_image = pygame.transform.scale(
            pygame_image,
            (
                self.window_size * (pil_image.size[0] / pil_image.size[1]),
                self.window_size,
            ),
        )

        self.window.blit(pygame_image, pygame_image.get_rect())
        pygame.event.pump()
        pygame.display.update()

        # add a delay to keep the framerate stable
        self.clock.tick(self.metadata["render_fps"])

    def _create_initial_traffic(self) -> None:
        """Creates a number of cars defined by the traffic density."""

        initial_car_positions = self.car_rng.choice(
            self.map.traffic_spawnable_positions,
            size=int(len(self.map.traffic_spawnable_positions) * self.traffic_density),
            replace=False,
        )

        initial_car_positions = [
            tuple(initial_car_position)
            for initial_car_position in initial_car_positions
        ]

        for initial_car_position in initial_car_positions:
            routes = [
                feature.split()[1]
                for feature in self.map.get_features_at(*initial_car_position)
                if "car_lane" in feature and "all" not in feature
            ]

            # Because the features are a set, their order once converted to a list can vary, even with the same seed. Sorting the list makes the environment deterministic again.
            routes.sort()

            assert (
                len(routes) > 0
            ), "a car was spawned on a field where no car lane was found"

            self.cars.append(
                Car(
                    id=self._next_car_id,
                    position=Position(*initial_car_position),
                    route=self.car_rng.choice(routes),
                )
            )

            self._next_car_id += 1

    def _get_next_car_position_and_route(self, car: Car) -> tuple[Position, str] | None:
        """Returns the next position and route of a car or none if there is no possible next position."""

        possible_directions = [
            (car.position.x, car.position.y - 1),
            (car.position.x, car.position.y + 1),
            (car.position.x - 1, car.position.y),
            (car.position.x + 1, car.position.y),
        ]
        possible_types = [
            "up",
            "down",
            "left",
            "right",
        ]

        for possible_position, type in zip(possible_directions, possible_types):
            if not self.map.inside_map(*possible_position):
                continue
                # ignore possible positions outside the map

            square_lanes = [
                feature
                for feature in self.map.get_features_at(*possible_position)
                if "car_lane" in feature
            ]

            lanes_for_all = [lane for lane in square_lanes if "all" in lane]
            if len(lanes_for_all) > 0 and type in lanes_for_all[0]:
                possible_routes = [
                    lane.split()[1] for lane in square_lanes if lane.split()[1] != "all"
                ]

                # Because the features are a set, their order once converted to a list can vary, even with the same seed. Sorting the list makes the environment deterministic again.
                possible_routes.sort()

                return (
                    Position(*possible_position),
                    self.car_rng.choice(possible_routes),
                )

            else:
                for lane in square_lanes:
                    if car.route != None and car.route in lane and type in lane:

                        # check if the possible position is a red traffic light
                        if (
                            self.map.feature_at(*possible_position, "traffic_light")
                            and self.get_traffic_light_phase() == "red"
                        ):
                            # return the car without moving
                            return (car.position, car.route)

                        # check if the possible position is already occupied by another car coming from or driving in the same direction
                        cars_on_next_position = [
                            car
                            for car in self.cars
                            if car.position == Position(*possible_position)
                        ]
                        if any(
                            car_on_next_position.route.split("_")[0]
                            == car.route.split("_")[0]
                            or car_on_next_position.route.split("_")[2]
                            == car.route.split("_")[2]
                            for car_on_next_position in cars_on_next_position
                        ):
                            # return the car without moving
                            return (car.position, car.route)

                        return (Position(*possible_position), car.route)

        return None

    def _spawn_new_car(self) -> Car:
        """Creates a new car and returns it. It still has to be stored.

        Returns:
            The newly created car.
        """

        position = Position(*tuple(self.car_rng.choice(self.map.car_spawners)))
        routes = [
            feature.split()[1]
            for feature in self.map.get_features_at(*position)
            if "car_lane" in feature and "all" not in feature
        ]

        # Because the features are a set, their order once converted to a list can vary, even with the same seed. Sorting the list makes the environment deterministic again.
        routes.sort()

        new_car = Car(
            id=self._next_car_id,
            position=position,
            route=self.car_rng.choice(routes),
        )
        self._next_car_id += 1

        return new_car

    def get_traffic_light_phase(self) -> str:
        """Returns the current traffic light phase."""
        if self._traffic_light_phase_counter < self.traffic_light_phases_duration[0]:
            return "green"
        elif (
            self._traffic_light_phase_counter
            < self.traffic_light_phases_duration[0]
            + self.traffic_light_phases_duration[1]
        ):
            return "yellow"
        else:
            return "red"

    def step(
        self, action: int
    ) -> tuple[dict, SupportsFloat, bool, bool, dict[str, Any]]:
        """Center piece of this class. Performs the given action and returns the results.

        Args:
            action: The action to perform.

        Returns:
            observation: One element from within the observation space.
            reward: The reward for the action.
            terminated flag: Whether or not the episode has ended.
            truncated flag: Whether or not the episode has been truncated.
            info: Additional information about the state of the environment. Analogous to the info returned by reset().
        """

        # check whether the game is already done
        if self.terminated or self.truncated:
            raise RuntimeError("Already done, step has no further effect")

        # increment the traffic light phase counter
        self._traffic_light_phase_counter = (
            self._traffic_light_phase_counter + 1
        ) % sum(self.traffic_light_phases_duration)

        # translate the action id to the acceleration
        acceleration = np.array(ACTIONS_TO_ACCELERATION[action])

        # cars move first
        for car in copy.copy(
            self.cars
        ):  # iterate over copy to be able to remove elements
            next_position_and_route = self._get_next_car_position_and_route(car)
            if next_position_and_route is None:
                self.cars.remove(car)
                self.cars.append(self._spawn_new_car())
            else:
                car.position, car.route = next_position_and_route

        # set start variables
        reward = 0
        current_position: npt.NDArray = copy.copy(self.position)

        # handle the velocity
        self.velocity = self.velocity + acceleration

        decomposed_velocity: list[npt.NDArray | None] = self._decompose_velocity()
        # a "stand still" check to also check the final tile of the step
        decomposed_velocity.append(None)

        # process the single steps
        while decomposed_velocity:
            velocity_part = decomposed_velocity.pop(0)

            current_position_x, current_position_y = current_position

            # case outside map, wall, or traffic
            if (
                not self.map.inside_map(current_position_x, current_position_y)
                or self.map.feature_at(current_position_x, current_position_y, "wall")
                or (
                    not self.ignore_traffic_collisions
                    and tuple(current_position) in [car.position for car in self.cars]
                )
            ):
                reward -= self.crash_penalty
                self.terminated = True
                break

            # case goal
            if self.map.feature_at(
                current_position_x, current_position_y, "final goal"
            ):
                reward += self.individual_subgoal_reward + self.final_goal_bonus
                self.terminated = True
                break

            # case subgoal
            if self.map.feature_at(current_position_x, current_position_y, "subgoal"):
                reward += self.individual_subgoal_reward
                self.map.set_subgoals_to_used(current_position_x, current_position_y)

            # if the last step -> only checking for goal and wall, skip the rest
            if velocity_part is None:
                continue

            # case red light
            next_position = current_position + velocity_part
            if self.map.inside_map(*next_position) and (
                self.map.feature_at(*next_position, "traffic_light")
                and self.get_traffic_light_phase() == "red"
            ):
                reward -= self.traffic_light_violation_penalty

            # case ice
            if (
                self.map.feature_at(current_position_x, current_position_y, "ice")
                and self.ice_rng.random() < self.ice_probability
            ):
                # pick a random action
                ice_action = self.ice_rng.choice(list(range(9)))
                ice_velocity = np.array(ACTIONS_TO_ACCELERATION[ice_action])
                velocity_part = ice_velocity
                self.noise_path.append(list(current_position))

            # case road_break
            if (
                self.map.feature_at(
                    current_position_x, current_position_y, "broken road"
                )
                and self.broken_road_rng.random() < self.street_damage_probability
            ):
                self.flat_tire = True
                self.noise_path.append(list(current_position))

            # case sand
            if (
                self.map.feature_at(current_position_x, current_position_y, "sand")
                and self.sand_rng.random() < self.sand_probability
            ):
                self.noise_path.append(list(current_position))
                current_position += velocity_part
                self.tile_path.append(list(current_position))
                self.velocity = np.array([0, 0])
                break

            current_position += velocity_part
            self.tile_path.append(list(current_position))

        # if there is a flat tire, the velocity is set to zero after each step.
        if self.flat_tire:
            self.velocity = np.array([0, 0])

        # apply penalty for moving to a already visited position
        if (
            self.already_visited_position_penalty != 0
            and not np.array_equal(acceleration, np.array([0, 0]))
            and any(
                [
                    np.array_equal(current_position, position_in_path)
                    for position_in_path in self.positions_path
                ]
            )
        ):
            reward -= self.already_visited_position_penalty

        # keep track of old and new position
        old_position = self.position
        self.position = current_position
        self.positions_path.append(list(self.position))

        if (
            self.standing_still_penalty != 0
            and np.array_equal(acceleration, np.array([0, 0]))
            and np.array_equal(old_position, current_position)
        ):
            reward -= self.standing_still_penalty

        if self.render_mode == "human":
            self._render_frame_for_human()

        # actually, the state is defined through pos and velocity and the distances are only features
        # for reasons of simpler implementation, the features here are returned together with the state
        return (
            self.get_observation(),
            reward,
            self.terminated,
            self.truncated,
            self.get_info(),
        )

    def light_step(
        self, action: int
    ) -> tuple[dict, SupportsFloat, bool, bool, dict[str, Any]]:
        """Copies the environment and executes a single step on it. The original environment remains unchanged.

        Args:
            action: The action to perform.

        Returns:
            observation: One element from within the observation space.
            reward: The reward for the action.
            terminated flag: Whether or not the episode has ended.
            truncated flag: Whether or not the episode has been truncated.
            info: Additional information about the state of the environment. Analogous to the info returned by reset().
        """

        env_copy = copy.deepcopy(self)
        return env_copy.step(action)

    def set_to_state(self, state: dict[str, Any]) -> tuple[dict, dict[str, Any]]:
        """Sets the environment to a given state.

        This function exists for easily making multiple recordings of a agents behavior starting from the same state.
        Setting two environments to the same state and choosing the same actions will NOT result in the same state afterwards, because the random number generators are not synchronized.

        Args:
            state: The state to set the environment to.

        Returns:
            observation: One element from within the observation space.
            info: Additional information about the state of the environment.
        """

        self.position[0] = state["x"]  # self.position[0] is the x coordinate
        self.position[1] = state["y"]  # self.position[1] is the y coordinate
        self.velocity[0] = state["x_velocity"]  # self.velocity[0] is the x velocity
        self.velocity[1] = state["y_velocity"]  # self.velocity[1] is the y velocity
        self.flat_tire = state["flat_tire"]

        self.cars = []
        if state["cars"] != None and len(state["cars"]) > 0:
            for car in state["cars"]:
                self.cars.append(
                    Car(
                        id=car["id"],
                        position=Position(x=car["x"], y=car["y"]),
                        route=car["route"],
                    )
                )
            self._next_car_id = self.cars[-1].id + 1

        return (self.get_observation(), self.get_info())

    def get_observation(self) -> dict[str, Any]:
        """Returns the current observation visible to the agent.

        Returns:
            A element from within the observation space.
        """

        # after the last step the agent could be outside the map
        position_inside_map_x = min(max(0, self.position[0]), self.map.width - 1)
        position_inside_map_y = min(max(0, self.position[1]), self.map.height - 1)

        tile_x = int(position_inside_map_x / TILE_WIDTH)
        tile_y = int(position_inside_map_y / TILE_HEIGHT)

        if not self.use_sliding_observation_window:
            cutout_top_left_x = tile_x * TILE_WIDTH
            cutout_top_left_y = tile_y * TILE_HEIGHT
            cutout_bottom_right_x = tile_x * TILE_WIDTH + TILE_WIDTH - 1
            cutout_bottom_right_y = tile_y * TILE_HEIGHT + TILE_HEIGHT - 1
        else:
            cutout_top_left_x = self.position[0] - self.sliding_observation_window_size
            cutout_top_left_y = self.position[1] - self.sliding_observation_window_size
            cutout_bottom_right_x = (
                self.position[0] + self.sliding_observation_window_size
            )
            cutout_bottom_right_y = (
                self.position[1] + self.sliding_observation_window_size
            )

        map_cutout = self.map.get_map_cutout(
            cutout_top_left_x,
            cutout_top_left_y,
            cutout_bottom_right_x,
            cutout_bottom_right_y,
            None if not self.use_sliding_observation_window else {"wall"},
        )

        map = {}
        # walls has to be handled separately because the feature is named "wall" but the key should be "walls"
        if "walls" in self.features_to_include_in_observation:
            map["walls"] = np.array(self.encode_map_with_hot_one(map_cutout, "wall"))

        # goals has to be handled separately because it consists of the features "subgoal" and "final goal"
        if "goals" in self.features_to_include_in_observation:
            map["goals"] = np.array(
                self.encode_map_with_hot_one(map_cutout, {"subgoal", "final goal"})
            )

        # traffic has to be handled separately because the car positions are not stored in the map
        if "traffic" in self.features_to_include_in_observation:
            # creates a 2d array of the same size as map_cutout filled with 0
            map["traffic"] = np.array(
                [[0] * len(map_cutout[0]) for _ in range(len(map_cutout))]
            )

            for car in self.cars:
                if (
                    cutout_top_left_x <= car.position.x <= cutout_bottom_right_x
                    and cutout_top_left_y <= car.position.y <= cutout_bottom_right_y
                ):
                    map["traffic"][car.position.x - cutout_top_left_x][
                        car.position.y - cutout_top_left_y
                    ] = 1

        # traffic_light has to be handled separately because the current phase has to be considered
        if "traffic_light" in self.features_to_include_in_observation:
            traffic_light_map = np.array(
                self.encode_map_with_hot_one(map_cutout, "traffic_light")
            )
            match self.get_traffic_light_phase():
                case "green":
                    # since all traffic lights are green, the red and yellow maps are empty
                    map["traffic_light_greed"] = traffic_light_map
                    map["traffic_light_yellow"] = np.array(
                        [[0] * len(map_cutout[0]) for _ in range(len(map_cutout))]
                    )
                    map["traffic_light_red"] = np.array(
                        [[0] * len(map_cutout[0]) for _ in range(len(map_cutout))]
                    )
                case "yellow":
                    # since all traffic lights are yellow, the green and red maps are empty
                    map["traffic_light_greed"] = np.array(
                        [[0] * len(map_cutout[0]) for _ in range(len(map_cutout))]
                    )
                    map["traffic_light_yellow"] = traffic_light_map
                    map["traffic_light_red"] = np.array(
                        [[0] * len(map_cutout[0]) for _ in range(len(map_cutout))]
                    )
                case "red":
                    # since all traffic lights are red, the green and yellow maps are empty
                    map["traffic_light_greed"] = np.array(
                        [[0] * len(map_cutout[0]) for _ in range(len(map_cutout))]
                    )
                    map["traffic_light_yellow"] = np.array(
                        [[0] * len(map_cutout[0]) for _ in range(len(map_cutout))]
                    )
                    map["traffic_light_red"] = traffic_light_map

        # the other features can all be treated the same way
        other_features = set(self.features_to_include_in_observation) - set(
            ["walls", "goals", "traffic", "traffic_light"]
        )
        for feature in other_features:
            map[feature] = np.array(self.encode_map_with_hot_one(map_cutout, feature))

        observation: dict[str, Any] = {
            "position": np.array(
                [
                    (
                        (position_inside_map_x - cutout_top_left_x)
                        if not self.use_sliding_observation_window
                        else 0
                    ),
                    (
                        (position_inside_map_y - cutout_top_left_y)
                        if not self.use_sliding_observation_window
                        else 0
                    ),
                ]
            ),
            "velocity": self.velocity,
            "map": map,
        }

        if self.use_next_subgoal_direction:
            next_subgoal_direction = self.map.get_next_subgoal_direction(
                position_inside_map_x, position_inside_map_y
            )
            observation["next_subgoal_direction"] = next_subgoal_direction

        return observation

    def encode_map_with_hot_one(
        self, map_cutout: list[list[set[str]]], features_to_match: str | set[str]
    ) -> list[list[int]]:
        """Transforms a map or map cutout into a hot-one encoding.
        If a square contains one or more of the features to match, the hot-one encoding will have a 1 at that position and otherwise it will have a 0.

        Args:
            map_cutout: The map or map cutout to transform.
            features_to_match: The feature(s) that will result in a 1 in the hot one encoding.

        Returns:
            The hot-one encoding for the specified feature(s).
        """

        assert isinstance(features_to_match, str) or isinstance(
            features_to_match, set
        ), "features_to_match must be a string or a set of strings"

        if isinstance(features_to_match, str):
            features_to_match = {features_to_match}

        # creates a 2d array of the same size as map_cutout filled with 0
        res = [[0] * len(map_cutout[0]) for _ in range(len(map_cutout))]

        for x in range(len(map_cutout)):
            for y in range(len(map_cutout[0])):
                if not map_cutout[x][y].isdisjoint(features_to_match):
                    res[x][y] = 1

        return res

    def get_info(self) -> dict[str, Any]:
        """Returns additional information about the state of the environment."""

        state = {
            "x": self.position[0],
            "y": self.position[1],
            "x_velocity": self.velocity[0],
            "y_velocity": self.velocity[1],
            "flat_tire": self.flat_tire,
            "cars": [],
        }
        for car in self.cars:
            state["cars"].append(
                {
                    "id": car.id,
                    "x": car.position.x,
                    "y": car.position.y,
                    "route": car.route,
                }
            )

        return state

    def applicable_actions(self) -> list[int]:
        """Returns list of applicable actions. For this environment it is always the same unless the episode is over.

        Returns:
            A list of applicable actions.
        """

        if not (self.terminated or self.truncated):
            return list(range(9))
        else:
            return []

    def get_observation_window_coordinates(self) -> tuple[int, int, int, int]:
        """Returns the top left and bottom right corner of the observation window.

        Returns:
            A tuple (top_left_x, top_left_y, bottom_right_x, bottom_right_y).
        """

        if not self.use_sliding_observation_window:
            # after the last step the agent could be outside the map
            position_inside_map_x = min(max(0, self.position[0]), self.map.width - 1)
            position_inside_map_y = min(max(0, self.position[1]), self.map.height - 1)

            tile_x = int(position_inside_map_x / TILE_WIDTH)
            tile_y = int(position_inside_map_y / TILE_HEIGHT)

            return (
                tile_x * TILE_WIDTH,
                tile_y * TILE_HEIGHT,
                tile_x * TILE_WIDTH + TILE_WIDTH - 1,
                tile_y * TILE_HEIGHT + TILE_HEIGHT - 1,
            )
        else:
            return (
                self.position[0] - self.sliding_observation_window_size,
                self.position[1] - self.sliding_observation_window_size,
                self.position[0] + self.sliding_observation_window_size,
                self.position[1] + self.sliding_observation_window_size,
            )
