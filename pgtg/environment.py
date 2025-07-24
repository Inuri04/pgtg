import copy
import logging
import warnings
from dataclasses import dataclass
from typing import Any, NamedTuple, SupportsFloat, Optional, Dict, List
from enum import Enum
import math

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pygame
from gymnasium import spaces
from PIL.Image import Image

import pgtg.graphic
from constants import (
    ACTIONS_TO_ACCELERATION,
    OBSTACLE_NAMES,
    TILE_HEIGHT,
    TILE_WIDTH,
)
from map import EpisodeMap
from map_generator import generate_map
from map_generator import generate_map_graph
from parser import find_direction, json_file_to_map_plan, parse_map_object, parse_tile_map_to_graph


def _round(x):
    return int(np.floor(x + 0.5))


class Position(NamedTuple):
    x: int
    y: int


class DriverProfile(Enum):
    """Different driver personality types with distinct behaviors"""
    CONSERVATIVE = "conservative"
    NORMAL = "normal" 
    AGGRESSIVE = "aggressive"
    ELDERLY = "elderly"
    RECKLESS = "reckless"


@dataclass
class DriverBehavior:
    """Driver behavior parameters that define how each profile acts"""
    # Traffic light behavior
    yellow_light_stop_probability: float  
    red_light_violation_probability: float
    
    # Following behavior  
    min_following_distance: int 
    patience_level: float  # How long they wait before trying alternate routes
    
    # Speed behavior
    speed_multiplier: float 
    reaction_delay_probability: float  


# Define behavior profiles
DRIVER_BEHAVIORS = {
    DriverProfile.CONSERVATIVE: DriverBehavior(
        yellow_light_stop_probability=0.95,
        red_light_violation_probability=0.01,
        min_following_distance=2,
        patience_level=0.9,
        speed_multiplier=0.8,
        reaction_delay_probability=0.1
    ),
    
    DriverProfile.NORMAL: DriverBehavior(
        yellow_light_stop_probability=0.75,
        red_light_violation_probability=0.05,
        min_following_distance=1,
        patience_level=0.7,
        speed_multiplier=1.0,
        reaction_delay_probability=0.15
    ),
    
    DriverProfile.AGGRESSIVE: DriverBehavior(
        yellow_light_stop_probability=0.3,
        red_light_violation_probability=0.15,
        min_following_distance=0,
        patience_level=0.3,
        speed_multiplier=1.3,
        reaction_delay_probability=0.05
    ),
    
    DriverProfile.ELDERLY: DriverBehavior(
        yellow_light_stop_probability=0.98,
        red_light_violation_probability=0.001,
        min_following_distance=3,
        patience_level=0.95,
        speed_multiplier=0.6,
        reaction_delay_probability=0.3
    ),
    
    DriverProfile.RECKLESS: DriverBehavior(
        yellow_light_stop_probability=0.1,
        red_light_violation_probability=0.3,
        min_following_distance=0,
        patience_level=0.1,
        speed_multiplier=1.5,
        reaction_delay_probability=0.1
    )
}


@dataclass
class Car:
    id: int
    position: Position
    route: str
    driver_profile: DriverProfile
    patience_counter: int = 0
    last_action_delay: int = 0
    stuck_counter: int = 0


@dataclass
class Maneuver:
    """Represents a maneuver rule for agent and traffic directions"""
    agent: str
    traffic: List[str]


@dataclass
class TrafficRule:
    """Represents a traffic rule that can trigger automatic braking"""
    name: str
    tile_type: str
    velocity_range: List[float]  
    min_traffic: int
    min_matching_traffic: int
    maneuvers: List[Maneuver]
    action: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, rule_dict: Dict[str, Any]) -> 'TrafficRule':
        """Create TrafficRule from dictionary"""
        maneuvers = [
            Maneuver(
                agent=m["agent"],
                traffic=m["traffic"]
            ) for m in rule_dict["maneuvers"]
        ]
        
        return cls(
            name=rule_dict["name"],
            tile_type=rule_dict["tile_type"],
            velocity_range=rule_dict["velocity_range"],
            min_traffic=rule_dict["min_traffic"],
            min_matching_traffic=rule_dict["min_matching_traffic"],
            maneuvers=maneuvers,
            action=rule_dict.get("action")
        )


class TrafficRuleEngine:
    """Class for evaluating traffic rules against current conditions."""
    
    def __init__(self):
        self.rules: List[TrafficRule] = []
        self.rule_triggers = []
    
    def add_rule(self, rule: TrafficRule):
        if any(r.name == rule.name for r in self.rules):
            raise ValueError(f"Rule with name {rule.name} already exists.")
        self.rules.append(rule)
    
    def add_rule_from_dict(self, rule_dict: Dict[str, Any]):
        rule = TrafficRule.from_dict(rule_dict)
        self.add_rule(rule)

    def remove_rule(self, name: str) -> bool:
        for i, rule in enumerate(self.rules):
            if rule.name == name:
                del self.rules[i]
                return True
        return False
    
    def get_agent_direction(self, env) -> str:
        """Determine agent direction based on subgoal compass direction"""
        compass_dirs = env._get_subgoal_compass_directions(env.position[0], env.position[1])
        
        # Map to main traffic directions (simplify diagonals to main directions)
        direction_mapping = {
            0: "south_to_north",    
            1: "south_to_north",    
            2: "west_to_east",      
            3: "west_to_east",      
            4: "north_to_south",    
            5: "north_to_south",    
            6: "east_to_west",      
            7: "east_to_west"       
        }

        for i, active in enumerate(compass_dirs):
            if active == 1:
                return direction_mapping[i]
            
        speed = np.linalg.norm(env.velocity)
        return "stationary" if speed < 0.1 else "near_goal"
    
    def get_traffic_in_tile(self, env, tile_x: int, tile_y: int) -> List[Dict[str, Any]]:
        """Get all traffic participants in the specified tile"""
        traffic_in_tile = []
        
        for car in env.cars:
            car_tile_x = int(car.position.x // TILE_WIDTH)
            car_tile_y = int(car.position.y // TILE_HEIGHT)
            
            if car_tile_x == tile_x and car_tile_y == tile_y:
                traffic_in_tile.append({
                    "id": car.id,
                    "route": car.route,
                    "position": (car.position.x, car.position.y),
                    "driver_profile": car.driver_profile.value
                })
                
        return traffic_in_tile
    
    def evaluate_rule(self, env, rule: TrafficRule) -> bool:
        """Evaluate if a rule should trigger given current environment state"""
        
        # Get current tile info
        tile_x = int(env.position[0] // TILE_WIDTH)
        tile_y = int(env.position[1] // TILE_HEIGHT)
        
        # Clamp to map bounds
        tile_x = max(0, min(tile_x, env.map_plan.width - 1))
        tile_y = max(0, min(tile_y, env.map_plan.height - 1))
        
        # Check tile type
        current_tile_exits = env.map_plan.tiles[tile_y][tile_x]["exits"]
        current_tile_type = "".join(str(exit) for exit in current_tile_exits)
        
        if current_tile_type != rule.tile_type:
            return False
            
        # Check velocity range
        agent_speed = np.linalg.norm(env.velocity)
        if not (rule.velocity_range[0] <= agent_speed <= rule.velocity_range[1]):
            return False
            
        # Get traffic in current tile
        traffic_in_tile = self.get_traffic_in_tile(env, tile_x, tile_y)
        
        # Check minimum traffic requirement
        if len(traffic_in_tile) < rule.min_traffic:
            return False
            
        # Get agent direction from subgoal compass direction
        agent_direction = self.get_agent_direction(env)
        
        # Check maneuvers
        matching_traffic_count = 0
        
        for maneuver in rule.maneuvers:
            if maneuver.agent == agent_direction:
                # Count traffic that matches this maneuver
                for traffic in traffic_in_tile:
                    if traffic["route"] in maneuver.traffic:
                        matching_traffic_count += 1
                        
        # Check minimum matching traffic requirement
        if matching_traffic_count < rule.min_matching_traffic:
            return False
            
        return True
    
    def evaluate_all_rules(self, env) -> List[str]:
        """Evaluate all rules and return list of triggered rule names"""
        triggered_rules = []
        
        for rule in self.rules:
            if self.evaluate_rule(env, rule):
                triggered_rules.append(rule.name)
                
        return triggered_rules
        
    def apply_braking(self, env) -> bool:
        """Apply braking if any rules are triggered. Returns True if braking was applied."""
        triggered_rules = self.evaluate_all_rules(env)
        self.rule_triggers = triggered_rules  
        
        if triggered_rules:
            env.velocity = np.array([0, 0])
            return True
            
        return False


class PGTGEnv(gym.Env):
    """Class representing the modular racetrack environment with driver profiles."""

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
        max_allowed_deviation: int = 10,
        conservative_driver_percentage: float = 0.25,
        normal_driver_percentage: float = 0.35,
        aggressive_driver_percentage: float = 0.20,
        elderly_driver_percentage: float = 0.15,
        reckless_driver_percentage: float = 0.05
    ):
        """Initialize PGTG Environment with traffic rules."""
        
        # Initialize traffic rule system first
        self.rule_engine = TrafficRuleEngine()
        self.braking_applied = False

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
            observation_space_dict["next_subgoal_direction"] = spaces.Discrete(9, start=-1)

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

        self.max_allowed_deviation = max_allowed_deviation

        # Driver profile percentages
        self.driver_profile_percentages = {
            DriverProfile.CONSERVATIVE: conservative_driver_percentage,
            DriverProfile.NORMAL: normal_driver_percentage,
            DriverProfile.AGGRESSIVE: aggressive_driver_percentage,
            DriverProfile.ELDERLY: elderly_driver_percentage,
            DriverProfile.RECKLESS: reckless_driver_percentage
        }
        
        total_percentage = sum(self.driver_profile_percentages.values())
        if total_percentage > 0:
            self.driver_profile_percentages = {
                k: v / total_percentage for k, v in self.driver_profile_percentages.items()
            }
        else:
            self.driver_profile_percentages = {profile: 0.0 for profile in DriverProfile}
            self.driver_profile_percentages[DriverProfile.NORMAL] = 1.0

        self.window_size = 720
        self.window = None
        self.clock = None

        # Initialize default traffic rules after all other initialization
        self._add_default_rules()

    def _add_default_rules(self):
        """Add some example traffic rules"""
        
        # Rule for 4-way intersection
        intersection_rule = {
            "name": "four_way_intersection_brake",
            "tile_type": "1111",  # All exits open
            "velocity_range": [0.5, 10.0],  # Only when moving
            "min_traffic": 1,
            "min_matching_traffic": 1,
            "maneuvers": [
                {
                    "agent": "west_to_east",
                    "traffic": ["north_to_south", "south_to_north"]
                },
                {
                    "agent": "east_to_west", 
                    "traffic": ["north_to_south", "south_to_north"]
                },
                {
                    "agent": "north_to_south",
                    "traffic": ["west_to_east", "east_to_west"]
                },
                {
                    "agent": "south_to_north",
                    "traffic": ["west_to_east", "east_to_west"]
                }
            ]
        }
        
        # Rule for T-intersection
        t_intersection_rule = {
            "name": "t_intersection_brake",
            "tile_type": "1110",  # North, East, South exits
            "velocity_range": [0.5, 10.0],
            "min_traffic": 1,
            "min_matching_traffic": 1,
            "maneuvers": [
                {
                    "agent": "south_to_north",
                    "traffic": ["west_to_east", "east_to_west"]
                },
                {
                    "agent": "west_to_east",
                    "traffic": ["south_to_north"]
                }
            ]
        }
        
        self.rule_engine.add_rule_from_dict(intersection_rule)
        self.rule_engine.add_rule_from_dict(t_intersection_rule)
        
    def add_traffic_rule(self, rule_dict: Dict[str, Any]):
        """Add a new traffic rule"""
        self.rule_engine.add_rule_from_dict(rule_dict)
        
    def remove_traffic_rule(self, rule_name: str) -> bool:
        """Remove a traffic rule by name"""
        return self.rule_engine.remove_rule(rule_name)
        
    def get_agent_direction_string(self) -> str:
        """Helper method to get current agent direction as string"""
        return self.rule_engine.get_agent_direction(self)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict, dict[str, Any]]:
        """Resets the environment. Must be used before starting a episode.

        Returns:
            observation: One element from within the observation space.
            info: Additional information about the state of the environment. Analogous to the info returned by step().
        """
        logging.debug("Resetting environment.")
        super().reset(seed=seed)

        (
            self.map_rng,
            self.car_rng,
            self.ice_rng,
            self.broken_road_rng,
            self.sand_rng,
        ) = self.np_random.spawn(5)

        if self.map_path is not None:  
            logging.debug("Loading map from path.")
            if self.map_plan is None:  # only load the file on the first reset
                self.map_plan = json_file_to_map_plan(self.map_path)
            self.map = EpisodeMap(self.map_plan)
            _, _, _, _, self.shortest_path = parse_map_object(self.map_plan)
        else:
            logging.debug("Generating random map.")
            # Generate a new map plan if map_path is None
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
            self.map_plan = random_generated_map_plan  # Ensure self.map_plan is set
            self.map = EpisodeMap(self.map_plan)  # Initialize self.map as EpisodeMap
            _, _, _, _, self.shortest_path = parse_map_object(self.map_plan)

        if self.map_plan is None or self.map is None:
            raise RuntimeError("Failed to initialize map_plan or map during reset.")

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

        logging.debug(f"Initial position: {self.position}, Shortest path: {self.shortest_path}")
        return (self.get_observation(), self.get_info())

    def _select_driver_profile(self) -> DriverProfile:
        """Select a driver profile based on the configured percentages"""
        profiles = list(self.driver_profile_percentages.keys())
        probabilities = list(self.driver_profile_percentages.values())
        return self.car_rng.choice(profiles, p=probabilities)

    def _should_car_stop_at_traffic_light(self, car: Car, light_phase: str) -> bool:
        """Determine if car should stop at traffic light based on driver profile"""
        behavior = DRIVER_BEHAVIORS[car.driver_profile]
        
        if light_phase == "green":
            return False
        elif light_phase == "yellow":
            return self.car_rng.random() < behavior.yellow_light_stop_probability
        elif light_phase == "red":
            # Most drivers stop at red, but some profiles might run it
            return self.car_rng.random() >= behavior.red_light_violation_probability
        
        return True

    def _should_car_move(self, car: Car) -> bool:
        """Determine if car should move this turn based on driver profile"""
        behavior = DRIVER_BEHAVIORS[car.driver_profile]
        
        # Handle reaction delays
        if car.last_action_delay > 0:
            car.last_action_delay -= 1
            return False
        
        # Random reaction delay
        if self.car_rng.random() < behavior.reaction_delay_probability:
            car.last_action_delay = self.car_rng.integers(1, 4) 
            return False
        return self.car_rng.random() < behavior.speed_multiplier

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
            m = np.sign(dy)  
            for i in range(1, np.abs(dy) + 1):
                res.append((0, i * m))
        elif dy == 0:
            m = np.sign(dx) 
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

        for i, vel in enumerate(res):
            tmp = np.array(vel)
            vel = tmp - pre
            res[i] = vel
            pre = tmp

        return res

    def generate_frame(self, hide_positions: bool = False, show_observation_window: bool = True,) -> Image:
        try:
            pic = graphic.create_map(
                self,
                show_path=(not hide_positions),
                show_observation_window=show_observation_window,
            )
            
            if pic is None:
                from PIL import Image as PILImage
                pic = PILImage.new('RGBA', (400, 400), (255, 255, 255, 255))
            
            if pic.mode != 'RGBA':
                pic = pic.convert('RGBA')
                
            return pic
        except Exception as e:
            print(f"Error in generate_frame: {e}")
            from PIL import Image as PILImage
            return PILImage.new('RGBA', (400, 400), (255, 255, 255, 255))

    def render(self) -> Image | npt.NDArray | None:
        """Returns a rendered representation of the game according to the render mode.

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

        num_positions = len(self.map.traffic_spawnable_positions)
        num_cars = int(num_positions * self.traffic_density)

        if num_cars > 0 and num_positions > 0:
            chosen_indices = self.car_rng.choice(
                num_positions,
                size=min(num_cars, num_positions),
                replace=False,
            )
            initial_car_positions = [
                self.map.traffic_spawnable_positions[idx] for idx in chosen_indices
            ]
        else:
            initial_car_positions = []

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

            # Select driver profile based on percentages
            driver_profile = self._select_driver_profile()

            self.cars.append(
                Car(
                    id=self._next_car_id,
                    position=Position(*initial_car_position),
                    route=self.car_rng.choice(routes),
                    driver_profile=driver_profile
                )
            )

            self._next_car_id += 1

    def _get_next_car_position_and_route(self, car: Car) -> tuple[Position, str] | None:
        """Returns the next position and route of a car or none if there is no possible next position."""

        # Check if car should move based on driver profile
        if not self._should_car_move(car):
            car.patience_counter += 1
            return (car.position, car.route)

        behavior = DRIVER_BEHAVIORS[car.driver_profile]

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

                car.patience_counter = 0  
                return (
                    Position(*possible_position),
                    self.car_rng.choice(possible_routes),
                )

            else:
                for lane in square_lanes:
                    if car.route != None and car.route in lane and type in lane:

                        if (
                            self.map.feature_at(*possible_position, "traffic_light")
                        ):
                            traffic_light_phase = self.get_traffic_light_phase()
                            if not self._should_car_stop_at_traffic_light(car, traffic_light_phase):
                                pass
                            elif traffic_light_phase in ["red", "yellow"]:
                                car.patience_counter += 1
                                return (car.position, car.route)

                        cars_on_next_position = [
                            other_car
                            for other_car in self.cars
                            if other_car.position == Position(*possible_position)
                        ]
                        
                        if cars_on_next_position:
                            min_distance = behavior.min_following_distance
                            
                            # More aggressive drivers might follow closer or overtake
                            if min_distance == 0 or car.patience_counter > (behavior.patience_level * 10):
                                # Aggressive driver or impatient driver might still proceed
                                if self.car_rng.random() < (1.0 - behavior.patience_level):
                                    car.patience_counter = 0
                                    return (Position(*possible_position), car.route)
                            
                            # Return without moving
                            car.patience_counter += 1
                            return (car.position, car.route)

                        car.patience_counter = 0 
                        return (Position(*possible_position), car.route)

        car.patience_counter += 1
        return None

    def _spawn_new_car(self) -> Car:
        """Creates a new car and returns it. It still has to be stored.

        Returns:
            The newly created car.
        """

        if len(self.map.car_spawners) > 0:
            spawner_idx = self.car_rng.choice(len(self.map.car_spawners))
            position = Position(*self.map.car_spawners[spawner_idx])
        else:
            position = Position(0, 0)
        routes = [
            feature.split()[1]
            for feature in self.map.get_features_at(*position)
            if "car_lane" in feature and "all" not in feature
        ]

        # Because the features are a set, their order once converted to a list can vary, even with the same seed. Sorting the list makes the environment deterministic again.
        routes.sort()

        # Select driver profile based on percentages
        driver_profile = self._select_driver_profile()

        new_car = Car(
            id=self._next_car_id,
            position=position,
            route=self.car_rng.choice(routes),
            driver_profile=driver_profile
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

    def get_driver_profile_stats(self) -> dict:
        """Get statistics about current driver profile distribution"""
        profile_counts = {profile.value: 0 for profile in DriverProfile}
        
        for car in self.cars:
            profile_counts[car.driver_profile.value] += 1
        
        total_cars = len(self.cars)
        if total_cars > 0:
            profile_percentages = {k: (v/total_cars)*100 for k, v in profile_counts.items()}
        else:
            profile_percentages = {k: 0 for k in profile_counts.keys()}
        
        return {
            'counts': profile_counts,
            'percentages': profile_percentages,
            'total_cars': total_cars,
            'configured_percentages': {k.value: v*100 for k, v in self.driver_profile_percentages.items()}
        }

    def _get_subgoal_compass_directions(self, x: float, y: float) -> list[int]:
        """Calculate direction indicators to the nearest subgoal.
        
        Returns:
            List with 8 elements [N, NE, E, SE, S, SW, W, NW], where 1 indicates active direction.
        """
        # Find the nearest subgoal or final goal
        nearest_subgoal = None
        min_distance = float('inf')
        
        for tx in range(self.map.width):
            for ty in range(self.map.height):
                if self.map.feature_at(tx, ty, "subgoal") or self.map.feature_at(tx, ty, "final goal"):
                    distance = abs(tx - x) + abs(ty - y)  
                    if distance < min_distance:
                        min_distance = distance
                        nearest_subgoal = (tx, ty)
        
        if nearest_subgoal is None:
            return [0, 0, 0, 0, 0, 0, 0, 0]  

        dx = nearest_subgoal[0] - x
        dy = nearest_subgoal[1] - y
        
        if abs(dx) <= self.sliding_observation_window_size and abs(dy) <= self.sliding_observation_window_size:
            return [0, 0, 0, 0, 0, 0, 0, 0]  
    
        directions = [0, 0, 0, 0, 0, 0, 0, 0]
        
        # Calculate angle to determine direction
        # +x is right/east, -x is left/west
        # +y is down/south, -y is up/north
        angle = math.atan2(dy, dx) 
        
        PI_8 = math.pi / 8
        
        if -PI_8 <= angle < PI_8:
            directions[2] = 1  # East
        elif PI_8 <= angle < 3*PI_8:
            directions[3] = 1  # Southeast
        elif 3*PI_8 <= angle < 5*PI_8:
            directions[4] = 1  # South
        elif 5*PI_8 <= angle < 7*PI_8:
            directions[5] = 1  # Southwest
        elif angle >= 7*PI_8 or angle < -7*PI_8:
            directions[6] = 1  # West
        elif -7*PI_8 <= angle < -5*PI_8:
            directions[7] = 1  # Northwest
        elif -5*PI_8 <= angle < -3*PI_8:
            directions[0] = 1  # North
        elif -3*PI_8 <= angle < -PI_8:
            directions[1] = 1  # Northeast
        
        return directions

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

        # Store original velocity before potential rule-based modifications
        original_velocity = self.velocity.copy()

        # Apply traffic rules (this might modify velocity)
        self.braking_applied = self.rule_engine.apply_braking(self)

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

        self._check_deviation_and_recalculate_path()

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

        self.position[0] = state["x"]  
        self.position[1] = state["y"]  
        self.velocity[0] = state["x_velocity"]  
        self.velocity[1] = state["y_velocity"]  
        self.flat_tire = state["flat_tire"]

        self.cars = []
        if state["cars"] != None and len(state["cars"]) > 0:
            for car_data in state["cars"]:
                # Handle backward compatibility - if driver_profile is missing, default to NORMAL
                driver_profile = DriverProfile.NORMAL
                if "driver_profile" in car_data:
                    try:
                        driver_profile = DriverProfile(car_data["driver_profile"])
                    except ValueError:
                        driver_profile = DriverProfile.NORMAL
                
                self.cars.append(
                    Car(
                        id=car_data["id"],
                        position=Position(x=car_data["x"], y=car_data["y"]),
                        route=car_data["route"],
                        driver_profile=driver_profile
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

        next_subgoal_direction = -1
        if self.use_next_subgoal_direction:
            next_subgoal_direction = self.map.get_next_subgoal_direction(
                position_inside_map_x, position_inside_map_y
            )

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

        if "walls" in self.features_to_include_in_observation:
            map["walls"] = np.array(self.encode_map_with_hot_one(map_cutout, "wall"))

        if "goals" in self.features_to_include_in_observation:
            map["goals"] = np.array(
                self.encode_map_with_hot_one(map_cutout, {"subgoal", "final goal"})
            )

        if "traffic" in self.features_to_include_in_observation:
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

        if "traffic_light" in self.features_to_include_in_observation:
            traffic_light_map = np.array(
                self.encode_map_with_hot_one(map_cutout, "traffic_light")
            )
            match self.get_traffic_light_phase():
                case "green":
                    map["traffic_light_green"] = traffic_light_map  
                    map["traffic_light_yellow"] = np.array(
                        [[0] * len(map_cutout[0]) for _ in range(len(map_cutout))]
                    )
                    map["traffic_light_red"] = np.array(
                        [[0] * len(map_cutout[0]) for _ in range(len(map_cutout))]
                    )
                case "yellow":
                    map["traffic_light_green"] = np.array(  
                        [[0] * len(map_cutout[0]) for _ in range(len(map_cutout))]
                    )
                    map["traffic_light_yellow"] = traffic_light_map
                    map["traffic_light_red"] = np.array(
                        [[0] * len(map_cutout[0]) for _ in range(len(map_cutout))]
                    )
                case "red":
                    map["traffic_light_green"] = np.array(  
                        [[0] * len(map_cutout[0]) for _ in range(len(map_cutout))]
                    )
                    map["traffic_light_yellow"] = np.array(
                        [[0] * len(map_cutout[0]) for _ in range(len(map_cutout))]
                    )
                    map["traffic_light_red"] = traffic_light_map

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
                        else self.sliding_observation_window_size
                    ),
                    (
                        (position_inside_map_y - cutout_top_left_y)
                        if not self.use_sliding_observation_window
                        else self.sliding_observation_window_size
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
            if next_subgoal_direction == -1 or self.use_sliding_observation_window:
                nearest_subgoal = None
                min_distance = float('inf')
                
                for tx in range(self.map.width):
                    for ty in range(self.map.height):
                        if self.map.feature_at(tx, ty, "subgoal") or self.map.feature_at(tx, ty, "final goal"):
                            distance = abs(tx - position_inside_map_x) + abs(ty - position_inside_map_y)
                            if distance < min_distance:
                                min_distance = distance
                                nearest_subgoal = (tx, ty)
                
                if nearest_subgoal:
                    dx = nearest_subgoal[0] - position_inside_map_x
                    dy = nearest_subgoal[1] - position_inside_map_y
                    
                    angle = math.atan2(-dy, dx)  

                    direction_index = int(((angle + math.pi) / (math.pi / 4)) % 8)
                    
                    # Direction mapping:
                    direction_remap = {
                        0: 2,  # East
                        1: 1,  # Northeast
                        2: 0,  # North
                        3: 7,  # Northwest
                        4: 6,  # West
                        5: 5,  # Southwest
                        6: 4,  # South
                        7: 3   # Southeast
                    }
                    
                    next_subgoal_direction = direction_remap[direction_index]
            
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

        res = [[0] * len(map_cutout[0]) for _ in range(len(map_cutout))]

        for x in range(len(map_cutout)):
            for y in range(len(map_cutout[0])):
                if not map_cutout[x][y].isdisjoint(features_to_match):
                    res[x][y] = 1

        return res

    def get_info(self) -> dict[str, Any]:
        """Returns additional information about the state of the environment."""

        # Calculate tile information
        tile_x = int(self.position[0] // TILE_WIDTH)
        tile_y = int(self.position[1] // TILE_HEIGHT)

        tile_x = max(0, min(tile_x, self.map_plan.width - 1))
        tile_y = max(0, min(tile_y, self.map_plan.height - 1))

        tile_exits = self.map_plan.tiles[tile_y][tile_x]["exits"]
        current_tile_type = "".join(str(exit) for exit in tile_exits)

        state = {
            "x": self.position[0],
            "y": self.position[1],
            "x_velocity": self.velocity[0],
            "y_velocity": self.velocity[1],
            "flat_tire": self.flat_tire,
            "current_tile_type": current_tile_type,  # Added as requested
            "cars": [],
            "driver_profile_stats": self.get_driver_profile_stats(),
            "traffic_rules": {
                'active_rules': [rule.name for rule in self.rule_engine.rules],
                'triggered_rules': getattr(self.rule_engine, 'rule_triggers', []),
                'braking_applied': getattr(self, 'braking_applied', False),
                'agent_direction': self.get_agent_direction_string()
            }
        }
        
        for car in self.cars:
            state["cars"].append({
                "id": car.id,
                "x": car.position.x,
                "y": car.position.y,
                "route": car.route,
                "driver_profile": car.driver_profile.value,
                "patience_counter": car.patience_counter
            })

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

    def distance_from_path(self, position, path):
        """Calculates the minimum Manhattan distance from the current position to the planned path."""
        logging.debug(f"Calculating distance from position: {position} to path.")
        
        # Convert position to tile coordinates for comparison
        pos_tile_x = int(position[0] // TILE_WIDTH)
        pos_tile_y = int(position[1] // TILE_HEIGHT)
        
        path_points = path.keys() if isinstance(path, dict) else path
        
        if not path_points:
            logging.warning("No path points available for distance calculation")
            return float('inf')
        
        min_distance = float('inf')
        for path_point in path_points:
            tile_distance = abs(pos_tile_x - path_point[0]) + abs(pos_tile_y - path_point[1])
            min_distance = min(min_distance, tile_distance)
        
        logging.debug(f"Calculated distance: {min_distance}")
        return min_distance

    def _check_deviation_and_recalculate_path(self):
        """Checks if the car has deviated significantly from the path and recalculates it if necessary."""
        
        agent_position = tuple(map(int, self.position))
        logging.debug(f"Agent position: {agent_position}")

        if not self.shortest_path:
            logging.debug("No shortest path available.")
            return
        
        if self.max_allowed_deviation is None:
            return

        deviation = self.distance_from_path(agent_position, self.shortest_path)
        logging.debug(f"Deviation: {deviation}, Max Allowed Deviation: {self.max_allowed_deviation}")

        if deviation > self.max_allowed_deviation:
            logging.info(f"Deviation ({deviation}) exceeded max allowed ({self.max_allowed_deviation}). Recalculating path.")
            self._recalculate_path(agent_position)
        else:
            logging.debug("Deviation within allowed range. No recalculation needed.")

    def _recalculate_path(self, current_position: tuple[int, int]):
        """Recalculates the shortest path from the current position to the goal."""
        logging.info(f"Recalculating path from position: {current_position}")
        
        # Convert current position to tile coordinates
        current_tile_x = int(current_position[0] // TILE_WIDTH)
        current_tile_y = int(current_position[1] // TILE_HEIGHT)
        
        current_tile_x = max(0, min(current_tile_x, self.map.tile_width - 1))
        current_tile_y = max(0, min(current_tile_y, self.map.tile_height - 1))
        
        logging.debug(f"Current tile position: ({current_tile_x}, {current_tile_y})")
        
        # Get goal tile coordinates
        goal_tile_x = self.map_plan.goal[0]
        goal_tile_y = self.map_plan.goal[1]
        
        try:
            # Create graph from current map state
            graph = parse_tile_map_to_graph(self.map_plan)
            
            # Find shortest path from current tile to goal tile
            shortest_path_result = graph.shortest_path(
                (current_tile_x, current_tile_y), 
                (goal_tile_x, goal_tile_y)
            )
            
            if shortest_path_result is None:
                logging.warning("No path found from current position to goal!")
                return
            
            new_path_tiles = shortest_path_result[1]
            
            new_shortest_path = {}
            for i in range(len(new_path_tiles) - 1):
                current_tile = new_path_tiles[i]
                next_tile = new_path_tiles[i + 1]
                direction = find_direction(current_tile, next_tile)
                new_shortest_path[current_tile] = direction
            
            new_shortest_path[new_path_tiles[-1]] = self.map_plan.goal[2]
            
            self.shortest_path = new_shortest_path
            logging.info(f"New shortest path calculated with {len(new_shortest_path)} waypoints")
            
        except Exception as e:
            logging.error(f"Error recalculating path: {e}")
            logging.info("Keeping original path as fallback")
    
    def get_current_tile_position(self):
        """Returns the current tile coordinates of the agent."""
        tile_x = int(self.position[0] // TILE_WIDTH)
        tile_y = int(self.position[1] // TILE_HEIGHT)
        return (tile_x, tile_y)

    def is_on_planned_path(self):
        """Returns True if the agent is currently on a tile that's part of the planned path."""
        current_tile = self.get_current_tile_position()
        return current_tile in self.shortest_path