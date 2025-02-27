import copy
import itertools

import numpy as np
import numpy.typing as npt
import pytest

from pgtg.constants import TILE_HEIGHT, TILE_WIDTH
from pgtg.environment import Car, PGTGEnv, Position
from tests.utils import assert_observations_equal


def assert_environments_equal(env_1: PGTGEnv, env_2: PGTGEnv) -> None:
    """Asserts that the two environments are equal.

    Compares the map, the state of the traffic, the position and velocity of the agent, and the visited position among others.
    Does not compare the random number generators.
    """

    assert env_1.map._map == env_2.map._map
    assert env_1.cars == env_2.cars
    assert env_1._next_car_id == env_2._next_car_id
    assert np.array_equal(env_1.position, env_2.position)
    assert np.array_equal(env_1.velocity, env_2.velocity)
    assert env_1.flat_tire == env_2.flat_tire
    assert env_1.positions_path == env_2.positions_path
    assert env_1.tile_path == env_2.tile_path
    assert env_1.noise_path == env_2.noise_path
    assert env_1.terminated == env_2.terminated
    assert env_1.truncated == env_2.truncated


def assert_step_returns_equal(
    env_1_step_returns: tuple, env_2_step_returns: tuple
) -> None:
    """Asserts that the returns of the step function of two environments are equal."""

    observation_1, reward_1, terminated_1, truncated_1, info_1 = env_1_step_returns
    observation_2, reward_2, terminated_2, truncated_2, info_2 = env_2_step_returns

    assert_observations_equal(observation_1, observation_2)
    assert reward_1 == reward_2
    assert terminated_1 == terminated_2
    assert truncated_1 == truncated_2
    assert info_1 == info_2


class TestRandomness:
    def test_different_seed(self, ice_env, ice_env_copy):
        env_0 = ice_env
        env_0.reset(seed=123)

        env_1 = ice_env_copy
        env_1.reset(seed=456)

        assert env_0.map._map != env_1.map._map

    def test_same_seed(self, ice_env, ice_env_copy):
        env_0 = ice_env
        env_0.reset(seed=3)

        env_1 = ice_env_copy
        env_1.reset(seed=3)

        for n in range(3):
            if n != 0:
                env_0.reset()
                env_1.reset()

            for action in [4, 7, 1, 7, 1, 4]:
                assert_environments_equal(env_0, env_1)

                env_0_step_returns = env_0.step(action)
                env_1_step_returns = env_1.step(action)

                assert_step_returns_equal(env_0_step_returns, env_1_step_returns)

    def test_different_seed_then_same(self, ice_env, ice_env_copy):
        env_0 = ice_env
        env_0.reset(seed=123)

        env_1 = ice_env_copy
        env_1.reset(seed=456)

        assert env_0.map._map != env_1.map._map

        env_0.reset(seed=789)
        env_1.reset(seed=789)

        for n in range(3):
            if n != 0:
                env_0.reset()
                env_1.reset()

            for action in [4, 7, 4]:
                assert_environments_equal(env_0, env_1)

                env_0_step_returns = env_0.step(action)
                env_1_step_returns = env_1.step(action)

                assert_step_returns_equal(env_0_step_returns, env_1_step_returns)

    def test_same_seed_many_steps(
        self,
        much_traffic_and_ignore_collisions_env,
        much_traffic_and_ignore_collisions_env_copy,
    ):
        env_0 = much_traffic_and_ignore_collisions_env
        env_0.reset(seed=0)
        env_1 = much_traffic_and_ignore_collisions_env_copy
        env_1.reset(seed=0)

        assert_environments_equal(env_0, env_1)

        for n in range(3):
            if n != 0:
                env_0.reset()
                env_1.reset()

            for _ in range(100):
                assert_environments_equal(env_0, env_1)

                env_0_step_returns = env_0.step(4)
                env_1_step_returns = env_1.step(4)

                assert_step_returns_equal(env_0_step_returns, env_1_step_returns)


class TestObservations:

    @pytest.mark.parametrize("random_map_percentage_of_connections", [0, 0.75, 1])
    def test_velocity_observation(
        self,
        random_map_percentage_of_connections,
    ):
        env = PGTGEnv(
            random_map_width=1,
            random_map_height=1,
            random_map_percentage_of_connections=random_map_percentage_of_connections,
        )
        env.reset(seed=0)

        observation, _, _, _, _ = env.step(4)

        assert np.array_equal(observation["velocity"], env.velocity)

    @pytest.mark.parametrize("random_map_percentage_of_connections", [0, 0.75, 1])
    class TestFixedObservationWindow:
        def test_position_observation(
            self,
            random_map_percentage_of_connections,
        ):
            env = PGTGEnv(
                random_map_width=1,
                random_map_height=1,
                random_map_percentage_of_connections=random_map_percentage_of_connections,
            )
            env.reset(seed=0)

            observation, _, _, _, _ = env.step(4)

            assert np.array_equal(observation["position"], env.position)

        def test_wall_observation(
            self,
            random_map_percentage_of_connections,
        ):
            env = PGTGEnv(
                random_map_width=1,
                random_map_height=1,
                random_map_percentage_of_connections=random_map_percentage_of_connections,
            )
            env.reset(seed=0)

            observation, _, _, _, _ = env.step(4)

            for x in range(TILE_WIDTH):
                for y in range(TILE_HEIGHT):
                    if env.map.feature_at(x, y, "wall"):
                        assert observation["map"]["walls"][x][y] == 1
                    else:
                        assert observation["map"]["walls"][x][y] == 0

        @pytest.mark.parametrize(
            "random_map_obstacle_probability, random_map_ice_probability_weight, random_map_broken_road_probability_weight, random_map_sand_probability_weight",
            [(0, 0, 0, 0), (1, 1, 0, 0), (1, 0, 1, 0), (1, 0, 0, 1), (0.5, 1, 1, 1)],
        )
        def test_obstacles_observation(
            self,
            random_map_percentage_of_connections,
            random_map_obstacle_probability,
            random_map_ice_probability_weight,
            random_map_broken_road_probability_weight,
            random_map_sand_probability_weight,
        ):
            env = PGTGEnv(
                random_map_width=1,
                random_map_height=1,
                random_map_percentage_of_connections=random_map_percentage_of_connections,
                random_map_obstacle_probability=random_map_obstacle_probability,
                random_map_ice_probability_weight=random_map_ice_probability_weight,
                random_map_broken_road_probability_weight=random_map_broken_road_probability_weight,
                random_map_sand_probability_weight=random_map_sand_probability_weight,
            )
            env.reset(seed=0)

            observation, _, _, _, _ = env.step(4)

            for obstacle_name in ["ice", "broken road", "sand"]:
                for x in range(TILE_WIDTH):
                    for y in range(TILE_HEIGHT):
                        if env.map.feature_at(x, y, obstacle_name):
                            assert observation["map"][obstacle_name][x][y] == 1
                        else:
                            assert observation["map"][obstacle_name][x][y] == 0

        @pytest.mark.parametrize("traffic_density", [0, 0.02, 0.1, 1])
        def test_traffic_observation(
            self, random_map_percentage_of_connections, traffic_density
        ):
            env = PGTGEnv(
                random_map_width=1,
                random_map_height=1,
                random_map_percentage_of_connections=random_map_percentage_of_connections,
                traffic_density=traffic_density,
                ignore_traffic_collisions=True,
            )
            env.reset(seed=0)

            observation, _, _, _, _ = env.step(4)

            for x in range(TILE_WIDTH):
                for y in range(TILE_HEIGHT):
                    if (x, y) in [car.position for car in env.cars]:
                        assert observation["map"]["traffic"][x][y] == 1
                    else:
                        assert observation["map"]["traffic"][x][y] == 0

        def test_goal_observation(self, random_map_percentage_of_connections):
            env = PGTGEnv(
                random_map_width=1,
                random_map_height=1,
                random_map_percentage_of_connections=random_map_percentage_of_connections,
            )
            env.reset(seed=0)

            observation, _, _, _, _ = env.step(4)

            for x in range(TILE_WIDTH):
                for y in range(TILE_HEIGHT):
                    if env.map.feature_at(x, y, ["final goal", "subgoal"]):
                        assert observation["map"]["goals"][x][y] == 1
                    else:
                        assert observation["map"]["goals"][x][y] == 0

    @pytest.mark.parametrize("random_map_percentage_of_connections", [0, 0.75, 1])
    class TestSlidingObservationWindow:

        def test_position_observation(
            self,
            random_map_percentage_of_connections,
        ):
            env = PGTGEnv(
                random_map_width=1,
                random_map_height=1,
                random_map_percentage_of_connections=random_map_percentage_of_connections,
                use_sliding_observation_window=True,
            )
            env.reset(seed=0)

            observation, _, _, _, _ = env.step(4)

            assert np.array_equal(observation["position"], np.array([0, 0]))

            env.position = np.array([4, 4])

            observation, _, _, _, _ = env.step(4)

            assert np.array_equal(observation["position"], np.array([0, 0]))

        @pytest.mark.parametrize("map_size", [1, 3, 5])
        @pytest.mark.parametrize(
            "random_map_obstacle_probability, random_map_ice_probability_weight, random_map_broken_road_probability_weight, random_map_sand_probability_weight",
            [(0, 0, 0, 0), (1, 1, 0, 0), (1, 0, 1, 0), (1, 0, 0, 1), (0.5, 1, 1, 1)],
        )
        def test_observation_window_covering_whole_map(
            self,
            random_map_percentage_of_connections,
            map_size,
            random_map_obstacle_probability,
            random_map_ice_probability_weight,
            random_map_broken_road_probability_weight,
            random_map_sand_probability_weight,
        ):
            env = PGTGEnv(
                random_map_width=map_size,
                random_map_height=map_size,
                random_map_percentage_of_connections=random_map_percentage_of_connections,
                use_sliding_observation_window=True,
                sliding_observation_window_size=int((map_size * 9) / 2),
                random_map_obstacle_probability=random_map_obstacle_probability,
                random_map_ice_probability_weight=random_map_ice_probability_weight,
                random_map_broken_road_probability_weight=random_map_broken_road_probability_weight,
                random_map_sand_probability_weight=random_map_sand_probability_weight,
                traffic_density=0.02,
                ignore_traffic_collisions=True,
            )
            env.reset(seed=0)

            env.position = np.array([int((map_size * 9) / 2), int((map_size * 9) / 2)])

            observation, _, _, _, _ = env.step(4)

            for feature_observation in observation["map"].values():
                assert len(feature_observation) == map_size * 9 == env.map.width
                assert len(feature_observation[0]) == map_size * 9 == env.map.height

            for x in range(env.map.width):
                for y in range(env.map.height):

                    # check if all features are observed correctly
                    for feature_observation_name, feature_map_name in [
                        ("walls", "wall"),
                        ("ice", "ice"),
                        ("broken road", "broken road"),
                        ("sand", "sand"),
                        ("goals", ["final goal", "subgoal"]),
                    ]:
                        if env.map.feature_at(x, y, feature_map_name):
                            assert (
                                observation["map"][feature_observation_name][x][y] == 1
                            )
                        else:
                            assert (
                                observation["map"][feature_observation_name][x][y] == 0
                            )

                    # check if traffic is observed correctly
                    if (x, y) in [car.position for car in env.cars]:
                        assert observation["map"]["traffic"][x][y] == 1
                    else:
                        assert observation["map"]["traffic"][x][y] == 0

        @pytest.mark.parametrize(
            "random_map_obstacle_probability, random_map_ice_probability_weight, random_map_broken_road_probability_weight, random_map_sand_probability_weight",
            [(0, 0, 0, 0), (1, 1, 0, 0), (1, 0, 1, 0), (1, 0, 0, 1), (0.5, 1, 1, 1)],
        )
        def test_too_big_observation_window(
            self,
            random_map_percentage_of_connections,
            random_map_obstacle_probability,
            random_map_ice_probability_weight,
            random_map_broken_road_probability_weight,
            random_map_sand_probability_weight,
        ):
            env = PGTGEnv(
                random_map_width=1,
                random_map_height=1,
                random_map_percentage_of_connections=random_map_percentage_of_connections,
                use_sliding_observation_window=True,
                sliding_observation_window_size=13,
                random_map_obstacle_probability=random_map_obstacle_probability,
                random_map_ice_probability_weight=random_map_ice_probability_weight,
                random_map_broken_road_probability_weight=random_map_broken_road_probability_weight,
                random_map_sand_probability_weight=random_map_sand_probability_weight,
                traffic_density=0.02,
                ignore_traffic_collisions=True,
            )
            env.reset(seed=0)

            env.position = np.array([4, 4])

            observation, _, _, _, _ = env.step(4)

            for feature_observation in observation["map"].values():
                assert len(feature_observation) == 27
                assert len(feature_observation[0]) == 27

            for x in range(env.map.width):
                for y in range(env.map.height):

                    # check if all features are observed correctly
                    for feature_observation_name, feature_map_name in [
                        ("walls", "wall"),
                        ("ice", "ice"),
                        ("broken road", "broken road"),
                        ("sand", "sand"),
                        ("goals", ["final goal", "subgoal"]),
                    ]:
                        if env.map.feature_at(x, y, feature_map_name):
                            assert (
                                observation["map"][feature_observation_name][x + 9][
                                    y + 9
                                ]
                                == 1
                            )
                        else:
                            assert (
                                observation["map"][feature_observation_name][x + 9][
                                    y + 9
                                ]
                                == 0
                            )

                    # check if traffic is observed correctly
                    if (x, y) in [car.position for car in env.cars]:
                        assert observation["map"]["traffic"][x + 9][y + 9] == 1
                    else:
                        assert observation["map"]["traffic"][x + 9][y + 9] == 0

            for x in itertools.chain(range(9), range(18, 27)):
                for y in itertools.chain(range(9), range(18, 27)):
                    for feature_observation_name in observation["map"].keys():
                        if feature_observation_name == "walls":
                            assert (
                                observation["map"][feature_observation_name][x][y] == 1
                            )
                        else:
                            assert (
                                observation["map"][feature_observation_name][x][y] == 0
                            )

        @pytest.mark.parametrize("map_length", [1, 3, 5])
        def test_observation_window_covering_the_map_partly(
            self, random_map_percentage_of_connections, map_length
        ):
            env = PGTGEnv(
                random_map_width=map_length,
                random_map_height=1,
                random_map_percentage_of_connections=random_map_percentage_of_connections,
                use_sliding_observation_window=True,
                sliding_observation_window_size=4,
                traffic_density=0.02,
                ignore_traffic_collisions=True,
            )
            env.reset(seed=0)

            env.position = np.array([4, 4])

            observation, _, _, _, _ = env.step(4)

            for feature_observation in observation["map"].values():
                assert len(feature_observation) == 9
                assert len(feature_observation[0]) == 9

            for x in range(9):
                for y in range(9):

                    # check if all features are observed correctly
                    for feature_observation_name, feature_map_name in [
                        ("walls", "wall"),
                        ("ice", "ice"),
                        ("broken road", "broken road"),
                        ("sand", "sand"),
                        ("goals", ["final goal", "subgoal"]),
                    ]:
                        if env.map.feature_at(x, y, feature_map_name):
                            assert (
                                observation["map"][feature_observation_name][x][y] == 1
                            )
                        else:
                            assert (
                                observation["map"][feature_observation_name][x][y] == 0
                            )

                    # check if traffic is observed correctly
                    if (x, y) in [car.position for car in env.cars]:
                        assert observation["map"]["traffic"][x][y] == 1
                    else:
                        assert observation["map"]["traffic"][x][y] == 0

        def test_moving_observation_window_horizontal(
            self, random_map_percentage_of_connections
        ):
            env = PGTGEnv(
                random_map_width=3,
                random_map_height=1,
                random_map_percentage_of_connections=random_map_percentage_of_connections,
                use_sliding_observation_window=True,
                sliding_observation_window_size=5,
                traffic_density=0.02,
                ignore_traffic_collisions=True,
            )
            env.reset(seed=0)

            for x_position in range(9 * 3):
                env.position = np.array([x_position, 4])

                observation, _, _, _, _ = env.step(4)

                for feature_observation in observation["map"].values():
                    assert len(feature_observation) == 11
                    assert len(feature_observation[0]) == 11

                for x in range(11):
                    for y in range(11):

                        x_on_map = env.position[0] + x - 5
                        y_on_map = env.position[1] + y - 5

                        # check if all features except walls are observed correctly (no feature should be observed outside the map)
                        for feature_observation_name, feature_map_name in [
                            ("ice", "ice"),
                            ("broken road", "broken road"),
                            ("sand", "sand"),
                            ("goals", ["final goal", "subgoal"]),
                        ]:

                            if env.map.inside_map(
                                x_on_map, y_on_map
                            ) and env.map.feature_at(
                                x_on_map, y_on_map, feature_map_name
                            ):
                                assert (
                                    observation["map"][feature_observation_name][x][y]
                                    == 1
                                )
                            else:
                                assert (
                                    observation["map"][feature_observation_name][x][y]
                                    == 0
                                )

                        # check if walls are observed correctly (outside the map walls should be observed)
                        if (
                            env.map.inside_map(x_on_map, y_on_map)
                            and env.map.feature_at(x_on_map, y_on_map, "wall")
                        ) or not env.map.inside_map(x_on_map, y_on_map):
                            assert observation["map"]["walls"][x][y] == 1
                        else:
                            assert observation["map"]["walls"][x][y] == 0

                        # check if traffic is observed correctly
                        if (x_on_map, y_on_map) in [car.position for car in env.cars]:
                            assert observation["map"]["traffic"][x][y] == 1
                        else:
                            assert observation["map"]["traffic"][x][y] == 0

        def test_moving_observation_window_vertical(
            self, random_map_percentage_of_connections
        ):
            env = PGTGEnv(
                random_map_width=1,
                random_map_height=3,
                random_map_percentage_of_connections=random_map_percentage_of_connections,
                use_sliding_observation_window=True,
                sliding_observation_window_size=5,
                traffic_density=0.02,
                ignore_traffic_collisions=True,
            )
            env.reset(seed=0)

            for y_position in range(9 * 3 - 4, 4, -1):
                env.position = np.array([4, y_position])

                observation, _, _, _, _ = env.step(4)

                for feature_observation in observation["map"].values():
                    assert len(feature_observation) == 11
                    assert len(feature_observation[0]) == 11

                for x in range(11):
                    for y in range(11):

                        x_on_map = env.position[0] + x - 5
                        y_on_map = env.position[1] + y - 5

                        # check if all features except walls are observed correctly (no feature should be observed outside the map)
                        for feature_observation_name, feature_map_name in [
                            ("ice", "ice"),
                            ("broken road", "broken road"),
                            ("sand", "sand"),
                            ("goals", ["final goal", "subgoal"]),
                        ]:

                            if env.map.inside_map(
                                x_on_map, y_on_map
                            ) and env.map.feature_at(
                                x_on_map, y_on_map, feature_map_name
                            ):
                                assert (
                                    observation["map"][feature_observation_name][x][y]
                                    == 1
                                )
                            else:
                                assert (
                                    observation["map"][feature_observation_name][x][y]
                                    == 0
                                )

                        # check if walls are observed correctly (outside the map walls should be observed)
                        if (
                            env.map.inside_map(x_on_map, y_on_map)
                            and env.map.feature_at(x_on_map, y_on_map, "wall")
                        ) or not env.map.inside_map(x_on_map, y_on_map):
                            assert observation["map"]["walls"][x][y] == 1
                        else:
                            assert observation["map"]["walls"][x][y] == 0

                        # check if traffic is observed correctly
                        if (x_on_map, y_on_map) in [car.position for car in env.cars]:
                            assert observation["map"]["traffic"][x][y] == 1
                        else:
                            assert observation["map"]["traffic"][x][y] == 0

    @pytest.mark.parametrize(
        "tile_coordinates, next_subgoal_direction",
        [
            ((0, 2), 0),
            ((0, 1), 0),
            ((0, 0), 1),
            ((1, 0), 1),
            ((2, 0), 2),
            ((2, 1), 3),
            ((1, 1), 2),
            ((1, 2), 1),
            ((2, 2), 1),
            ((3, 2), 0),
            ((3, 1), 0),
            ((3, 0), 1),
            ((4, 0), 1),
            ((4, 1), -1),
            ((4, 2), -1),
        ],
    )
    def test_next_subgoal_direction(self, tile_coordinates, next_subgoal_direction):
        env = PGTGEnv(
            map_path="tests/test_data/map_with_all_directions",
            use_next_subgoal_direction=True,
        )
        env.reset(seed=0)

        # move the agent to the center of the tile
        env.position = np.array(
            [tile_coordinates[0] * 9 + 4, tile_coordinates[1] * 9 + 4]
        )

        observation, _, _, _, _ = env.step(4)

        assert observation["next_subgoal_direction"] == next_subgoal_direction


class TestTraffic:
    class TestInitialPlacement:
        def test_initial_traffic_placement_fully_filled(self):
            env = PGTGEnv(map_path="tests/test_data/1x1_map", traffic_density=1)
            env.reset(seed=0)

            assert len(env.cars) == 18

            car_positions = [car.position for car in env.cars]

            # check for duplicates
            assert all(car_positions.count(x) == 1 for x in car_positions)

            # top lane
            assert (0, 3) in car_positions
            assert (1, 3) in car_positions
            assert (2, 3) in car_positions
            assert (3, 3) in car_positions
            assert (4, 3) in car_positions
            assert (5, 3) in car_positions
            assert (6, 3) in car_positions
            assert (7, 3) in car_positions
            assert (8, 3) in car_positions

            # bottom lane
            assert (0, 5) in car_positions
            assert (1, 5) in car_positions
            assert (2, 5) in car_positions
            assert (3, 5) in car_positions
            assert (4, 5) in car_positions
            assert (5, 5) in car_positions
            assert (6, 5) in car_positions
            assert (7, 5) in car_positions
            assert (8, 5) in car_positions

            # test whether all cars have routes
            assert all(car.route is not None for car in env.cars)

            car_ids = [car.id for car in env.cars]

            # check ids
            assert car_ids == list(range(18))

        def test_initial_traffic_placement_half_filled(self):
            env = PGTGEnv(map_path="tests/test_data/1x1_map", traffic_density=0.5)
            env.reset(seed=0)

            assert len(env.cars) == 9

            car_positions = [tuple(car.position) for car in env.cars]

            # check for duplicates
            assert all(car_positions.count(x) == 1 for x in car_positions)

            # test whether all cars have routes
            assert all(car.route is not None for car in env.cars)

            car_ids = [car.id for car in env.cars]

            # check ids
            assert car_ids == list(range(9))

        def test_spawn_no_cars_if_traffic_density_is_zero(self):
            env = PGTGEnv(traffic_density=0.0)
            env.reset(seed=0)

            # there should be no car
            assert len(env.cars) == 0

            for _ in range(20):
                env.step(4)  # standing still

                # there should be no car
                assert len(env.cars) == 0

    class TestRespawning:
        def test_single_respawning(self):
            env = PGTGEnv(
                map_path="tests/test_data/1x1_map",
                traffic_density=1,
                ignore_traffic_collisions=True,
            )
            env.reset(seed=1)

            env.step(4)  # standing still

            # there should still be 18 cars
            assert len(env.cars) == 18

        def test_multiple_respawning_fully_filled(self):
            env = PGTGEnv(
                map_path="tests/test_data/1x1_map",
                traffic_density=1,
                ignore_traffic_collisions=True,
            )
            env.reset(seed=0)

            for _ in range(20):
                env.step(4)  # standing still

            # there should still be 18 cars
            assert len(env.cars) == 18

        def test_multiple_respawning_half_filled(self):
            env = PGTGEnv(
                map_path="tests/test_data/1x1_map",
                traffic_density=0.5,
                ignore_traffic_collisions=True,
            )
            env.reset(seed=0)

            for _ in range(20):
                env.step(4)  # standing still

            # there should still be 9 cars
            assert len(env.cars) == 9

        def test_never_respawn_with_density_zero(self):
            env = PGTGEnv(
                map_path="tests/test_data/1x1_map",
                traffic_density=0,
                ignore_traffic_collisions=True,
            )
            env.reset(seed=0)

            for _ in range(20):
                env.step(4)  # standing still

                # there should be no car
                assert len(env.cars) == 0

    class TestOverlappingTraffic:
        def test_no_overlapping_traffic_driving_in_the_same_direction(self):
            env = PGTGEnv(
                map_path="tests/test_data/1x1_crossing_map",
                traffic_density=0,
                ignore_traffic_collisions=True,
            )
            env.reset(seed=0)

            env.cars.append(Car(id=0, position=Position(3, 2), route="north_to_east"))
            env.cars.append(Car(id=1, position=Position(0, 5), route="west_to_east"))

            for _ in range(4):
                env.step(4)

            # there should be no overlapping traffic
            assert len(env.cars) == 2
            assert len([car.position for car in env.cars]) == len(
                set([car.position for car in env.cars])
            )  # no duplicates

        def test_no_overlapping_traffic_coming_from_the_same_direction(self):
            env = PGTGEnv(
                map_path="tests/test_data/1x1_crossing_map",
                traffic_density=0,
                ignore_traffic_collisions=True,
            )
            env.reset(seed=0)

            """two cars are spawned in the same cell, the first one will move, since the next cell is empty
            the second one should not move, since the next cell is now occupied.
            There is no other easy way to place a car where it would move into another car coming from the same direction"""
            env.cars.append(Car(id=0, position=Position(3, 5), route="west_to_east"))
            env.cars.append(Car(id=1, position=Position(3, 5), route="west_to_east"))

            env.step(4)

            # there should be no overlapping traffic
            assert len(env.cars) == 2
            assert len([car.position for car in env.cars]) == len(
                set([car.position for car in env.cars])
            )

        def test_overlapping_traffic_coming_from_and_driving_in_different_directions(
            self,
        ):
            """traffic that shares neither the direction it comes from nor the direction it drives in should overlap"""
            env = PGTGEnv(
                map_path="tests/test_data/1x1_crossing_map",
                traffic_density=0,
                ignore_traffic_collisions=True,
            )
            env.reset(seed=0)

            env.cars.append(Car(id=0, position=Position(3, 2), route="north_to_south"))
            env.cars.append(Car(id=1, position=Position(0, 5), route="west_to_east"))

            for _ in range(3):
                env.step(4)

            # there should be overlapping traffic
            assert len(env.cars) == 2
            assert env.cars[0].position == env.cars[1].position

    def test_ignore_traffic_collisions(self):
        env = PGTGEnv(
            map_path="tests/test_data/1x1_map",
            traffic_density=0,
            ignore_traffic_collisions=True,
        )
        env.reset(seed=0)

        env.position = np.array((1, 5))  # move the agent onto the west to east lane
        env.cars.append(
            Car(id=0, position=Position(0, 5), route="west_to_east")
        )  # add a car that will drive into the agent

        _, _, terminated, _, _ = env.step(4)
        assert not terminated
        assert tuple(env.position) in [car.position for car in env.cars]

        _, _, terminated, _, _ = env.step(7)
        assert not terminated
        _, _, terminated, _, _ = env.step(7)
        assert not terminated

        env.cars.append(
            Car(id=1, position=Position(5, 5), route="west_to_east")
        )  # add a car that will drive into the cell the agent is moving into

        _, _, terminated, _, _ = env.step(4)
        assert not terminated
        assert tuple(env.position) in [car.position for car in env.cars]


class TestReward:
    @pytest.mark.parametrize("sum_subgoals_reward", [100, 444, 4, 0])
    def test_subgoal_reward(self, sum_subgoals_reward):
        env = PGTGEnv(
            "tests/test_data/4x1_map.json",
            sum_subgoals_reward=sum_subgoals_reward,
            final_goal_bonus=0,  # set all other rewards / penalties to zero to prevent interference
            crash_penalty=0,
            standing_still_penalty=0,
            already_visited_position_penalty=0,
        )
        env.reset()

        for n in range(4):
            if n == 0:
                env.step(7)

                for _ in range(6):
                    env.step(4)
            else:
                for _ in range(8):
                    env.step(4)

            _, reward, _, _, _ = env.step(4)
            assert reward == sum_subgoals_reward / 4

    @pytest.mark.parametrize("final_goal_bonus", [100, 10000, 1, 0])
    def test_final_goal_bonus_reward(self, final_goal_bonus):
        env = PGTGEnv(
            "tests/test_data/1x1_map.json",
            final_goal_bonus=final_goal_bonus,
            sum_subgoals_reward=0,  # set all other rewards / penalties to zero to prevent interference
            crash_penalty=0,
            standing_still_penalty=0,
            already_visited_position_penalty=0,
        )
        env.reset()

        env.step(7)

        for _ in range(6):
            env.step(4)

        _, reward, _, _, _ = env.step(4)
        assert reward == final_goal_bonus

    @pytest.mark.parametrize("crash_penalty", [100, 10000, 1, 0])
    def test_crash_penalty(self, crash_penalty):
        env = PGTGEnv(
            "tests/test_data/1x1_map.json",
            crash_penalty=crash_penalty,
            sum_subgoals_reward=0,  # set all other rewards / penalties to zero to prevent interference
            final_goal_bonus=0,
            standing_still_penalty=0,
            already_visited_position_penalty=0,
        )
        env.reset()

        env.position = np.array(
            [0, 4]
        )  # reset position so the agent always crashes at the same step

        env.step(5)

        _, reward, _, _, _ = env.step(4)
        assert reward == -1 * crash_penalty

    @pytest.mark.parametrize("standing_still_penalty", [10, 1000, 1, 0])
    def test_standing_still_penalty_reward(self, standing_still_penalty):
        env = PGTGEnv(
            "tests/test_data/1x1_map.json",
            standing_still_penalty=standing_still_penalty,
            sum_subgoals_reward=0,  # set all other rewards / penalties to zero to prevent interference
            final_goal_bonus=0,
            crash_penalty=0,
            already_visited_position_penalty=0,
        )
        env.reset()

        for _ in range(3):
            # standing still should be penalized
            _, reward, _, _, _ = env.step(4)
            assert reward == -1 * standing_still_penalty

            assert np.array_equal(env.velocity, np.array([0, 0]))  # standing still

        # moving should not be penalized
        _, reward, _, _, _ = env.step(7)
        assert reward == 0

        for _ in range(3):
            # moving should not be penalized, even if there is no acceleration
            _, reward, _, _, _ = env.step(4)
            assert reward == 0

            assert not np.array_equal(
                env.velocity, np.array([0, 0])
            )  # not standing still

        # coming to a halt should not be penalized
        _, reward, _, _, _ = env.step(1)
        assert reward == 0

        for _ in range(3):
            # standing still should be penalized
            _, reward, _, _, _ = env.step(4)
            assert reward == -1 * standing_still_penalty

            assert np.array_equal(env.velocity, np.array([0, 0]))  # standing still

    @pytest.mark.parametrize("already_visited_position_penalty", [10, 1000, 1, 0])
    def test_already_visited_position_penalty_reward(
        self, already_visited_position_penalty
    ):
        env = PGTGEnv(
            "tests/test_data/1x1_map.json",
            already_visited_position_penalty=already_visited_position_penalty,
            sum_subgoals_reward=0,  # set all other rewards / penalties to zero to prevent interference
            final_goal_bonus=0,
            crash_penalty=0,
            standing_still_penalty=0,
        )
        env.reset()

        env.position = np.array(
            [0, 3]
        )  # reset position so the agent always visits the same positions

        # moving should not be penalized
        _, reward, _, _, _ = env.step(7)
        assert reward == 0

        for _ in range(3):
            # moving to a new position should not be penalized
            _, reward, _, _, _ = env.step(4)
            assert reward == 0

            assert not np.array_equal(
                env.velocity, np.array([0, 0])
            )  # not standing still

        # stopping on a already visited position should be penalized
        _, reward, _, _, _ = env.step(1)
        assert reward == -1 * already_visited_position_penalty

        for _ in range(3):
            # standing still on a already visited position should not be penalized
            _, reward, _, _, _ = env.step(4)
            assert reward == 0

            assert np.array_equal(env.velocity, np.array([0, 0]))  # standing still

        # moving to a already visited position should be penalized
        _, reward, _, _, _ = env.step(1)
        assert reward == -1 * already_visited_position_penalty

        # stopping on a already visited position should be penalized
        _, reward, _, _, _ = env.step(7)
        assert reward == -1 * already_visited_position_penalty

        # moving to a already visited position should be penalized
        _, reward, _, _, _ = env.step(7)
        assert reward == -1 * already_visited_position_penalty

        # moving to a new position should not be penalized
        _, reward, _, _, _ = env.step(4)
        assert reward == 0

        # moving to a new position should not be penalized
        _, reward, _, _, _ = env.step(2)
        assert reward == 0

        # moving to a new position should not be penalized
        _, reward, _, _, _ = env.step(4)
        assert reward == 0

        # moving to a new position should not be penalized
        _, reward, _, _, _ = env.step(6)
        assert reward == 0

        # moving to a new position should not be penalized
        _, reward, _, _, _ = env.step(4)
        assert reward == 0

        # moving to a new position should not be penalized
        _, reward, _, _, _ = env.step(0)
        assert reward == 0

        # moving to a new position should not be penalized
        _, reward, _, _, _ = env.step(2)
        assert reward == 0

        # moving to a new position over a already visited position should not be penalized
        _, reward, _, _, _ = env.step(1)
        assert reward == 0

        # moving to a new position should not be penalized
        _, reward, _, _, _ = env.step(4)
        assert reward == 0

        # moving to a new position should not be penalized
        _, reward, _, _, _ = env.step(7)
        assert reward == 0

        # moving to a already visited position should be penalized
        _, reward, _, _, _ = env.step(6)
        assert reward == -1 * already_visited_position_penalty

        # moving to a already visited position should be penalized
        _, reward, _, _, _ = env.step(8)
        assert reward == -1 * already_visited_position_penalty

        # moving to a already visited position over a already visited position should only be penalized once
        _, reward, _, _, _ = env.step(7)
        assert reward == -1 * already_visited_position_penalty


class TestStates:
    @pytest.mark.parametrize(
        "env", ["smallest_simple_env", "simple_env", "obstacle_env"]
    )
    def test_load_state(self, env: PGTGEnv, request):
        env = request.getfixturevalue(env)

        env.step(7)  # right
        env.step(1)  # left
        env.step(7)  # right
        env.step(4)  # nothing
        env.step(4)  # nothing

        before_save_position = env.position
        before_save_velocity = env.velocity
        before_save_flat_tire = env.flat_tire
        before_save_map = copy.deepcopy(
            [
                [env.map.get_features_at(x, y) for y in range(env.map.width)]
                for x in range(env.map.width)
            ]
        )
        before_save_cars = copy.deepcopy(env.cars)

        saved_state = env.get_info()

        env.step(4)  # nothing
        env.step(4)  # nothing

        env.set_to_state(saved_state)

        assert all(before_save_position == env.position)
        assert all(before_save_velocity == env.velocity)
        assert before_save_flat_tire == env.flat_tire
        assert before_save_map == [
            [env.map.get_features_at(x, y) for y in range(env.map.width)]
            for x in range(env.map.width)
        ]
        assert before_save_cars == env.cars


class TestHelperFunctions:
    def test_decompose_velocity_easy(self, smallest_simple_env):
        assert np.array_equal(
            smallest_simple_env._decompose_velocity((3, 0)),
            np.array([(1, 0), (1, 0), (1, 0)]),
        )

        assert np.array_equal(
            smallest_simple_env._decompose_velocity((0, 3)),
            np.array([(0, 1), (0, 1), (0, 1)]),
        )

    def test_decompose_velocity_complex(self, smallest_simple_env):
        assert np.array_equal(
            smallest_simple_env._decompose_velocity((3, -3)),
            np.array([(1, -1), (1, -1), (1, -1)]),
        )

        assert np.array_equal(
            smallest_simple_env._decompose_velocity((3, 1)),
            np.array([(1, 0), (1, 1), (1, 0)]),
        )

        assert np.array_equal(
            smallest_simple_env._decompose_velocity((-1, -3)),
            np.array([(0, -1), (-1, -1), (0, -1)]),
        )


@pytest.fixture
def smallest_simple_env():
    env = PGTGEnv(
        random_map_width=1,
        random_map_height=1,
        random_map_obstacle_probability=0,
    )
    env.reset(seed=0)
    return env


@pytest.fixture
def simple_env():
    env = PGTGEnv(
        random_map_width=3,
        random_map_height=3,
        random_map_obstacle_probability=0,
    )
    env.reset(seed=0)
    return env


@pytest.fixture
def obstacle_env():
    env = PGTGEnv(
        random_map_width=3,
        random_map_height=3,
        traffic_density=0.02,
        ignore_traffic_collisions=True,
    )
    env.reset(seed=0)
    return env


@pytest.fixture
def ice_env():
    env = PGTGEnv(
        random_map_obstacle_probability=1,
        random_map_ice_probability_weight=1000,
        traffic_density=0.02,
    )
    env.reset(seed=0)
    return env


# for some tests two copies of the ice_env are needed. It is not possible to use the same fixture twice in one test, thus this copy.
ice_env_copy = ice_env


@pytest.fixture
def much_traffic_and_ignore_collisions_env():
    env = PGTGEnv(traffic_density=0.1, ignore_traffic_collisions=True)
    env.reset(seed=0)
    return env


# for some tests two copies of the much_traffic_and_ignore_collisions_env are needed. It is not possible to use the same fixture twice in one test, thus this copy.
much_traffic_and_ignore_collisions_env_copy = much_traffic_and_ignore_collisions_env
