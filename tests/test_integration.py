import gymnasium
import toml
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

import pgtg
from pgtg.environment import PGTGEnv
from tests.test_data.reproducibility_data import COMPLICATED_ENVIRONMENT
from tests.utils import assert_observations_equal


def test_simple_random_generated_map_environment():
    env = PGTGEnv(random_map_width=3, random_map_height=3)
    env.reset()

    env.render()
    env.get_info()

    env.step(4)

    env.render()
    env.get_info()


def test_complex_random_generated_map_environment():
    env = PGTGEnv(
        random_map_width=9,
        random_map_height=9,
        random_map_percentage_of_connections=0.9,
        random_map_obstacle_probability=0.8,
        random_map_broken_road_probability_weight=2,
        random_map_ice_probability_weight=4,
        random_map_sand_probability_weight=8,
        render_mode="pil_image",
        final_goal_bonus=100,
        standing_still_penalty=5,
        ice_probability=0.5,
        street_damage_probability=0.2,
        traffic_density=0.02,
        ignore_traffic_collisions=True,
    )
    env.reset()

    env.render()
    env.get_info()

    env.step(4)

    env.render()
    env.get_info()


def test_loaded_map_environment():
    env = PGTGEnv(map_path="tests/test_data/1x1_map")
    env.reset()

    env.render()
    env.get_info()

    env.step(4)

    env.render()
    env.get_info()


def test_observation_consistency():
    env = gymnasium.make("pgtg-v4", **COMPLICATED_ENVIRONMENT["environment_arguments"])

    # there should be the same number of actions, rewards, terminated, and truncated and one more observation because of th initial state
    assert (
        (len(COMPLICATED_ENVIRONMENT["action_list"]))
        == len(COMPLICATED_ENVIRONMENT["observation_list"]) - 1
        == len(COMPLICATED_ENVIRONMENT["reward_list"])
        == len(COMPLICATED_ENVIRONMENT["terminated_list"])
        == len(COMPLICATED_ENVIRONMENT["truncated_list"])
    )

    observation, _ = env.reset(seed=COMPLICATED_ENVIRONMENT["seed"])
    assert_observations_equal(
        observation, COMPLICATED_ENVIRONMENT["observation_list"][0]
    )

    for n in range(len(COMPLICATED_ENVIRONMENT["action_list"])):
        observation, reward, terminated, truncated, _ = env.step(
            COMPLICATED_ENVIRONMENT["action_list"][n]
        )

        assert_observations_equal(
            observation, COMPLICATED_ENVIRONMENT["observation_list"][n + 1]
        )
        assert reward == COMPLICATED_ENVIRONMENT["reward_list"][n]
        assert terminated == COMPLICATED_ENVIRONMENT["terminated_list"][n]
        assert truncated == COMPLICATED_ENVIRONMENT["truncated_list"][n]


class TestGymnasiumAPIConformity:
    def test_fixed_observation_window(self, recwarn):
        env = gymnasium.make("pgtg-v4")

        check_env(env.unwrapped)

        # check_env does not return anything but instead issues warnings
        # thus it is necessary tu use the recwarn fixture that records all warnings
        assert len(recwarn) == 0, "at least one warning was issued: " + str(
            [warning.message for warning in recwarn.list]
        )

    def test_sliding_observation_window(self, recwarn):
        env = gymnasium.make(
            "pgtg-v4",
            use_sliding_observation_window=True,
            sliding_observation_window_size=2,
        )

        check_env(env.unwrapped)

        # check_env does not return anything but instead issues warnings
        # thus it is necessary tu use the recwarn fixture that records all warnings
        assert len(recwarn) == 0, "at least one warning was issued: " + str(
            [warning.message for warning in recwarn.list]
        )

    def test_next_subgoal_direction(self, recwarn):
        env = gymnasium.make(
            "pgtg-v4",
            use_next_subgoal_direction=True,
        )

        check_env(env.unwrapped)

        # check_env does not return anything but instead issues warnings
        # thus it is necessary tu use the recwarn fixture that records all warnings
        assert len(recwarn) == 0, "at least one warning was issued: " + str(
            [warning.message for warning in recwarn.list]
        )


def test_consistent_version_numbers():
    pyproject = toml.loads(open("pyproject.toml").read())

    assert pgtg.__version__ == pyproject["tool"]["poetry"]["version"]
