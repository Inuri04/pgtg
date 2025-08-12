import pathlib
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch

import environment

if __name__ == '__main__':
    script_dir = pathlib.Path(__file__).parent.resolve()

    env = environment.PGTGEnv(
        random_map_width=4,
        random_map_height=4,
        random_map_obstacle_probability=0.2,
        random_map_percentage_of_connections=0.8,
        traffic_density=0.2,
        conservative_driver_percentage=0.15,
        normal_driver_percentage=0.50,
        aggressive_driver_percentage=0.20,
        elderly_driver_percentage=0.10,
        reckless_driver_percentage=0.05,
        sliding_observation_window_size=5,
        max_allowed_deviation=15,
        use_sliding_observation_window=True,
        use_next_subgoal_direction=True,
        final_goal_bonus=200,
        standing_still_penalty=1
    )
    env = TimeLimit(env, max_episode_steps=100)
    env = FlattenObservation(env)

    agent = PPO(
        "MlpPolicy",
        env,
        verbose=1
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    agent.learn(total_timesteps=1_000_000, log_interval=1000, progress_bar=True)

    agent.save(script_dir / "ppo_agent")



