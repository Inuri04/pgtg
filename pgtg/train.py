#!/usr/bin/env python3
"""
Simple DQN Baseline for PGTG Environment
Exploration Experiments - Baseline RL
"""

import os
import logging
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import FlattenObservation  # --- CHANGE 1: Import the wrapper ---
import torch

# Import the PGTG environment
from environment import PGTGEnv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EpisodeRewardCallback(gym.Wrapper):
    """Simple wrapper to track episode statistics"""
    def __init__(self, env):
        super().__init__(env)
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_episode_reward += reward
        self.current_episode_length += 1

        if terminated or truncated:
            self.episode_count += 1
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)

            # Log every 100 episodes
            if self.episode_count % 100 == 0:
                recent_rewards = self.episode_rewards[-100:]
                recent_lengths = self.episode_lengths[-100:]
                if recent_rewards:
                    success_rate = sum(1 for r in recent_rewards if r > 0) / len(recent_rewards)
                    avg_reward = np.mean(recent_rewards)
                    avg_length = np.mean(recent_lengths)
                    logger.info(f"Episode {self.episode_count}: "
                                f"Success Rate = {success_rate:.3f}, "
                                f"Avg Reward = {avg_reward:.3f}, "
                                f"Avg Length = {avg_length:.1f}")

            self.current_episode_reward = 0
            self.current_episode_length = 0

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.current_episode_reward = 0
        self.current_episode_length = 0
        return self.env.reset(**kwargs)

def create_env(map_size=2, **env_kwargs):
    """Create PGTG environment with specified settings"""
    env = PGTGEnv(
        random_map_width=map_size,
        random_map_height=map_size,
        **env_kwargs
    )
    env = EpisodeRewardCallback(env)
    # --- CHANGE 2: Wrap the environment to flatten the observation ---
    env = FlattenObservation(env)
    return env

def main():
    # Environment configuration - your specified settings
    MAP_SIZE = 2  # Start with 2x2 maps

    env_kwargs = {
        # Traffic settings
        'traffic_density': 0.2,
        'random_map_percentage_of_connections': 0.8,

        # Reward settings
        'sum_subgoals_reward': 0,   # No subgoal rewards
        'final_goal_bonus': 1,      # Goal reward: +1
        'crash_penalty': 1,         # Crash penalty: -1
        'standing_still_penalty': 0.1,  # Standing still: -1/10

        # Observation settings
        'use_sliding_observation_window': True,
        'sliding_observation_window_size': 4,
        'use_next_subgoal_direction': True,

        # Features to include
        'features_to_include_in_observation': [
            'walls', 'goals', 'ice', 'broken road', 'sand', 'traffic',
            'traffic_light_green', 'traffic_light_yellow', 'traffic_light_red'
        ],

        # Default obstacle settings
        'random_map_obstacle_probability': 0.0,
        'ice_probability': 0.1,
        'street_damage_probability': 0.1,
        'sand_probability': 0.2,
    }

    # Create environments
    logger.info(f"Creating environment with {MAP_SIZE}x{MAP_SIZE} maps...")

    train_env = make_vec_env(
        lambda: create_env(map_size=MAP_SIZE, **env_kwargs),
        n_envs=1,
        vec_env_cls=DummyVecEnv
    )

    eval_env = make_vec_env(
        lambda: create_env(map_size=MAP_SIZE, **env_kwargs),
        n_envs=1,
        vec_env_cls=DummyVecEnv
    )

    # Set up logging
    log_dir = "/netscratch/$USER/logs/dqn_baseline" if "/netscratch" in os.environ.get('TMPDIR', '') else "./logs/dqn_baseline"
    os.makedirs(log_dir, exist_ok=True)

    # DQN Configuration - simple baseline
    model = DQN(
        # --- CHANGE 3: Change policy to MlpPolicy for the flattened observation ---
        'MlpPolicy',
        train_env,
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=5000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=5000,
        exploration_fraction=0.5,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        max_grad_norm=10,
        policy_kwargs={'net_arch': [256, 256]},
        tensorboard_log=os.path.join(log_dir, "tensorboard"),
        verbose=1,
        device='auto'
    )

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval_logs"),
        eval_freq=5000,
        deterministic=True,
        n_eval_episodes=10,
        verbose=1
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="dqn_model"
    )

    callbacks = [eval_callback, checkpoint_callback]

    # Training parameters
    total_timesteps = 200000  # 200k steps for baseline

    logger.info("=" * 60)
    logger.info("DQN Baseline Training - Exploration Experiment")
    logger.info("=" * 60)
    logger.info(f"Map size: {MAP_SIZE}x{MAP_SIZE}")
    logger.info(f"Traffic density: {env_kwargs['traffic_density']}")
    logger.info(f"Connections: {env_kwargs['random_map_percentage_of_connections']}")
    logger.info(f"Standing still penalty: -{env_kwargs['standing_still_penalty']}")
    logger.info(f"Total timesteps: {total_timesteps}")
    logger.info(f"Log directory: {log_dir}")
    logger.info("=" * 60)

    try:
        # Train the model
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=100,
            progress_bar=True
        )

        # Save final model
        final_model_path = os.path.join(log_dir, "final_model")
        model.save(final_model_path)
        logger.info(f"Final model saved to {final_model_path}")

        # Final evaluation
        logger.info("Running final evaluation...")
        episode_rewards = []

        for episode in range(20):
            episode_reward = 0
            done = False
            obs = eval_env.reset()

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                episode_reward += reward[0]

                if done[0]:
                    episode_rewards.append(episode_reward)
                    break
        
        if episode_rewards:
            success_rate = sum(1 for r in episode_rewards if r > 0) / len(episode_rewards)
            avg_reward = np.mean(episode_rewards)

            logger.info("=" * 60)
            logger.info("FINAL RESULTS")
            logger.info("=" * 60)
            logger.info(f"Success rate: {success_rate:.3f}")
            logger.info(f"Average reward: {avg_reward:.3f}")
            logger.info(f"Reward range: [{min(episode_rewards):.2f}, {max(episode_rewards):.2f}]")
            logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        train_env.close()
        eval_env.close()

if __name__ == "__main__":
    main()
