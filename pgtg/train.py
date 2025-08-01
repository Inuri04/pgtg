#!/usr/bin/env python3

import os
import pathlib
import time
from datetime import datetime
from typing import Dict, Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FlattenObservation, TimeLimit
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from environment import PGTGEnv


class PGTGTrainingConfig:
    """Configuration class for PGTG training parameters."""
    
    def __init__(self):
        # Environment parameters
        self.max_episode_steps = 200
        
        # Map generation parameters
        self.random_map_width = 6
        self.random_map_height = 6
        self.random_map_percentage_of_connections = 0.7
        self.random_map_obstacle_probability = 0.2
        
        # Traffic parameters
        self.traffic_density = 0.3
        self.ignore_traffic_collisions = False
        
        # Driver profile distribution
        self.conservative_driver_percentage = 0.20
        self.normal_driver_percentage = 0.40
        self.aggressive_driver_percentage = 0.25
        self.elderly_driver_percentage = 0.10
        self.reckless_driver_percentage = 0.05
        
        # Reward parameters
        self.sum_subgoals_reward = 100
        self.final_goal_bonus = 200
        self.crash_penalty = 150
        self.traffic_light_violation_penalty = 75
        self.standing_still_penalty = 1
        self.already_visited_position_penalty = 10
        
        # Training parameters
        self.total_timesteps = 500000
        self.algorithm = "DQN"  # Options: "DQN", "PPO", "A2C"
        self.learning_rate = 0.0005
        self.batch_size = 64
        self.buffer_size = 50000
        self.exploration_fraction = 0.3
        self.exploration_final_eps = 0.02
        
        # Evaluation parameters
        self.eval_freq = 10000
        self.n_eval_episodes = 20
        self.eval_deterministic = True
        
        # Logging parameters
        self.log_interval = 1000
        self.save_freq = 50000
        self.verbose = 1


def create_pgtg_env(config: PGTGTrainingConfig, seed: Optional[int] = None) -> gym.Env:
    """Create a PGTG environment with specified configuration."""
    
    env_kwargs = {
        'random_map_width': config.random_map_width,
        'random_map_height': config.random_map_height,
        'random_map_percentage_of_connections': config.random_map_percentage_of_connections,
        'random_map_obstacle_probability': config.random_map_obstacle_probability,
        'traffic_density': config.traffic_density,
        'ignore_traffic_collisions': config.ignore_traffic_collisions,
        'conservative_driver_percentage': config.conservative_driver_percentage,
        'normal_driver_percentage': config.normal_driver_percentage,
        'aggressive_driver_percentage': config.aggressive_driver_percentage,
        'elderly_driver_percentage': config.elderly_driver_percentage,
        'reckless_driver_percentage': config.reckless_driver_percentage,
        'sum_subgoals_reward': config.sum_subgoals_reward,
        'final_goal_bonus': config.final_goal_bonus,
        'crash_penalty': config.crash_penalty,
        'traffic_light_violation_penalty': config.traffic_light_violation_penalty,
        'standing_still_penalty': config.standing_still_penalty,
        'already_visited_position_penalty': config.already_visited_position_penalty,
        'features_to_include_in_observation': [
            "walls", "goals", "ice", "broken road", "sand", "traffic",
            "traffic_light_green", "traffic_light_yellow", "traffic_light_red"
        ],
        'use_sliding_observation_window': True,
        'sliding_observation_window_size': 4,
        'use_next_subgoal_direction': True
    }
    
    env = PGTGEnv(**env_kwargs)
    env = TimeLimit(env, max_episode_steps=config.max_episode_steps)
    env = Monitor(env)
    env = FlattenObservation(env)
    
    if seed is not None:
        env.reset(seed=seed)
    
    return env


def create_algorithm(config: PGTGTrainingConfig, env, tensorboard_log: str):
    """Create the specified RL algorithm with optimized hyperparameters."""
    
    if config.algorithm == "DQN":
        return DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=config.learning_rate,
            buffer_size=config.buffer_size,
            learning_starts=1000,
            batch_size=config.batch_size,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            exploration_fraction=config.exploration_fraction,
            exploration_initial_eps=1.0,
            exploration_final_eps=config.exploration_final_eps,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=config.verbose,
            tensorboard_log=tensorboard_log,
            seed=42,
            device="cuda"  # Explicitly use CUDA
        )
    
    elif config.algorithm == "PPO":
        return PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=config.learning_rate,
            n_steps=2048,
            batch_size=config.batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=config.verbose,
            tensorboard_log=tensorboard_log,
            seed=42,
            device="cuda"  # Explicitly use CUDA
        )
    
    elif config.algorithm == "A2C":
        return A2C(
            policy="MlpPolicy",
            env=env,
            learning_rate=config.learning_rate,
            n_steps=5,
            gamma=0.99,
            gae_lambda=1.0,
            ent_coef=0.01,
            vf_coef=0.25,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=config.verbose,
            tensorboard_log=tensorboard_log,
            seed=42,
            device="cuda"  # Explicitly use CUDA
        )
    
    else:
        raise ValueError(f"Unsupported algorithm: {config.algorithm}")


def setup_callbacks(config: PGTGTrainingConfig, eval_env, log_dir: str):
    """Setup training callbacks for evaluation and checkpointing."""
    
    callbacks = []
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval_logs"),
        eval_freq=config.eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        deterministic=config.eval_deterministic,
        render=False,
        verbose=1
    )
    callbacks.append(eval_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=config.save_freq,
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="pgtg_model"
    )
    callbacks.append(checkpoint_callback)
    
    return callbacks


def print_training_info(config: PGTGTrainingConfig, log_dir: str):
    """Print training configuration information."""
    
    print("=" * 60)
    print("PGTG TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Environment: PGTGEnv (Single Environment)")
    print(f"Algorithm: {config.algorithm}")
    print(f"Total timesteps: {config.total_timesteps:,}")
    print(f"Max episode steps: {config.max_episode_steps}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Traffic density: {config.traffic_density}")
    print(f"Map size: {config.random_map_width}x{config.random_map_height}")
    print(f"Final goal bonus: {config.final_goal_bonus}")
    print(f"Crash penalty: {config.crash_penalty}")
    print(f"Log directory: {log_dir}")
    print("=" * 60)


def evaluate_final_model(agent, eval_env, n_episodes: int = 100):
    """Evaluate the final trained model."""
    
    print(f"\nEvaluating final model over {n_episodes} episodes...")
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    crash_count = 0
    
    for episode in range(n_episodes):
        obs, info = eval_env.reset()
        episode_reward = 0
        episode_length = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Check if episode was successful (reached goal)
        if episode_reward > 100:  # Assuming positive reward indicates success
            success_count += 1
        elif episode_reward < -50:  # Assuming large negative reward indicates crash
            crash_count += 1
    
    # Calculate statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    success_rate = success_count / n_episodes
    crash_rate = crash_count / n_episodes
    
    print(f"Mean episode reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean episode length: {mean_length:.2f}")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Crash rate: {crash_rate:.2%}")
    
    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'mean_length': mean_length,
        'success_rate': success_rate,
        'crash_rate': crash_rate
    }


def main():
    """Main training function."""
    
    # Create configuration
    config = PGTGTrainingConfig()
    
    # Setup directories
    script_path = pathlib.Path(__file__).parent.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = script_path / f"logs/{config.algorithm.lower()}_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print_training_info(config, str(log_dir))
    
    # Create single training environment (no multiprocessing)
    print("Creating training environment...")
    train_env = create_pgtg_env(config, seed=42)
    
    print("Creating evaluation environment...")
    eval_env = create_pgtg_env(config, seed=123)
    
    # Create algorithm
    print(f"Initializing {config.algorithm} agent...")
    agent = create_algorithm(config, train_env, str(log_dir))
    
    # Print device info
    print(f"Using device: {agent.device}")
    
    # Setup callbacks
    callbacks = setup_callbacks(config, eval_env, str(log_dir))
    
    # Start training
    print("Starting training...")
    start_time = time.time()
    
    try:
        agent.learn(
            total_timesteps=config.total_timesteps,
            callback=callbacks,
            log_interval=config.log_interval,
            tb_log_name=f"{config.algorithm.lower()}_pgtg"
        )
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds!")
        
        # Save final model
        final_model_path = log_dir / f"final_{config.algorithm.lower()}_agent"
        agent.save(str(final_model_path))
        print(f"Final model saved to: {final_model_path}")
        
        # Final evaluation
        eval_results = evaluate_final_model(agent, eval_env)
        
        # Save evaluation results
        import json
        results_path = log_dir / "final_evaluation.json"
        with open(results_path, 'w') as f:
            json.dump({
                'config': config.__dict__,
                'training_time': training_time,
                'evaluation': eval_results
            }, f, indent=2)
        
        print(f"Evaluation results saved to: {results_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        
        # Save current model
        interrupted_model_path = log_dir / f"interrupted_{config.algorithm.lower()}_agent"
        agent.save(str(interrupted_model_path))
        print(f"Model saved to: {interrupted_model_path}")
    
    finally:
        # Clean up
        train_env.close()
        eval_env.close()
        print("Training completed!")


if __name__ == "__main__":
    main()
