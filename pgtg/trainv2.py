import os, pathlib
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.logger import configure
import torch
import environment

print("Training script started...")

# Prevent thread thrashing across subprocess envs
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
torch.set_num_threads(1)

def make_env(seed=0):
    def _thunk():
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
        return env
    return _thunk

if __name__ == "__main__":
    script_dir = pathlib.Path(__file__).parent.resolve()
    log_dir = pathlib.Path(os.environ.get("LOGDIR", f"/netscratch/{os.environ.get('USER','user')}/pgtg_logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    # use all allocated CPUs by default
    n_envs = int(os.environ.get("N_ENVS", os.environ.get("SLURM_CPUS_PER_TASK", 16)))
    n_envs = max(1, n_envs)

    # vectorized envs across processes
    env = SubprocVecEnv([make_env(seed=i) for i in range(n_envs)], start_method="forkserver")
    env = VecMonitor(env, filename=str(log_dir / "monitor"))

    # choose n_steps so that n_steps * n_envs is divisible by batch_size
    n_steps = 128  # with n_envs=16 -> rollout = 2048
    batch_size = 1024  # divides 2048; adjust if you change n_envs/n_steps

    agent = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=n_steps,
        batch_size=batch_size,
        tensorboard_log=str(log_dir),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # live logs to stdout + csv + tensorboard
    agent.set_logger(configure(str(log_dir), ["stdout", "csv", "tensorboard"]))

    agent.learn(total_timesteps=2_000_000, log_interval=1, progress_bar=True)
    agent.save(script_dir / "ppo_agent")
