"""Train PPO using a reward function produced by llm_reward.py.

Loads generated_rewards/<name>.py, wraps HalfCheetah with its reward_fn,
trains for 300k PPO steps, saves to models/<name>.zip.

Usage:  uv run python train_generated.py hop_on_one_leg
Then:   uv run python evaluate.py hop_on_one_leg
"""
import importlib
import sys
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

from custom_envs import CustomRewardWrapper

if len(sys.argv) != 2:
    sys.exit("usage: python train_generated.py <slug>")

name = sys.argv[1]
reward_fn = importlib.import_module(f"generated_rewards.{name}").reward_fn

Path("models").mkdir(exist_ok=True)
Path("logs").mkdir(exist_ok=True)

env = CustomRewardWrapper(gym.make("HalfCheetah-v5"), reward_fn)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"logs/{name}")
model.learn(total_timesteps=300_000, progress_bar=True)
model.save(f"models/{name}")
print(f"\nSaved -> models/{name}.zip")
