"""Baseline PPO on HalfCheetah-v5 — default hyperparameters, 300k steps.

Run:    uv run python train.py
Logs:   ./logs/baseline/PPO_<run>/    (view with: tensorboard --logdir logs)
Model:  ./models/baseline.zip
"""
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

Path("models").mkdir(exist_ok=True)
Path("logs").mkdir(exist_ok=True)

env = gym.make("HalfCheetah-v5")
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="logs/baseline")
model.learn(total_timesteps=300_000, progress_bar=True)
model.save("models/baseline")
print("\nSaved -> models/baseline.zip")
