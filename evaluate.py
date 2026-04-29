"""Render a trained policy and save as MP4.

Usage: uv run python evaluate.py <name>
       (loads ./models/<name>.zip, writes ./videos/<name>.mp4)
"""
import sys
from pathlib import Path

import gymnasium as gym
import imageio
from stable_baselines3 import PPO

if len(sys.argv) != 2:
    sys.exit("usage: python evaluate.py <model_name>")

name = sys.argv[1]
model_path = Path(f"models/{name}.zip")
if not model_path.exists():
    sys.exit(f"model not found: {model_path}")

Path("videos").mkdir(exist_ok=True)
video_path = Path(f"videos/{name}.mp4")

env = gym.make("HalfCheetah-v5", render_mode="rgb_array")
model = PPO.load(model_path, env=env)

frames = []
total_reward = 0.0
obs, _ = env.reset(seed=0)
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    frames.append(env.render())
    if terminated or truncated:
        obs, _ = env.reset()
env.close()

imageio.mimsave(video_path, frames, fps=30)
print(f"reward (1000 steps): {total_reward:.2f}")
print(f"saved -> {video_path}")
