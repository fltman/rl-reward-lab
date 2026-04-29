"""Train PPO with a custom reward — here: reward backward motion.

HalfCheetah's native reward is `reward_forward + reward_ctrl` (info dict).
We negate `reward_forward` so the agent learns to run *backwards*, while
keeping the same control-cost penalty so it still cares about smooth gait.

Run:    uv run python train_custom.py
Verify: uv run python evaluate.py backwards   (native reward will go negative)
"""
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

from custom_envs import CustomRewardWrapper


def backwards_reward(obs, action, info):
    return -info["reward_forward"] + info["reward_ctrl"]


Path("models").mkdir(exist_ok=True)
Path("logs").mkdir(exist_ok=True)

env = gym.make("HalfCheetah-v5")
env = CustomRewardWrapper(env, backwards_reward)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="logs/backwards")
model.learn(total_timesteps=300_000, progress_bar=True)
model.save("models/backwards")
print("\nSaved -> models/backwards.zip")
