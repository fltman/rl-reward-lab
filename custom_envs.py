"""Gymnasium wrapper that replaces the env's reward with a custom function."""
import gymnasium as gym


class CustomRewardWrapper(gym.Wrapper):
    """Replace per-step reward with reward_fn(obs, action, info).

    Termination, observations, and action space are unchanged. The original
    reward components are still available via `info` (e.g. info["reward_forward"]
    on HalfCheetah), so reward_fn can shape from them.
    """

    def __init__(self, env, reward_fn):
        super().__init__(env)
        self.reward_fn = reward_fn

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = float(self.reward_fn(obs, action, info))
        return obs, reward, terminated, truncated, info
