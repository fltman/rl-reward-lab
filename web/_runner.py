"""Subprocess pipeline: Gemma -> PPO with mid-training clips -> final eval.

Invoked by web/server.py as: python web/_runner.py <goal> <run_id>
Emits one JSON event per stdout line, e.g.:
    {"event": "status", "data": {"phase": "training", ...}}

Runs in its own process so MuJoCo's renderer gets the main thread (required
on macOS — calling env.render() off-main-thread crashes the process).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import gymnasium as gym
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

ROOT = Path(__file__).resolve().parent.parent
WEB = Path(__file__).resolve().parent
RUNS = WEB / "runs"

sys.path.insert(0, str(ROOT))
from custom_envs import CustomRewardWrapper  # noqa: E402
import llm_reward as llm_mod                  # noqa: E402

GEMMA = llm_mod.GEMMA
PROMPT_TEMPLATE = llm_mod.PROMPT_TEMPLATE
strip_fences = llm_mod.strip_fences

TIMESTEPS = 300_000
CLIP_FRAMES = 150         # ~5s at 30fps
CLIP_EVERY = 12           # rollouts; 147 total -> ~12 clips
RENDER_W = 240
RENDER_H = 240


def emit(event: str, **data) -> None:
    print(json.dumps({"event": event, "data": data}), flush=True)


def call_gemma(goal: str) -> str:
    emit("status", phase="generating", message="Asking Gemma 4 e4b…")
    prompt = PROMPT_TEMPLATE.replace("__GOAL__", goal)
    proc = subprocess.run(
        [GEMMA, prompt], capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"gemma exit {proc.returncode}: {proc.stderr[:400]}")
    raw = proc.stdout
    if "ready." in raw:
        raw = raw.split("ready.", 1)[1]
    return strip_fences(raw)


def import_reward(code: str, run_dir: Path):
    path = run_dir / "reward.py"
    path.write_text(code + "\n")
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"_reward_{run_dir.name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "reward_fn"):
        raise RuntimeError("generated code has no reward_fn(obs, action, info)")
    return mod.reward_fn


class StreamCallback(BaseCallback):
    def __init__(self, run_dir: Path, run_id: str):
        super().__init__()
        self.run_dir = run_dir
        self.run_id = run_id
        self.rollout_idx = 0
        self.render_env = None

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self.rollout_idx += 1
        ep_rew = self.logger.name_to_value.get("rollout/ep_rew_mean", 0.0)
        emit("train_step",
             step=int(self.num_timesteps),
             reward=float(ep_rew),
             iter=self.rollout_idx,
             total_steps=TIMESTEPS)
        if self.rollout_idx == 1 or self.rollout_idx % CLIP_EVERY == 0:
            url = self._render_clip()
            emit("clip", url=url, iter=self.rollout_idx,
                 step=int(self.num_timesteps), reward=float(ep_rew))

    def _on_training_end(self) -> None:
        if self.render_env is not None:
            self.render_env.close()
            self.render_env = None

    def _render_clip(self) -> str:
        if self.render_env is None:
            self.render_env = gym.make(
                "HalfCheetah-v5", render_mode="rgb_array",
                width=RENDER_W, height=RENDER_H,
            )
        frames = []
        obs, _ = self.render_env.reset(seed=42)
        for _ in range(CLIP_FRAMES):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = self.render_env.step(action)
            frames.append(self.render_env.render())
            if term or trunc:
                break
        path = self.run_dir / f"clip_{self.rollout_idx:03d}.mp4"
        imageio.mimsave(path, frames, fps=30)
        return f"/runs/{self.run_id}/{path.name}"


def render_final(model: PPO, run_dir: Path, run_id: str) -> tuple[str, float]:
    emit("status", phase="rendering", message="Rendering final 1000-step rollout…")
    env = gym.make("HalfCheetah-v5", render_mode="rgb_array")
    frames = []
    total = 0.0
    obs, _ = env.reset(seed=0)
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(action)
        total += r
        frames.append(env.render())
        if term or trunc:
            obs, _ = env.reset()
    env.close()
    path = run_dir / "final.mp4"
    imageio.mimsave(path, frames, fps=30)
    return f"/runs/{run_id}/{path.name}", float(total)


def main() -> None:
    if len(sys.argv) != 3:
        emit("error", message="usage: _runner.py <goal> <run_id>")
        sys.exit(2)
    goal, run_id = sys.argv[1], sys.argv[2]
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    code = call_gemma(goal)
    emit("code", code=code, goal=goal)
    reward_fn = import_reward(code, run_dir)

    emit("status", phase="training",
         message=f"Training PPO for {TIMESTEPS:,} steps…")
    env = CustomRewardWrapper(gym.make("HalfCheetah-v5"), reward_fn)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=TIMESTEPS,
                callback=StreamCallback(run_dir, run_id),
                progress_bar=False)
    env.close()

    url, native = render_final(model, run_dir, run_id)
    emit("final", url=url, native_reward=native,
         elapsed_seconds=int(time.time() - started), run_id=run_id)
    emit("status", phase="done", message="Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        emit("error", message=f"{type(exc).__name__}: {exc}")
        sys.exit(1)
