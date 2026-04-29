# RL Reward Lab

Train a simulated cheetah from a plain-English goal. A local LLM (Gemma 4 e4b
on Apple Silicon) writes the reward function, PPO trains HalfCheetah on it for
300k steps, and you watch the policy evolve live in the browser.

```
"run forward as fast as possible"   → cheetah runs
"run backwards"                     → cheetah runs left
"jump straight up repeatedly"       → cheetah hops in place
"balance on one foot"               → cheetah… mostly stands still
```

All inference is on-device. No API keys, no cloud, no GPU required.

## What this is (and isn't)

This is a small teaching project that demonstrates two things:

1. **Reward shaping is everything.** Same agent, same physics, same
   algorithm — change only the reward function and the behaviour swings from
   "elegant gait" to "rolls backwards" to "lies still and vibrates".
2. **Natural language → behaviour** via an LLM that writes the reward.
   Often beautifully. Sometimes pathologically — see "hop on one leg" → just
   stand still, a textbook case of reward hacking.

It is **not** a serious RL framework, not optimised for performance, not
multi-user, not a research benchmark. ~5 small Python files plus a single-page
web demo.

## How it works

```
  goal text  ──►  Gemma 4 e4b  ──►  reward_fn(obs, action, info)
                  (local MLX)
                                     │
                                     ▼
              PPO + HalfCheetah-v5 (300k steps, ~60s on M-series CPU)
                                     │
                ┌────────────────────┼─────────────────────┐
                ▼                    ▼                     ▼
         live reward chart   evolution clips        final 1000-step video
                             (every 12 rollouts)
```

The web app streams training progress over Server-Sent Events. Mid-training
mini-clips are rendered every ~12 PPO rollouts (so ~12 snapshots over a full
run), each ~5 seconds at 240×240, autoplaying side by side so you can watch the
gait coalesce in front of you.

## Requirements

- macOS on Apple Silicon (M1/M2/M3/M4). Linux/Intel will work for everything
  except the local Gemma path; swap in your favourite text-completion CLI.
- [`uv`](https://docs.astral.sh/uv/) for environment management.
- A working local **Gemma 4 e4b** at `~/Projekt/gemma/bin/gemma`. Set up via
  [fltman/gemma4-mac](https://github.com/fltman/gemma4-mac), or change the
  `GEMMA` constant in `llm_reward.py` to point at any CLI that takes a prompt
  and prints a completion.
- ~1 GB disk for Python deps; another ~3.5 GB for the Gemma weights (cached
  to `~/.cache/huggingface/hub/`).

## Setup

```bash
git clone https://github.com/<you>/rl-reward-lab.git
cd rl-reward-lab
uv sync
```

`uv sync` installs into `.venv/` from the locked `pyproject.toml` / `uv.lock`.
Python 3.12 is used (managed by uv).

## Usage

### Web demo (recommended)

```bash
uv run uvicorn web.server:app --host 127.0.0.1 --port 8765
```

Open <http://127.0.0.1:8765/>, type a goal, click Run. One run takes ~90 s end
to end (Gemma ~25–35 s, training ~50–60 s with mid-rollout clip rendering,
final eval ~5 s).

### Command-line scripts

```bash
# 1. Baseline: train PPO with HalfCheetah's native forward-running reward.
uv run python train.py

# 2. Render any trained policy to mp4.
uv run python evaluate.py baseline

# 3. A hand-written custom reward that flips the sign of the forward term —
#    cheetah learns to run backwards.
uv run python train_custom.py
uv run python evaluate.py backwards

# 4. Have Gemma write a reward function from natural language.
uv run python llm_reward.py "jump straight up repeatedly without moving sideways"
# inspect generated_rewards/<slug>.py

# 5. Train using the generated reward, then evaluate.
uv run python train_generated.py jump_straight_up_repeatedly_without_moving_sideway
uv run python evaluate.py jump_straight_up_repeatedly_without_moving_sideway
```

## Project structure

```
rl-reward-lab/
├── train.py                # baseline PPO on HalfCheetah-v5 (300k steps)
├── evaluate.py             # render any model in models/ to mp4
├── custom_envs.py          # CustomRewardWrapper(env, reward_fn)
├── train_custom.py         # backwards-running example
├── llm_reward.py           # Gemma → generated_rewards/<slug>.py
├── train_generated.py      # train PPO on a generated reward
├── web/
│   ├── server.py           # FastAPI + SSE orchestrator
│   ├── _runner.py          # subprocess pipeline (Gemma → train → eval)
│   └── index.html          # vanilla HTML + Chart.js, no build step
├── generated_rewards/      # example LLM outputs (kept for illustration)
├── pyproject.toml / uv.lock
└── README.md
```

## Technical notes

A handful of things that are not obvious if you try to rebuild this:

- **`gymnasium` is pinned `<1.3`.** Stable-Baselines3 2.8 requires
  `gymnasium<1.3.0`; if uv resolves freely it walks back through old SB3
  versions and tries to build the legacy `gym==0.21.0`, which fails on
  modern setuptools.
- **No `stable-baselines3[extra]`.** Its extras pull `ale-py`, which pulls
  `gym==0.21.0`, same build failure. Atari isn't needed for HalfCheetah anyway.
- **MuJoCo rendering on macOS requires the main thread.** That's why the web
  pipeline runs in a *subprocess* (`web/_runner.py`) rather than a thread.
  Calling `env.render()` off-main-thread crashes the process at the OS level
  with no Python traceback.
- **Gemma 4 e4b leaks special tokens.** Both `<|channel>thought ... <channel|>`
  reasoning blocks and stray `<unused42>`-type reserved tokens appear in
  outputs and break the imported file unless stripped. `strip_fences` in
  `llm_reward.py` handles both.

## Reward shaping pitfalls (the actually interesting part)

Some example failure modes you'll find:

- **"Hop on one leg"** → Gemma rewards instantaneous z-velocity. Over a
  closed jump cycle that integrates to zero, while the penalty terms reward
  stillness. Result: the agent stands still.
- **"Jump as high as possible"** → if the reward is "z-velocity" rather than
  "height above baseline", same problem.
- **"Stay alive"** → ambiguous. Gemma usually picks "minimise everything",
  giving you a vibrating corpse.

Each of these is a tiny preview of the *specification gap* — the distance
between what we said and what we meant — that makes the broader AI alignment
conversation hard. Watching it manifest in real-time on a 2D cheetah is more
pedagogically convincing than most whitepapers.

## Stack

- [MuJoCo](https://mujoco.org/) physics
- [gymnasium](https://gymnasium.farama.org/) env API (HalfCheetah-v5)
- [stable-baselines3](https://stable-baselines3.readthedocs.io/) (PPO)
- [Gemma 4 e4b](https://huggingface.co/mlx-community/gemma-4-e4b-it-4bit) via
  Apple [MLX](https://github.com/ml-explore/mlx-lm)
- [FastAPI](https://fastapi.tiangolo.com/) + Server-Sent Events
- [Chart.js](https://www.chartjs.org/) for the live training curve

## Licence

MIT.
