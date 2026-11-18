# src/env

Vectorized robotics and prompt environments.

Start with the [outermost AGENTS.md](../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

BaseVecEnv separates observation/reward dynamics from policy learning. PPO uses reset/step; GRPO’s prompt iterator supplies independent prompt batches rather than simulator transitions.

## File map

| File(s) | Responsibility |
| --- | --- |
| [base_vec_env.py](base_vec_env.py) | Vectorized reset/step/render/close interface. |
| [dummy_vec_env.py](dummy_vec_env.py) | Deterministic lightweight environment for tests and smoke runs. |
| [simpler_env.py](simpler_env.py) | SIMPLER/ManiSkill robotics adapter and observation/action conversion. |
| [prompt_iterator_env.py](prompt_iterator_env.py) | Prompt/reference batching from a text dataset for GRPO. |
| [README.md](README.md) | Existing usage/design detail; verify historical claims against code. |
| [__init__.py](__init__.py) | Package marker/exports; inspect when changing public imports. |

## Working here

SIMPLER is an optional external simulator dependency; its dependency entry is commented out in pyproject.toml. Dummy environment tests do not validate simulator installation.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
