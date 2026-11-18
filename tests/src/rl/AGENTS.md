# tests/src/rl

Policy loss, rollout, advantage, and token-policy tests.

Start with the [outermost AGENTS.md](../../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Covers individual RL components; test_grpo.py checks advantage mathematics, not full GRPOAlgorithm plus RLTrainer execution.

## File map

| File(s) | Responsibility |
| --- | --- |
| [test_advantages.py](test_advantages.py) | Unit tests for GAE advantage computation. |
| [test_autoregressive_policy_wrapper.py](test_autoregressive_policy_wrapper.py) | Unit tests for the GRPO/RLHF AutoregressivePolicyWrapper. |
| [test_grpo.py](test_grpo.py) | Unit tests for GRPO group-relative advantage computation. |
| [test_policy_losses.py](test_policy_losses.py) | Unit tests for PPO / value / entropy losses. |
| [test_rollout_buffer.py](test_rollout_buffer.py) | Unit tests for OnPolicyRolloutBuffer. |
| [test_text_rollout_buffer.py](test_text_rollout_buffer.py) | Unit tests for TextRolloutBuffer. |
| [__init__.py](__init__.py) | Package marker/exports; inspect when changing public imports. |

## Child guides

| Directory guide | Information |
| --- | --- |
| [verifiers/](verifiers/AGENTS.md) | GSM8K numeric reward tests |

## Working here

Run `uv run pytest tests/src/rl`; default marker exclusions still apply. Consult the corresponding runtime and config guides before changing expected behavior.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
