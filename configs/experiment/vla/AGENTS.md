# configs/experiment/vla

Vision-language-action diffusion and robotics RL recipes.

Start with the [outermost AGENTS.md](../../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Hydra YAML here selects or configures runtime components; it does not implement their algorithms. Check defaults order, @package placement, interpolations, _target_ imports, and the final composed config before editing.

## File map

| File(s) | Responsibility |
| --- | --- |
| [vla_diffusion_qwen3.yaml](vla_diffusion_qwen3.yaml) | Qwen-conditioned action diffusion on Open-X data. |
| [vla_rl_ppo.yaml](vla_rl_ppo.yaml) | Diffusion-policy PPO with simulator rollouts. |
| [vla_rl_eval_sim.yaml](vla_rl_eval_sim.yaml) | Simulator evaluation configuration. |

## Working here

Requires compatible pretrained models/data and optional SIMPLER installation. The diffusion policy uses final-action Gaussian log probabilities; full DPPO is deferred.

Validate composition with `uv run pytest tests/integration/test_config_loading.py`; use component tests for behavior changes.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
