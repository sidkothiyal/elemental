# configs/scheduler

Learning-rate scheduling recipes.

Start with the [outermost AGENTS.md](../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Hydra YAML here selects or configures runtime components; it does not implement their algorithms. Check defaults order, @package placement, interpolations, _target_ imports, and the final composed config before editing.

## File map

| File(s) | Responsibility |
| --- | --- |
| [cosine_annealing.yaml](cosine_annealing.yaml) | Epoch-based cosine scheduling. |
| [one_cycle.yaml](one_cycle.yaml) | One-cycle schedule. |
| [step_lr.yaml](step_lr.yaml) | Periodic step decay. |
| [nanochat_warmup_cosine.yaml](nanochat_warmup_cosine.yaml) | Optimizer-step warmup plus cosine decay. |
| [README.md](README.md) | Existing usage/design detail; verify historical claims against code. |

## Working here

Keep scheduler step units aligned with gradient accumulation and max_steps.

Validate composition with `uv run pytest tests/integration/test_config_loading.py`; use component tests for behavior changes.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
