# configs/graph

Reusable customizable-trainer batch graphs.

Start with the [outermost AGENTS.md](../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Hydra YAML here selects or configures runtime components; it does not implement their algorithms. Check defaults order, @package placement, interpolations, _target_ imports, and the final composed config before editing.

## File map

| File(s) | Responsibility |
| --- | --- |
| [gan_per_batch.yaml](gan_per_batch.yaml) | Generator/discriminator calls, losses, backward, and optimizer choreography. |
| [distill_per_batch.yaml](distill_per_batch.yaml) | Frozen teacher forward and student optimization. |
| [README.md](README.md) | Existing usage/design detail; verify historical claims against code. |

## Working here

These graphs run inside CustomizableTrainer; standalone graph experiments live under experiment/node.

Validate composition with `uv run pytest tests/integration/test_config_loading.py`; use component tests for behavior changes.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
