# configs/profiler

PyTorch profiling presets.

Start with the [outermost AGENTS.md](../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Hydra YAML here selects or configures runtime components; it does not implement their algorithms. Check defaults order, @package placement, interpolations, _target_ imports, and the final composed config before editing.

## File map

| File(s) | Responsibility |
| --- | --- |
| [disabled.yaml](disabled.yaml) | Profiling off. |
| [cpu_only.yaml](cpu_only.yaml) | CPU activity collection. |
| [standard.yaml](standard.yaml) | Standard CPU/CUDA profiling window. |
| [debug.yaml](debug.yaml) | Debug-oriented collection. |
| [full.yaml](full.yaml) | Detailed activity/shape/stack collection. |
| [memory.yaml](memory.yaml) | Memory-focused profiling. |

## Working here

Consumed by training/helpers.py; output and analysis guidance lives in [docs/profiling.md](../../docs/profiling.md).

Validate composition with `uv run pytest tests/integration/test_config_loading.py`; use component tests for behavior changes.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
