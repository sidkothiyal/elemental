# tests/src/callback

Callback persistence, metrics, logging, and benchmark tests.

Start with the [outermost AGENTS.md](../../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Uses small or mocked runtime state to check lifecycle behavior, checkpoint timing, media transforms, and benchmark accounting.

## File map

| File(s) | Responsibility |
| --- | --- |
| [test_benchmark_callback.py](test_benchmark_callback.py) | Unit tests for callback.benchmark_callback.BenchmarkCallback. |
| [test_callback.py](test_callback.py) | Unit tests for callback.callback.Callback and CallbackList. |
| [test_checkpoint_callback.py](test_checkpoint_callback.py) | Unit tests for callback.checkpoint_callback.CheckpointCallback. |
| [test_metrics_callback.py](test_metrics_callback.py) | Unit tests for callback.metrics_callback.MetricsCallback. |
| [test_transforms.py](test_transforms.py) | Unit tests for callback.transforms. |
| [test_wandb_callback.py](test_wandb_callback.py) | Unit tests for WandbCallback. |
| [__init__.py](__init__.py) | Package marker/exports; inspect when changing public imports. |

## Working here

Run `uv run pytest tests/src/callback`; default marker exclusions still apply. Consult the corresponding runtime and config guides before changing expected behavior.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
