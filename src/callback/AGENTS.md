# src/callback

Training lifecycle observers and output persistence.

Start with the [outermost AGENTS.md](../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

CallbackList dispatches lifecycle events; initializer normally creates callbacks only on rank zero. Callbacks consume named data-container fields and metrics. Keep observation, logging, and checkpoint timing aligned with optimizer-step accounting.

## File map

| File(s) | Responsibility |
| --- | --- |
| [callback.py](callback.py) | Callback interface and list dispatch. |
| [checkpoint_callback.py](checkpoint_callback.py) | Best/last/periodic checkpoints, resume metadata, remote backup scheduling. |
| [metrics_callback.py](metrics_callback.py) | Phase-specific TorchMetrics instances, key mapping, transforms, logging cadence. |
| [wandb_callback.py](wandb_callback.py) | W&B run/resume management and media/text/scalar logging. |
| [transforms.py](transforms.py) | FlattenTemporal maps video batches to frame batches for image metrics. |
| [rollout_metrics_callback.py](rollout_metrics_callback.py) | Reward/success summaries from the rollout phase. |
| [rollout_video_callback.py](rollout_video_callback.py) | Buffers rendered rollout frames and logs video. |
| [benchmark_callback.py](benchmark_callback.py) | Warmup/measurement windows over real training; writes benchmark.json. |
| [trial_summary_callback.py](trial_summary_callback.py) | Finalizes machine-readable trial status, metrics, and timing for autoresearch. |
| [README.md](README.md), [ARCHITECTURE.md](ARCHITECTURE.md) | Existing usage/design detail; verify historical claims against code. |
| [__init__.py](__init__.py) | Package marker/exports; inspect when changing public imports. |

## Working here

Trial finalization is handled explicitly by Trainer, not merely by adding a standard event method. Benchmark records are throughput evidence, not model-quality validation.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
