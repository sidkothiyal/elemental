# src/metrics

Streaming image, language, action, and episode metrics.

Start with the [outermost AGENTS.md](../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

TorchMetrics-compatible modules maintain accumulated state; callbacks select data keys, apply transforms, and reset by phase. BaseMetric also supports sample-level values where meaningful.

## File map

| File(s) | Responsibility |
| --- | --- |
| [base_metric.py](base_metric.py) | Common reduction/state and per-sample metric interface. |
| [mse.py](mse.py) | Mean squared error with normalization/range controls. |
| [psnr.py](psnr.py) | Peak signal-to-noise ratio. |
| [ssim.py](ssim.py) | Structural similarity. |
| [iou.py](iou.py) | Segmentation intersection-over-union; deliberately lacks per-sample scalar API. |
| [bits_per_byte.py](bits_per_byte.py) | Language cross-entropy normalized by tokenizer byte lengths. |
| [next_token_accuracy.py](next_token_accuracy.py) | Token prediction accuracy with shift/mask handling. |
| [action_mae.py](action_mae.py) | Action mean absolute error. |
| [action_displacement.py](action_displacement.py) | Action-trajectory displacement error. |
| [gripper_accuracy.py](gripper_accuracy.py) | Gripper-state prediction accuracy. |
| [episode_metrics.py](episode_metrics.py) | Episode return/length/success accumulation. |
| [README.md](README.md) | Existing usage/design detail; verify historical claims against code. |
| [__init__.py](__init__.py) | Package marker/exports; inspect when changing public imports. |

## Working here

Match metric data_range to actual pixel normalization; video callbacks flatten time into batch. The IoU NotImplementedError is an intentional interface limitation, not an unfinished global IoU implementation.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
