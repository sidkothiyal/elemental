# configs/callback

Checkpoint, logging, metric, and rollout callback wiring.

Start with the [outermost AGENTS.md](../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Hydra YAML here selects or configures runtime components; it does not implement their algorithms. Check defaults order, @package placement, interpolations, _target_ imports, and the final composed config before editing.

## File map

| File(s) | Responsibility |
| --- | --- |
| [default.yaml](default.yaml) | Default callback bundle and composition order. |
| [checkpoint.yaml](checkpoint.yaml) | Save cadence and checkpoint options. |
| [wandb.yaml](wandb.yaml) | W&B logging settings. |
| [benchmark.yaml](benchmark.yaml) | Throughput-window observer. |
| [trial_summary.yaml](trial_summary.yaml) | Autoresearch JSON summary observer. |
| [degraded_image_reconstruction_metrics.yaml](degraded_image_reconstruction_metrics.yaml), [image_reconstruction_metrics.yaml](image_reconstruction_metrics.yaml) | Image reconstruction and repeated-degradation metrics. |
| [video_reconstruction_metrics.yaml](video_reconstruction_metrics.yaml) | Frame-flattened MSE/PSNR/SSIM for video. |
| [lm_eval_metrics.yaml](lm_eval_metrics.yaml) | Language validation/test metrics and correct prediction/target routing. |
| [vla_action_metrics.yaml](vla_action_metrics.yaml) | Action-space evaluation. |
| [rollout_metrics.yaml](rollout_metrics.yaml), [rollout_video.yaml](rollout_video.yaml) | Rollout reward/success and video logging. |
| [README.md](README.md) | Existing usage/design detail; verify historical claims against code. |

## Working here

Global callbacks are under `callbacks` (plural). Preserve metric composition order so later defaults do not overwrite selected metrics.

Validate composition with `uv run pytest tests/integration/test_config_loading.py`; use component tests for behavior changes.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
