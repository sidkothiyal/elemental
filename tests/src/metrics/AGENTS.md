# tests/src/metrics

Image, language, and action metric tests.

Start with the [outermost AGENTS.md](../../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Checks sample/aggregate values, ranges, reductions, and reset behavior.

## File map

| File(s) | Responsibility |
| --- | --- |
| [test_action_metrics.py](test_action_metrics.py) | Unit tests for action metrics (ActionADE, ActionFDE, ActionMAE, GripperL1). |
| [test_base_metric.py](test_base_metric.py) | Tests for BaseMetric extended statistics and concrete metric refactors. |
| [test_bits_per_byte.py](test_bits_per_byte.py) | Unit tests for BitsPerByte. |
| [test_iou.py](test_iou.py) | Unit tests for IoU metric. |
| [test_mse.py](test_mse.py) | Unit tests for MSE metric. |
| [test_next_token_accuracy.py](test_next_token_accuracy.py) | Unit tests for NextTokenAccuracy. |
| [test_psnr.py](test_psnr.py) | Unit tests for PSNR metric. |
| [test_ssim.py](test_ssim.py) | Unit tests for SSIM metric. |
| [__init__.py](__init__.py) | Package marker/exports; inspect when changing public imports. |

## Working here

Run `uv run pytest tests/src/metrics`; default marker exclusions still apply. Consult the corresponding runtime and config guides before changing expected behavior.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
