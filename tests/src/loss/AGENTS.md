# tests/src/loss

Loss numerics, routing, masks, and schedules.

Start with the [outermost AGENTS.md](../../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Checks loss values/gradients and data-container routing; golden tests pin seeded outputs.

## File map

| File(s) | Responsibility |
| --- | --- |
| [test_charbonnier_loss.py](test_charbonnier_loss.py) | Unit tests for loss.charbonnier_loss.CharbonnierLoss. |
| [test_gradient_loss.py](test_gradient_loss.py) | Unit tests for loss.gradient_loss.GradientLoss. |
| [test_grpo_loss.py](test_grpo_loss.py) | Unit tests for GRPOLoss. |
| [test_loss_container.py](test_loss_container.py) | Unit tests for loss.loss_container.LossComponent and LossContainer. |
| [test_loss_golden_values.py](test_loss_golden_values.py) | Golden value tests for loss functions. |
| [test_loss_utils.py](test_loss_utils.py) | Unit tests for loss.utils.call_loss_with_epoch. |
| [test_masked_cross_entropy.py](test_masked_cross_entropy.py) | Unit tests for loss.masked_cross_entropy.MaskedCrossEntropy. |
| [test_masked_loss.py](test_masked_loss.py) | Unit tests for loss.masked_loss.MaskedLoss. |
| [test_moving_weighted_loss.py](test_moving_weighted_loss.py) | Unit tests for MovingWeightedLoss. |
| [test_phased_loss.py](test_phased_loss.py) | Unit tests for loss.phased_loss.PhasedLoss. |
| [test_shifted_teacher_forcing.py](test_shifted_teacher_forcing.py) | Unit tests for loss.shifted_teacher_forcing.ShiftedTeacherForcing. |
| [test_ssim.py](test_ssim.py) | Unit tests for SSIMLoss. |
| [test_vgg_loss.py](test_vgg_loss.py) | Unit tests for VGGLoss. |
| [test_weighted_loss.py](test_weighted_loss.py) | Unit tests for loss.weighted_loss.WeightedLoss. |
| [__init__.py](__init__.py) | Package marker/exports; inspect when changing public imports. |

## Working here

Run `uv run pytest tests/src/loss`; default marker exclusions still apply. Consult the corresponding runtime and config guides before changing expected behavior.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
