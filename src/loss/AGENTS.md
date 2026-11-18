# src/loss

Composable reconstruction, language, distillation, and policy losses.

Start with the [outermost AGENTS.md](../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

LossComponent resolves predictions, targets, and context from the trainer data dictionary; LossContainer aggregates named weighted components. Underlying loss modules consume tensors. Key routing is part of the training contract.

## File map

| File(s) | Responsibility |
| --- | --- |
| [loss_container.py](loss_container.py) | Key resolution, context_mapping, component weighting, named aggregation. |
| [weighted_loss.py](weighted_loss.py) | Fixed-weight loss combination. |
| [moving_weighted_loss.py](moving_weighted_loss.py) | Epoch-dependent interpolation of component weights. |
| [phased_loss.py](phased_loss.py) | Switches active losses by training phase/epoch. |
| [masked_loss.py](masked_loss.py) | Mask-aware wrapper around a configured loss. |
| [masked_cross_entropy.py](masked_cross_entropy.py) | Token cross-entropy selected by a boolean reconstruction mask. |
| [shifted_teacher_forcing.py](shifted_teacher_forcing.py) | Configurable alignment/shift of token logits and targets. |
| [deep_supervision_ce.py](deep_supervision_ce.py) | Cross-entropy for per-recursion-loop logits. |
| [charbonnier_loss.py](charbonnier_loss.py) | Smooth robust reconstruction error. |
| [gradient_loss.py](gradient_loss.py) | Spatial image-gradient reconstruction loss. |
| [ssim.py](ssim.py) | Structural similarity loss backed by pytorch-msssim. |
| [vgg_loss.py](vgg_loss.py) | VGG feature perceptual loss. |
| [lpips_loss.py](lpips_loss.py) | LPIPS wrapper with input-range conversion and optional resize. |
| [distillation.py](distillation.py) | Teacher/student feature matching with prefix removal and optional projection. |
| [policy_losses.py](policy_losses.py) | PPO clipped policy objective, value regression, and entropy bonus. |
| [grpo_loss.py](grpo_loss.py) | Completion-masked token surrogate with optional reference-policy KL. |
| [utils.py](utils.py) | Calls losses with epoch context where supported. |
| [README.md](README.md), [ARCHITECTURE.md](ARCHITECTURE.md) | Existing usage/design detail; verify historical claims against code. |
| [__init__.py](__init__.py) | Package marker/exports; inspect when changing public imports. |

## Working here

Do not double-shift FineWeb/SmolTalk targets. Video metric flattening does not automatically make SSIMLoss accept 5D tensors. See [loss configs](../../configs/loss/AGENTS.md) and [status](../../docs/STATUS.md) for GRPO context wiring.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
