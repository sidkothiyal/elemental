# configs/loss

Loss composition and data-container key routing.

Start with the [outermost AGENTS.md](../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Hydra YAML here selects or configures runtime components; it does not implement their algorithms. Check defaults order, @package placement, interpolations, _target_ imports, and the final composed config before editing.

## File map

| File(s) | Responsibility |
| --- | --- |
| [bce_logits.yaml](bce_logits.yaml) | Binary logits objective. |
| [cross_entropy.yaml](cross_entropy.yaml) | Classification objective. |
| [mse.yaml](mse.yaml) | Squared reconstruction error. |
| [l1.yaml](l1.yaml) | Absolute reconstruction error. |
| [charbonnier.yaml](charbonnier.yaml) | Smooth robust reconstruction error. |
| [ssim.yaml](ssim.yaml) | Structural similarity loss. |
| [l1_perceptual.yaml](l1_perceptual.yaml) | L1 plus perceptual reconstruction. |
| [phased_mse_ssim.yaml](phased_mse_ssim.yaml) | Epoch-dependent MSE/SSIM phases. |
| [masked_loss.yaml](masked_loss.yaml) | Generic masked objective. |
| [gan.yaml](gan.yaml), [gan_compression.yaml](gan_compression.yaml), [vq_gan_compression.yaml](vq_gan_compression.yaml) | Generator/discriminator and reconstruction/perceptual loss combinations. |
| [distillation.yaml](distillation.yaml) | Teacher/student feature alignment. |
| [shifted_teacher_forcing.yaml](shifted_teacher_forcing.yaml), [shifted_teacher_forcing_char.yaml](shifted_teacher_forcing_char.yaml) | Token-sequence objectives for generic/character LMs. |
| [lm_pretrain.yaml](lm_pretrain.yaml) | Already-shifted next-token language objective. |
| [lm_pretrain_deep_supervision.yaml](lm_pretrain_deep_supervision.yaml) | Final and per-loop recursive LM supervision. |
| [lm_sft.yaml](lm_sft.yaml) | Assistant-only mask over SFT targets. |
| [hierarchical_thinking_compressor_mae.yaml](hierarchical_thinking_compressor_mae.yaml), [hierarchical_thinking_masked_diffusion.yaml](hierarchical_thinking_masked_diffusion.yaml) | Masked-token compressor and diffusion objectives. |
| [ppo.yaml](ppo.yaml) | Policy/value/entropy losses using rl/* context. |
| [grpo.yaml](grpo.yaml) | Token surrogate with completion-mask context. |
| [vla_huber_slices.yaml](vla_huber_slices.yaml), [vla_mse_slices.yaml](vla_mse_slices.yaml) | Action-slice MSE/Huber objectives. |
| [README.md](README.md) | Existing usage/design detail; verify historical claims against code. |

## Working here

Trace every pred_key, target_key, and context_mapping through the producing trainer. GRPO currently has a completion-mask namespace mismatch; see STATUS.md.

Validate composition with `uv run pytest tests/integration/test_config_loading.py`; use component tests for behavior changes.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
