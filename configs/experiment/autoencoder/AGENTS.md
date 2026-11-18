# configs/experiment/autoencoder

Image/video reconstruction, diffusion, and prediction recipes.

Start with the [outermost AGENTS.md](../../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Hydra YAML here selects or configures runtime components; it does not implement their algorithms. Check defaults order, @package placement, interpolations, _target_ imports, and the final composed config before editing.

## File map

| File(s) | Responsibility |
| --- | --- |
| [compression_3d_ae.yaml](compression_3d_ae.yaml) | 3D-convolution video baseline. |
| [compression_perceiver_video_ae_v1.yaml](compression_perceiver_video_ae_v1.yaml) | Shared-latent video V1. |
| [frame_prediction_oxe.yaml](frame_prediction_oxe.yaml) | Next-frame prediction from Open-X clips. |
| [depth_estimation.yaml](depth_estimation.yaml) | Depth regression. |
| [compression_dinov2_ae.yaml](compression_dinov2_ae.yaml), [compression_dinov3_ae.yaml](compression_dinov3_ae.yaml), [compression_dinov3_vit_ae.yaml](compression_dinov3_vit_ae.yaml), [compression_dinov3_vith_ae.yaml](compression_dinov3_vith_ae.yaml) | DINO vision-encoder reconstruction variants. |
| [compression_qwen3_vl_ae.yaml](compression_qwen3_vl_ae.yaml) | Qwen vision-feature reconstruction. |
| [compression_conditional_attn_diffusion.yaml](compression_conditional_attn_diffusion.yaml), [compression_conditional_diffusion.yaml](compression_conditional_diffusion.yaml), [compression_conditional_unet_attn_diffusion.yaml](compression_conditional_unet_attn_diffusion.yaml), [compression_conditional_unet_attn_diffusion_multi_cond.yaml](compression_conditional_unet_attn_diffusion_multi_cond.yaml), [compression_conditional_unet_spatial_diffusion.yaml](compression_conditional_unet_spatial_diffusion.yaml) | Conditioned diffusion with convolution, attention, or spatial conditioning variants. |
| [compression_transformer_conditional_diffusion.yaml](compression_transformer_conditional_diffusion.yaml) | Transformer-conditioned diffusion reconstruction. |
| [compression_vq_transformer_conditional_diffusion.yaml](compression_vq_transformer_conditional_diffusion.yaml) | Quantized transformer-conditioned diffusion. |
| [compression_diffusion.yaml](compression_diffusion.yaml) | Unconditional diffusion reconstruction recipe. |
| [compression_vq_ae.yaml](compression_vq_ae.yaml), [compression_vq_gan.yaml](compression_vq_gan.yaml), [compression_vqvae.yaml](compression_vqvae.yaml) | Vector-quantized autoencoder/GAN recipes. |
| [compression_unet.yaml](compression_unet.yaml), [compression_unet_l1.yaml](compression_unet_l1.yaml), [compression_unet_ssim.yaml](compression_unet_ssim.yaml) | U-Net reconstruction with alternative losses. |
| [compression_mae.yaml](compression_mae.yaml) | Patch-based MAE reconstruction pipeline. |
| [masked_autoencoder.yaml](masked_autoencoder.yaml) | Image masked-autoencoder pretraining. |
| [compression_ae.yaml](compression_ae.yaml), [compression_ae_oxe.yaml](compression_ae_oxe.yaml) | Residual image reconstruction on DIV2K or Open-X. |
| [fast_compression_ae.yaml](fast_compression_ae.yaml) | Faster residual AE recipe. |
| [simple_compression_ae.yaml](simple_compression_ae.yaml) | Minimal convolutional AE recipe. |
| [README.md](README.md) | Existing usage/design detail; verify historical claims against code. |

## Working here

Video V1 uses continuous latents and MSE; frame-flattened metrics do not provide an entropy-coded bitrate. Residual video models and rate-distortion evaluation remain roadmap work.

Known config gap: compression_ae.yaml and fast_compression_ae.yaml set `grad_accum_steps`, but Trainer reads `gradient_accumulation_steps`; the intended accumulation is not applied. See [STATUS.md](../../../docs/STATUS.md).

Validate composition with `uv run pytest tests/integration/test_config_loading.py`; use component tests for behavior changes.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
