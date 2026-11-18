# configs/model

Model construction recipes and shape parameters.

Start with the [outermost AGENTS.md](../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Hydra YAML here selects or configures runtime components; it does not implement their algorithms. Check defaults order, @package placement, interpolations, _target_ imports, and the final composed config before editing.

## File map

| File(s) | Responsibility |
| --- | --- |
| [autoregressive_lm.yaml](autoregressive_lm.yaml), [autoregressive_lm_small.yaml](autoregressive_lm_small.yaml) | Generic language-model sizes. |
| [nanochat_gpt_d12.yaml](nanochat_gpt_d12.yaml), [nanochat_gpt_d20.yaml](nanochat_gpt_d20.yaml), [nanochat_gpt_d26.yaml](nanochat_gpt_d26.yaml) | Dense nanochat GPT depths. |
| [recursive_memory_gpt_1m.yaml](recursive_memory_gpt_1m.yaml), [recursive_memory_gpt_d12eq.yaml](recursive_memory_gpt_d12eq.yaml) | Repeated-trunk PEER models including a million-expert capacity variant. |
| [memory_bank_gpt_d12.yaml](memory_bank_gpt_d12.yaml) | Flat twelve-layer shared-bank control with one loop. |
| [classification_cnn.yaml](classification_cnn.yaml) | Convolutional classifier. |
| [diffusion_unet.yaml](diffusion_unet.yaml) | U-Net denoising model. |
| [dinov2_decoder_ae.yaml](dinov2_decoder_ae.yaml), [dinov2_encoder.yaml](dinov2_encoder.yaml), [dinov3_decoder_ae.yaml](dinov3_decoder_ae.yaml), [dinov3_encoder.yaml](dinov3_encoder.yaml), [dinov3_vit_decoder_ae.yaml](dinov3_vit_decoder_ae.yaml), [dinov3_vith_decoder_ae.yaml](dinov3_vith_decoder_ae.yaml) | DINO encoders and reconstruction decoders. |
| [qwen3_vl_vision_decoder_ae.yaml](qwen3_vl_vision_decoder_ae.yaml) | Qwen vision reconstruction. |
| [sam_encoder.yaml](sam_encoder.yaml) | SAM pretrained vision features. |
| [radio_encoder.yaml](radio_encoder.yaml) | RADIO vision features. |
| [distill_dinov3_to_vit.yaml](distill_dinov3_to_vit.yaml) | Teacher/student module composition. |
| [fast_residual_autoencoder.yaml](fast_residual_autoencoder.yaml) | Reduced-cost residual AE. |
| [residual_autoencoder.yaml](residual_autoencoder.yaml) | Residual image AE. |
| [simple_autoencoder.yaml](simple_autoencoder.yaml) | Basic convolutional AE. |
| [simple_3d_autoencoder.yaml](simple_3d_autoencoder.yaml) | Video 3D-convolution AE. |
| [simple_vq_autoencoder.yaml](simple_vq_autoencoder.yaml) | Basic VQ AE. |
| [residual_vq_gan.yaml](residual_vq_gan.yaml) | Residual quantized adversarial model. |
| [vq_vae_transformer.yaml](vq_vae_transformer.yaml) | Patch/transformer quantized AE. |
| [mae_div2k.yaml](mae_div2k.yaml) | Masked patch autoencoder for DIV2K reconstruction. |
| [masked_autoencoder.yaml](masked_autoencoder.yaml) | Masked-image pretraining architecture. |
| [perceiver_resampler.yaml](perceiver_resampler.yaml) | Generic fixed-latent bottleneck. |
| [perceiver_video_autoencoder.yaml](perceiver_video_autoencoder.yaml) | Shared video latent and learned-query decoder. |
| [frame_predictor.yaml](frame_predictor.yaml) | Pretrained frame encoder/decoder with latent sequence combiner. |
| [gan_compression_unet.yaml](gan_compression_unet.yaml), [gan_segmentation.yaml](gan_segmentation.yaml) | Generator/discriminator ModuleDicts. |
| [unet.yaml](unet.yaml), [unet_compression.yaml](unet_compression.yaml), [unet_depth.yaml](unet_depth.yaml), [unet_segmentation.yaml](unet_segmentation.yaml) | U-Net task heads for reconstruction/depth/segmentation. |
| [transformer_encoder.yaml](transformer_encoder.yaml) | Reusable transformer encoder stack. |
| [vla_diffusion_qwen3.yaml](vla_diffusion_qwen3.yaml) | Qwen-conditioned action denoiser. |
| [README.md](README.md) | Existing usage/design detail; verify historical claims against code. |

## Child guides

| Directory guide | Information |
| --- | --- |
| [hierarchical_thinking/](hierarchical_thinking/AGENTS.md) | Fixed-scale compressor and multi-scale language architectures |

## Working here

Hierarchical model shapes have their own child guide. Changing shared LM parameter names requires matching optimizer partition checks; tokenizer vocab and checkpoint shapes must stay consistent.

Validate composition with `uv run pytest tests/integration/test_config_loading.py`; use component tests for behavior changes.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
