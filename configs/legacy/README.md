# Legacy Configurations

This folder contains the original flat configuration files that have been superseded by the new hierarchical configuration structure.

## Status

These configs are kept for **backward compatibility** and **reference**. New experiments should use the hierarchical configs in `configs/experiment/`.

## Mapping

| Legacy Config | New Hierarchical Config |
|---------------|-------------------------|
| `base_autoencoder.yaml` | `experiment/autoencoder/depth_estimation` |
| `base_autoregressive_lm.yaml` | `experiment/autoregressive/wikitext_lm` |
| `base_autoregressive_tiny_shakespeare.yaml` | `experiment/autoregressive/tiny_shakespeare` |
| `base_classification.yaml` | `experiment/classification/imagenet` |
| `base_gan.yaml` | `experiment/gan/cityscapes_seg` |
| `base_masked_autoencoder.yaml` | `experiment/autoencoder/masked_autoencoder` |
| `compression_autoencoder_conditional_diffusion.yaml` | `experiment/autoencoder/compression_conditional_diffusion` |
| `compression_autoencoder_diffusion.yaml` | `experiment/autoencoder/compression_diffusion` |
| `compression_autoencoder_gan_unet.yaml` | `experiment/gan/compression_unet` |
| `compression_autoencoder_mae.yaml` | `experiment/autoencoder/compression_mae` |
| `compression_autoencoder_transformer_conditional_diffusion.yaml` | `experiment/autoencoder/compression_transformer_conditional_diffusion` |
| `compression_autoencoder_unet.yaml` | `experiment/autoencoder/compression_unet` |
| `compression_autoencoder_unet_4_8_16_32_64_skip_first.yaml` | `experiment/autoencoder/compression_unet_l1` |
| `compression_autoencoder_unet_4_8_16_32_64_skip_first_ssim.yaml` | `experiment/autoencoder/compression_unet_ssim` |
| `compression_autoencoder_vq_transformer_conditional_diffusion.yaml` | `experiment/autoencoder/compression_vq_transformer_conditional_diffusion` |
| `compression_autoencoder_vq_vae.yaml` | `experiment/autoencoder/compression_vqvae` |

## Usage

Legacy configs can still be used with:

```bash
python src/trainer.py legacy/base_autoencoder
```

However, the recommended approach is to use the new hierarchical configs:

```bash
python src/trainer.py experiment/autoencoder/depth_estimation
```
