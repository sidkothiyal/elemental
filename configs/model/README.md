# Model Configs

Neural network architecture configurations.

## Available Models

### Autoencoder Models
| File | Description | Architecture |
|------|-------------|--------------|
| `unet.yaml` | Basic UNet | UNet encoder-decoder |
| `unet_compression.yaml` | UNet for compression | UNet with skip connections |
| `unet_depth.yaml` | UNet for depth | UNet (3→1 channels) |
| `unet_segmentation.yaml` | UNet for segmentation | UNet (3→N classes) |

### Diffusion Models
| File | Description | Architecture |
|------|-------------|--------------|
| `diffusion_unet.yaml` | Diffusion with UNet | Denoising diffusion model |

### Transformer Models
| File | Description | Architecture |
|------|-------------|--------------|
| `transformer_encoder.yaml` | Transformer encoder | Base transformer |
| `autoregressive_lm.yaml` | Language model | Transformer LM |
| `autoregressive_lm_small.yaml` | Small LM | Tiny transformer |
| `perceiver_resampler.yaml` | Perceiver Resampler | Learnable latents cross-attending to variable-length context |

### Autoencoder Variants
| File | Description | Architecture |
|------|-------------|--------------|
| `masked_autoencoder.yaml` | MAE | ViT-style MAE |
| `mae_div2k.yaml` | MAE for DIV2K | MAE with conv decoder |
| `vq_vae_transformer.yaml` | VQ-VAE | Transformer VQ-VAE |

### GAN Models
| File | Description | Architecture |
|------|-------------|--------------|
| `gan_segmentation.yaml` | Seg GAN | UNet gen + Conv disc |
| `gan_compression_unet.yaml` | Compression GAN | UNet gen + Conv disc |

### Classification
| File | Description | Architecture |
|------|-------------|--------------|
| `classification_cnn.yaml` | CNN classifier | ConvBlock + Linear |

## Usage

### In Experiment Configs

```yaml
defaults:
  - /model: unet_compression
```

### Overriding Model Parameters

```yaml
# In experiment config
model:
  channels: [8, 16, 32, 64, 128]  # Deeper network
```

## Model Config Structure

```yaml
# model/my_model.yaml
model:
  _target_: model.my_module.MyModel
  param1: value1
  param2: value2
```

## Adding Custom Models

1. Implement model class in `src/model/`
2. Create `model/your_model.yaml`:

```yaml
model:
  _target_: model.your_module.YourModel
  in_channels: 3
  out_channels: 3
  # ... other params
```

3. Reference in experiments: `- /model: your_model`
