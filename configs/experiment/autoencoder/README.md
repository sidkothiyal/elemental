# Autoencoder Experiments

Image autoencoder and compression experiments.

## Available Experiments

| Config | Description | Dataset | Model |
|--------|-------------|---------|-------|
| `compression_unet.yaml` | UNet autoencoder with MSE | DIV2K | UNet |
| `compression_unet_l1.yaml` | UNet autoencoder with L1 | DIV2K | UNet |
| `compression_unet_ssim.yaml` | UNet with phased L1+SSIM | DIV2K | UNet |
| `compression_diffusion.yaml` | Diffusion model | DIV2K | Diffusion UNet |
| `compression_conditional_diffusion.yaml` | Conditional diffusion | DIV2K | CNN conditional |
| `compression_transformer_conditional_diffusion.yaml` | Transformer conditional diffusion | DIV2K | Transformer |
| `compression_vq_transformer_conditional_diffusion.yaml` | VQ-VAE + Transformer diffusion | DIV2K | VQ-VAE + Transformer |
| `compression_vqvae.yaml` | VQ-VAE compression | DIV2K | VQ-VAE Transformer |
| `compression_mae.yaml` | Masked autoencoder | DIV2K | MAE |
| `depth_estimation.yaml` | Depth estimation | Cityscapes | UNet |
| `masked_autoencoder.yaml` | Masked autoencoder | Cityscapes | MAE |

## Usage

```bash
# Basic UNet compression
python src/trainer.py experiment/autoencoder/compression_unet

# With SSIM loss
python src/trainer.py experiment/autoencoder/compression_unet_ssim

# Diffusion-based
python src/trainer.py experiment/autoencoder/compression_diffusion

# Depth estimation
python src/trainer.py experiment/autoencoder/depth_estimation
```

## Common Overrides

```bash
# Change batch size and learning rate
python src/trainer.py experiment/autoencoder/compression_unet batch_size=8 lr=0.001

# Change number of epochs
python src/trainer.py experiment/autoencoder/compression_unet num_epochs=500

# Use different dtype
python src/trainer.py experiment/autoencoder/compression_unet dtype=null  # float32
```
