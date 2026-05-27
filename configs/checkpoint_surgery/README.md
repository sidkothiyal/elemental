# Checkpoint Surgery Configs

Configurations for manipulating and transforming saved model checkpoints.

## Files

| File | Description |
|------|-------------|
| `rename_transformer_autoencoder.yaml` | Rename keys in transformer autoencoder checkpoints |
| `rename_transformer_autoencoder_vq_vae.yaml` | Rename keys in VQ-VAE transformer checkpoints |

## Purpose

These configs are used when:
- Migrating checkpoints between model versions
- Renaming state dict keys after architecture changes
- Loading pretrained weights with different key names

## Usage

These configs are typically used with checkpoint loading utilities rather than the main trainer.
