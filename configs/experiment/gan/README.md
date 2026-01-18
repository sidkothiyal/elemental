# GAN Experiments

Generative Adversarial Network experiments.

## Available Experiments

| Config | Description | Dataset | Generator | Discriminator |
|--------|-------------|---------|-----------|---------------|
| `cityscapes_seg.yaml` | Semantic segmentation GAN | Cityscapes | UNet | ConvBlock |
| `compression_unet.yaml` | Image compression GAN | DIV2K | UNet | ConvBlock |

## Usage

```bash
# Segmentation GAN
python src/trainer.py experiment/gan/cityscapes_seg

# Compression GAN
python src/trainer.py experiment/gan/compression_unet
```

## GAN Mode

GANs use `mode: "gan"` which triggers special handling in the trainer:
- Separate optimizers for generator and discriminator
- Alternating training updates
- GAN-specific loss computation

## Model Structure

GAN models use `torch.nn.ModuleDict` with `generator` and `discriminator` keys:

```yaml
model:
  _target_: torch.nn.ModuleDict
  modules:
    generator:
      _target_: model.unet.UNet
      # ...
    discriminator:
      _target_: model.blocks.ConvBlock
      # ...
```

## Loss Structure

GAN losses typically have `feature_matching` and `adversarial` components:

```yaml
loss:
  feature_matching:
    _target_: torch.nn.L1Loss  # or phased loss
  adversarial:
    _target_: torch.nn.BCEWithLogitsLoss
```

## Common Overrides

```bash
# Adjust learning rate
python src/trainer.py experiment/gan/cityscapes_seg lr=0.0002

# Change batch size
python src/trainer.py experiment/gan/compression_unet batch_size=16
```
