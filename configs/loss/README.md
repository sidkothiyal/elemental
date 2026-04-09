# Loss Function Configs

Loss function configurations for different training objectives.

## Available Losses

### Basic Losses
| File | Description | Use Case |
|------|-------------|----------|
| `mse.yaml` | Mean Squared Error | Regression, autoencoders |
| `l1.yaml` | L1/Mean Absolute Error | Image reconstruction |
| `cross_entropy.yaml` | Cross Entropy | Classification |
| `bce_logits.yaml` | BCE with Logits | Binary classification, GAN |

### Specialized Losses
| File | Description | Use Case |
|------|-------------|----------|
| `phased_mse_ssim.yaml` | Phased MSE→SSIM transition | Image compression |
| `masked_mse.yaml` | Masked MSE | Masked autoencoder |
| `masked_loss.yaml` | Configurable masked loss | MAE variants |
| `shifted_teacher_forcing.yaml` | Shifted loss for LM | Language modeling |
| `shifted_teacher_forcing_char.yaml` | Char-level LM loss | Character LM |
| `gan.yaml` | GAN losses (feature + adversarial) | Segmentation GAN |
| `gan_compression.yaml` | Phased GAN for compression | Compression GAN |

## Usage

### In Experiment Configs

```yaml
defaults:
  - /loss: mse
```

### Phased Loss

The phased loss transitions between losses during training:

```yaml
# Phase 1 (0-20%): Pure L1
# Phase 2 (20-80%): Gradually introduce SSIM
# Phase 3 (80-100%): Fixed L1+SSIM weights
```

## Creating Custom Losses

```yaml
# loss/my_loss.yaml
loss:
  _target_: loss.my_module.MyLoss
  param1: value1
  param2: value2
```

For weighted combinations:
```yaml
loss:
  _target_: loss.weighted_loss.WeightedLoss
  weights: [1.0, 0.5]
  loss_fns:
    - _target_: torch.nn.L1Loss
    - _target_: loss.ssim.SSIMLoss
```
