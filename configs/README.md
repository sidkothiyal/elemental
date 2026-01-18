# Configs

Hierarchical Hydra configuration system for the Elemental training framework.

## Directory Structure

```
configs/
├── config.yaml              # Main entry point
├── default.yaml             # Base training defaults
├── callback/                # Callback configurations
├── checkpoint_surgery/      # Checkpoint manipulation configs
├── dataset/                 # Dataset configurations
├── experiment/              # Complete experiment configs (main entry points)
│   ├── autoencoder/        # Autoencoder experiments
│   ├── autoregressive/     # Language model experiments
│   ├── classification/     # Classification experiments
│   └── gan/                # GAN experiments
├── loss/                    # Loss function configurations
├── model/                   # Model architecture configurations
├── optimizer/               # Optimizer configurations
└── scheduler/               # Learning rate scheduler configurations
```

## Usage

### Running Experiments (Hierarchical)

```bash
# Autoencoder experiments
python src/trainer.py experiment/autoencoder/compression_unet
python src/trainer.py experiment/autoencoder/compression_diffusion

# Classification
python src/trainer.py experiment/classification/imagenet

# GAN
python src/trainer.py experiment/gan/cityscapes_seg

# Language Models
python src/trainer.py experiment/autoregressive/wikitext_lm
```

### Legacy Flat Configs

Old flat configs still work:
```bash
python src/trainer.py compression_autoencoder_unet
python src/trainer.py base_classification
```

## How Composition Works

Experiment configs use Hydra's defaults to compose from reusable components:

```yaml
# experiment/autoencoder/compression_unet.yaml
defaults:
  - /default                    # Base training params
  - /dataset: div2k             # Dataset config
  - /model: unet_compression    # Model architecture
  - /loss: mse                  # Loss function
  - /optimizer: adamw           # Optimizer
  - /scheduler: cosine_annealing # LR scheduler
  - /callback: default          # Callbacks (wandb + checkpoint)
  - _self_

# Override specific values
lr: 0.0003
batch_size: 4
```

## Overriding Values

Override any parameter from command line:
```bash
python src/trainer.py experiment/autoencoder/compression_unet lr=0.001 batch_size=8
```

Override nested values:
```bash
python src/trainer.py experiment/autoencoder/compression_unet callbacks.checkpoint.save_every_n_epochs=5
```
