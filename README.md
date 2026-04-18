# Elemental

A modular PyTorch ML training library supporting image compression, classification, generative models (diffusion, GANs), and autoregressive language models. Configuration is managed by [Hydra](https://hydra.cc/), experiments are tracked with [Weights & Biases](https://wandb.ai/).

## Key Features

- **Multi-task training**: autoencoder, classification, GAN, diffusion, masked autoencoder, autoregressive LM
- **Hydra-based config**: hierarchical, composable YAML configs with experiment overrides
- **Remote datasets**: ZeroMQ dataset server + client for training on data held on another machine
- **Graph-based trainer**: node/computation-graph alternative to the standard trainer
- **Mixed precision**: bfloat16/float16 with optional `GradScaler`
- **Distributed training**: PyTorch DDP via `torchrun`
- **Flexible loss system**: composable `LossContainer`, phased, weighted, moving-weighted losses
- **W&B integration**: images, text, 3D point clouds, auto-resume
- **Checkpoint management**: best/last/periodic saves, remote SSH/W&B backup, auto-discovery of latest

## Installation

```bash
# Install with uv (recommended)
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

Requirements: Python 3.11–3.12, PyTorch ≥ 2.9, CUDA recommended.

## Running Training

All training goes through `src/trainer.py` using Hydra's Compose API:

```bash
# Standard training
uv run python src/trainer.py <experiment_config>

# Image compression autoencoder
uv run python src/trainer.py experiment/autoencoder/compression_unet

# ImageNet classification
uv run python src/trainer.py experiment/classification/imagenet

# WikiText language model
uv run python src/trainer.py experiment/autoregressive/wikitext_lm

# GAN segmentation
uv run python src/trainer.py experiment/gan/cityscapes_seg

# Override any config value on the command line (Hydra syntax)
uv run python src/trainer.py experiment/autoencoder/compression_unet lr=0.001 num_epochs=500

# Distributed training (2 GPUs)
torchrun --nproc_per_node=2 src/trainer.py experiment/autoencoder/compression_unet distributed=true
```

### Graph-based trainer

```bash
uv run python src/trainer.py experiment/node/node_train_autoencoder   # mode: graph dispatches to GraphTrainer
```

## Directory Structure

```
elemental/
├── src/
│   ├── trainer.py            # Main training entry point
│   ├── graph_trainer.py      # Node-graph-based training
│   ├── callback/             # Training callbacks (checkpoint, metrics, W&B)
│   ├── data_processor/       # Dataset downloading, tokenization
│   ├── dataset/              # Dataset implementations
│   ├── loss/                 # Loss functions (MSE, SSIM, VGG, phased, etc.)
│   ├── metrics/              # TorchMetrics wrappers (IoU, MSE, PSNR, SSIM)
│   ├── model/                # Model architectures
│   │   └── blocks/           # Building blocks (conv, resnet, linear, etc.)
│   ├── nodes/                # Computation graph nodes (graph_trainer)
│   ├── postprocessor/        # Output post-processing pipeline
│   ├── tools/                # Utilities (checkpoint_surgeon)
│   └── utils/                # Shared utilities (checkpoint, distributed, etc.)
│       └── remote_backup/    # Remote backup strategies (SSH, W&B)
├── configs/
│   ├── default.yaml          # Base config shared by all experiments
│   ├── config.yaml           # Hydra entry point
│   ├── experiment/           # Per-experiment overrides (hierarchical)
│   │   ├── autoencoder/
│   │   ├── autoregressive/
│   │   ├── classification/
│   │   └── gan/
│   ├── dataset/              # Dataset configs
│   ├── model/                # Model configs
│   ├── loss/                 # Loss configs
│   ├── callback/             # Callback configs
│   ├── optimizer/            # Optimizer configs
│   ├── scheduler/            # LR scheduler configs
│   ├── profiler/             # Profiler presets
│   └── remote_backup/        # Remote backup configs
├── data/wikitext/            # WikiText data + processing script
├── scripts/                  # Helper scripts
├── tests/                    # Test suite
└── docs/                     # Additional documentation
```

## Key Concepts

### Hydra Config System

Configs are composed via `defaults` lists. An experiment config (e.g., `configs/experiment/autoencoder/compression_unet.yaml`) pulls in base configs:

```yaml
defaults:
  - /default          # base hyperparams
  - /dataset: div2k   # dataset config
  - /model: unet_compression
  - /loss: mse
  - /optimizer: adamw
  - /scheduler: cosine_annealing
  - /callback: default
  - _self_            # experiment-specific overrides
```

### Data Container

The trainer builds a unified `dict` per batch with namespaced keys:
- `input/<key>` — inputs from the batch
- `target` — ground truth
- `pred` — model predictions
- `pred/<key>` — dict predictions (e.g. diffusion outputs)
- `post_processed/<key>` — post-processor outputs
- `model` — model reference
- `epoch` — current epoch

Losses, post-processors, and callbacks all read from this container by key.

### Loss System

Losses are wrapped in `LossContainer` / `LossComponent`, each specifying `pred_key` and `target_key` to pull from the data container. Special losses: `PhasedLoss`, `WeightedLoss`, `MovingWeightedLoss`, `MaskedLoss`, `SSIMLoss`, `VGGLoss`, `GradientLoss`, `ShiftedTeacherForcing`.

### Remote Datasets

Two approaches:
1. **Remote filesystem** (`BaseDataset` + `fsspec`): transparently downloads/caches files from SSH/S3/HTTP
2. **Dataset server** (`DatasetServer` + `RemoteDataset`): ZeroMQ server on the data machine, clients fetch processed samples over the network

### Checkpoint Resume

Set `resume_training: true` in the config plus `load_checkpoint_path: "latest"` to auto-discover and resume from the latest matching checkpoint.

## Documentation Convention

- `README.md` files: **features and usage** — what each file does, when to use it
- `ARCHITECTURE.md` files: **implementation details** — classes, methods, design patterns

**When modifying code, always update both `README.md` and `ARCHITECTURE.md` in the relevant directory.**
