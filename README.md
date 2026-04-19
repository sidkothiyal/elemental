# Elemental

A modular PyTorch ML training library supporting image compression, classification, generative models (diffusion, GANs), and autoregressive language models. Configuration is managed by [Hydra](https://hydra.cc/), experiments are tracked with [Weights & Biases](https://wandb.ai/).

## Key Features

- **Multi-task training**: autoencoder, classification, GAN, diffusion, masked autoencoder, autoregressive LM, RL (PPO)
- **ONNX export**: export any trained model or sub-module to ONNX via `src/exporter.py`; supports PyTorch-level submodule extraction and ONNX-level graph slicing (`onnx.utils.extract_model`)
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

# VLA diffusion PPO in SIMPLER (ManiSkill)
uv run python src/trainer.py experiment/vla/vla_rl_ppo

# Override any config value on the command line (Hydra syntax)
uv run python src/trainer.py experiment/autoencoder/compression_unet lr=0.001 num_epochs=500

# Distributed training (2 GPUs)
torchrun --nproc_per_node=2 src/trainer.py experiment/autoencoder/compression_unet distributed=true
```

### Graph-based trainer

```bash
uv run python src/trainer.py experiment/node/node_train_autoencoder   # mode: graph dispatches to GraphTrainer
```

## Exporting Models

Trained models can be exported to ONNX via `src/exporter.py`. The config can be a full
experiment config, a model-only config, or a dedicated export config in `configs/export/`.

```bash
# Dedicated export config (recommended — fine-grained sub-module control)
uv run python src/exporter.py export/vq_vae_transformer \
    --checkpoint checkpoints/vqvae.pt \
    --output-dir exports/vqvae/

# Model-only config (no dataset / optimizer / loss needed)
uv run python src/exporter.py model/vq_vae_transformer \
    --checkpoint checkpoints/vqvae.pt \
    --output-dir exports/vqvae/

# Full experiment config with Hydra override
uv run python src/exporter.py experiment/autoencoder/compression_ae \
    --checkpoint results/autoencoder/run/best.pt \
    --output-dir exports/ \
    export.opset_version=18
```

Export configs (`configs/export/`) declare which sub-modules to export and how:

```yaml
export:
  opset_version: 17
  dynamic_axes: true
  modules:
    # PyTorch-level: extract submodule by dotted path before ONNX export
    - name: "encoder"
      input_layer: "model.encoder"
      onnx_filename: "encoder.onnx"
      inputs:
        - {name: "input", shape: [1, 3, 640, 960], dtype: "float32"}
      outputs:
        - {name: "latents"}

    # ONNX-level: slice an already-exported graph via onnx.utils.extract_model
    - name: "decoder_onnx"
      source_onnx: "full_model.onnx"
      input_tensor: "/quantizer/Gemm_output_0"
      output_tensor: "reconstructed_image"
      onnx_filename: "decoder_extracted.onnx"
```

See `configs/export/vq_vae_transformer.yaml` for a complete example.

## Directory Structure

```
elemental/
├── src/
│   ├── trainer.py            # Main training entry point
│   ├── exporter.py           # ONNX model export entry point
│   ├── graph_trainer.py      # Node-graph-based training
│   ├── callback/             # Training callbacks (checkpoint, metrics, W&B)
│   ├── data_processor/       # Dataset downloading, tokenization
│   ├── dataset/              # Dataset implementations
│   ├── env/                  # Vectorized RL environments (BaseVecEnv, DummyVecEnv, SimplerEnv)
│   ├── loss/                 # Loss functions (MSE, SSIM, VGG, phased, PPO / value / entropy, etc.)
│   ├── metrics/              # TorchMetrics wrappers (IoU, MSE, PSNR, SSIM, episode metrics)
│   ├── model/                # Model architectures
│   │   └── blocks/           # Building blocks (conv, resnet, linear, etc.)
│   ├── nodes/                # Computation graph nodes (graph_trainer)
│   ├── postprocessor/        # Output post-processing pipeline
│   ├── rl/                   # RL algorithms, policy wrappers, rollout buffer, GAE
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
│   ├── export/               # ONNX export configs (default.yaml + per-model)
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
