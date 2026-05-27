# configs — Hydra Configuration

All training configurations are managed by Hydra. Configs are modular and composable.

## Structure

```
configs/
├── config.yaml              # Main entry point, inherits default
├── default.yaml             # Base hyperparams shared by all experiments
├── experiment/              # Per-experiment overrides
│   ├── autoencoder/         # Image compression / reconstruction experiments
│   ├── autoregressive/      # Language model experiments
│   ├── classification/      # Image classification experiments
│   └── gan/                 # GAN experiments
├── dataset/                 # Dataset configurations
├── model/                   # Model configurations
├── loss/                    # Loss configurations
├── callback/                # Callback configurations
├── optimizer/               # Optimizer configurations (adam, adamw, sgd)
├── scheduler/               # LR scheduler configurations
├── profiler/                # PyTorch profiler presets
├── remote_backup/           # Remote backup strategy configs
├── eval/                    # Evaluation overlays (e.g., degradation metrics)
├── legacy/                  # Old flat configs (kept for reference)
└── checkpoint_surgery/      # Checkpoint rename/migration configs
```

## How Configs Compose

An experiment config composes sub-configs via `defaults`:

```yaml
# configs/experiment/autoencoder/compression_unet.yaml
defaults:
  - /default                 # base hyperparams (lr, num_epochs, etc.)
  - /dataset: div2k          # pulls in configs/dataset/div2k.yaml
  - /model: unet_compression
  - /loss: mse
  - /optimizer: adamw
  - /scheduler: cosine_annealing
  - /callback: default
  - /torch_compile
  - /remote_backup: wandb
  - _self_                   # experiment-level overrides at end
```

## Key Default Values (default.yaml)

| Key | Default | Description |
|-----|---------|-------------|
| `lr` | 0.0001 | Learning rate |
| `num_epochs` | 100 | Training epochs |
| `batch_size` | 64 | Batch size |
| `seed` | 42 | Random seed |
| `device` | `"gpu"` | Training device |
| `dtype` | null (→ bfloat16) | Precision |
| `validation_interval` | 5 | Run val every N epochs |
| `gradient_accumulation_steps` | 1 | Grad accum steps |
| `optimizer` | `"AdamW"` | Optimizer class name |
| `lr_scheduler` | `"CosineAnnealingLR"` | LR scheduler |
| `channels_last` | true | NHWC memory format |
| `cudnn_benchmark` | true | Auto-tune conv algorithms |
| `resume_training` | false | Resume from checkpoint |

## Experiment Configs

See `configs/experiment/README.md` (each subdirectory has its own README).

## Adding a New Experiment

1. Create `configs/experiment/<task>/my_experiment.yaml`
2. Declare `defaults` list pointing to existing sub-configs
3. Override specific values below `_self_`
4. Set `experiment_name` and `results_dir`

## Profiler Presets

| Config | Purpose |
|--------|---------|
| `disabled` | No profiling |
| `standard` | CPU+CUDA, chrome trace |
| `memory` | Memory profiling |
| `full` | All activities + shapes + stack |
| `cpu_only` | CPU-only profiling |
| `debug` | Minimal (dev use) |

## Evaluation Overlays (`eval/`)

Composable configs that add test-time evaluation pipelines via a single Hydra override.

| Config | Purpose | Usage |
|--------|---------|-------|
| `degradation` | Run model N times, compute metrics on degraded output | `+eval=degradation` |

```bash
# Degradation evaluation (100 iterations, default)
uv run python src/trainer.py experiment/autoencoder/compression_ae test=true +eval=degradation

# Custom iteration count
uv run python src/trainer.py experiment/autoencoder/compression_ae test=true +eval=degradation \
  postprocessors.steps.degradation.processor.num_iterations=50
```
