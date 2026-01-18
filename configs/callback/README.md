# Callback Configs

Training callback configurations for logging and checkpointing.

## Files

| File | Description |
|------|-------------|
| `default.yaml` | Composes wandb + checkpoint callbacks |
| `wandb.yaml` | Weights & Biases logging callback |
| `checkpoint.yaml` | Model checkpoint saving callback |

## Usage

### In Experiment Configs

```yaml
defaults:
  - /callback: default  # Uses both wandb and checkpoint
```

### Overriding Specific Callbacks

Callbacks use named keys (`wandb`, `checkpoint`) for easy overrides:

```yaml
# Override specific values
callbacks:
  wandb:
    logging_interval: 100
  checkpoint:
    save_every_n_epochs: 5
```

## Callback Components

### WandB Callback (`wandb.yaml`)
- Logs metrics to Weights & Biases
- Logs sample images/predictions
- Configurable logging interval

### Checkpoint Callback (`checkpoint.yaml`)
- Saves model checkpoints
- Supports best/last checkpoint saving
- Configurable save frequency
- Monitors specified metric

## Creating Custom Callback Configs

```yaml
# callback/checkpoint_only.yaml
defaults:
  - checkpoint@callbacks.checkpoint

# callback/frequent_logging.yaml
defaults:
  - wandb@callbacks.wandb
  - checkpoint@callbacks.checkpoint

callbacks:
  wandb:
    logging_interval: 10
```
