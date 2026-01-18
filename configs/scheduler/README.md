# Learning Rate Scheduler Configs

Learning rate scheduler configurations.

## Available Schedulers

| File | Description | Best For |
|------|-------------|----------|
| `step_lr.yaml` | Step decay | Standard training |
| `cosine_annealing.yaml` | Cosine annealing | Long training runs |

## Usage

### In Experiment Configs

```yaml
defaults:
  - /scheduler: cosine_annealing
```

### Overriding Scheduler Parameters

```yaml
# Override in experiment config
lr_scheduler_params:
  T_max: 1000  # For cosine annealing
  # or
  step_size: 50  # For step LR
  gamma: 0.5
```

## Config Structure

```yaml
# scheduler/cosine_annealing.yaml
lr_scheduler: "CosineAnnealingLR"
lr_scheduler_params:
  T_max: ${num_epochs}  # Interpolates from experiment
```

## How It Works

The trainer uses these strings with `getattr`:

```python
scheduler_class = getattr(torch.optim.lr_scheduler, config.get("lr_scheduler", "CosineAnnealingLR"))
scheduler = scheduler_class(optimizer, **config.lr_scheduler_params)
```

## Adding Custom Schedulers

For standard PyTorch schedulers:

```yaml
# scheduler/exponential.yaml
lr_scheduler: "ExponentialLR"
lr_scheduler_params:
  gamma: 0.95
```

```yaml
# scheduler/plateau.yaml
lr_scheduler: "ReduceLROnPlateau"
lr_scheduler_params:
  mode: "min"
  factor: 0.5
  patience: 10
```
