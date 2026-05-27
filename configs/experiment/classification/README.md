# Classification Experiments

Image classification experiments.

## Available Experiments

| Config | Description | Dataset | Model |
|--------|-------------|---------|-------|
| `imagenet.yaml` | CNN classifier | TinyImageNet | ConvBlock + Linear |

## Usage

```bash
python src/trainer.py experiment/classification/imagenet
```

## Model Architecture

The default classifier uses:
- 6-layer ConvBlock encoder (3→32→64→128→256→512→1024)
- Flatten layer
- LinearBlock head with batch norm

## Common Overrides

```bash
# Adjust learning rate and batch size
python src/trainer.py experiment/classification/imagenet lr=0.0005 batch_size=256

# Change scheduler
python src/trainer.py experiment/classification/imagenet lr_scheduler=CosineAnnealingLR
```

## Adding Custom Classification Experiments

1. Create a new model config in `model/` if needed
2. Create `experiment/classification/your_experiment.yaml`:

```yaml
# @package _global_
defaults:
  - /default
  - /dataset: imagenet  # or your dataset
  - /model: your_model
  - /loss: cross_entropy
  - /optimizer: adam
  - /scheduler: step_lr
  - /callback: default
  - _self_

experiment_name: your_classification_${now:%Y-%m-%d}_${now:%H-%M-%S}
results_dir: results/classification/
mode: "classification"
```
