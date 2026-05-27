# Experiment Configs

Complete experiment configurations that compose all components.

## Subdirectories

| Directory | Description |
|-----------|-------------|
| `autoencoder/` | Image autoencoder and compression experiments |
| `autoregressive/` | Language model experiments |
| `classification/` | Image classification experiments |
| `gan/` | Generative adversarial network experiments |

## Usage

```bash
# Run an experiment
python src/trainer.py experiment/autoencoder/compression_unet

# Override parameters
python src/trainer.py experiment/autoencoder/compression_unet lr=0.001 batch_size=8
```

## Experiment Config Structure

Each experiment config uses `# @package _global_` to merge into the root namespace:

```yaml
# @package _global_
# Description of the experiment

defaults:
  - /default                    # Base training params
  - /dataset: div2k             # Dataset
  - /model: unet_compression    # Model
  - /loss: mse                  # Loss function
  - /optimizer: adamw           # Optimizer
  - /scheduler: cosine_annealing # Scheduler
  - /callback: default          # Callbacks
  - _self_

experiment_name: my_experiment_${now:%Y-%m-%d}_${now:%H-%M-%S}
results_dir: results/autoencoder/
mode: "autoencoder"

# Training overrides
lr: 0.0003
num_epochs: 200
batch_size: 4
```

## Creating New Experiments

1. Choose the appropriate subdirectory (or create new)
2. Create `experiment/category/your_experiment.yaml`
3. Add `# @package _global_` at the top
4. Define defaults to compose from
5. Override specific parameters as needed
