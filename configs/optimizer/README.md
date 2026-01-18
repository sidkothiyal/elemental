# Optimizer Configs

Optimizer configurations for training.

## Available Optimizers

| File | Description | Best For |
|------|-------------|----------|
| `adam.yaml` | Adam optimizer | General purpose |
| `adamw.yaml` | AdamW (Adam with decoupled weight decay) | Transformers, modern networks |
| `sgd.yaml` | Stochastic Gradient Descent | ConvNets, fine-tuning |

## Usage

### In Experiment Configs

```yaml
defaults:
  - /optimizer: adamw
```

### How It Works

These configs set the `optimizer` string, which the trainer uses with `getattr`:

```python
optimizer_class = getattr(torch.optim, config.get("optimizer", "AdamW"))
```

## Config Structure

```yaml
# optimizer/adamw.yaml
optimizer: "AdamW"
```

The learning rate is controlled by the top-level `lr` parameter in experiment configs.

## Adding Custom Optimizers

For standard PyTorch optimizers, just create a config with the class name:

```yaml
# optimizer/rmsprop.yaml
optimizer: "RMSprop"
```

For custom optimizers, you may need to modify the trainer to handle instantiation.
