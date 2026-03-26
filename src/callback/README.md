# src/callback — Training Callbacks

Training lifecycle hooks that monitor, log, and checkpoint training runs.

## Files

### `callback.py` — Base classes

Defines the `Callback` interface and `CallbackList` orchestrator.

**When to use**: Subclass `Callback` when adding a new monitoring or side-effect to training. Compose callbacks in a `CallbackList` via Hydra config.

### `checkpoint_callback.py` — Checkpoint saving

Saves model, optimizer, scheduler, and loss state to disk.

**Features**:
- Save best model based on any monitored metric (`monitor_metric`, `mode: "min"/"max"`)
- Save last model at end of training
- Periodic saves every N epochs or N steps
- Rolling window (delete old periodic checkpoints beyond `max_checkpoints_saved`)
- Remote backup upload (SSH or W&B) after each save

**When to use**: Include in `callbacks` config for all training runs.

```yaml
# configs/callback/checkpoint.yaml
_target_: callback.checkpoint_callback.CheckpointCallback
save_dir: ${results_dir}/${experiment_name}
save_every_n_epochs: 10
save_best: true
monitor_metric: loss_total
mode: min
max_checkpoints_saved: 3
```

### `metrics_callback.py` — TorchMetrics integration

Computes metrics (IoU, PSNR, SSIM, accuracy, etc.) using [TorchMetrics](https://lightning.ai/docs/torchmetrics/).

**Features**:
- Per-phase metric tracking (train/val/test have separate metric instances)
- Configurable compute interval per phase
- Configurable `pred_key` and `target_key` for data-container access
- Optional `transform_pred` / `transform_target` callables
- Injects computed metrics into `log_metrics()` so they reach `WandbCallback`

**When to use**: Add when you need quantitative metrics beyond loss values.

```yaml
_target_: callback.metrics_callback.MetricsCallback
metrics:
  psnr:
    _target_: torchmetrics.image.PeakSignalNoiseRatio
phases:
  train: 10   # compute every 10 epochs
  val: 1      # compute every epoch
pred_key: post_processed/formatted_pred
target_key: target
```

### `wandb_callback.py` — Weights & Biases logging

Logs scalars, images, text, and 3D point clouds to W&B.

**Features**:
- Logs visual data at configurable `logging_interval` steps
- Logs scalar metrics each epoch
- Supports classification mode (shows target/pred labels in image captions)
- Text decoding via tokenizer for language model runs
- 3D point cloud / line logging via Plotly
- Selective logging via `to_log` list (logs all keys by default)
- `field_loggers` dict to override per-key logging method
- Auto-resume: saves run ID to file, resumes W&B run on `resume_training: true`

**When to use**: Include in all runs. Configure `log_text_data: true` for language models.

```yaml
_target_: callback.wandb_callback.WandbCallback
exp_name: ${experiment_name}
project: elemental
entity: elemental-ai
logging_interval: 100
```
