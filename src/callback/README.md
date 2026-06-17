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
- Persists resume bookkeeping: `global_optim_step` (the true optimizer-step count, for a correct LR-schedule resume under gradient accumulation) and, when a `StatefulDataLoader` train loader is wired, its `dataloader_state` (streaming position). Both are restored by `Trainer` on resume.

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

**Ordering**: `MetricsCallback` mutates the shared metrics dict, and `WandbCallback.log_metrics` commits immediately — so the metrics callback must come **before** `WandbCallback` in the broadcast. Place it ahead of `/callback: default` in the experiment `defaults` list (e.g. `- /callback/lm_eval_metrics@callbacks.metrics` before `- /callback: default`). See `configs/callback/lm_eval_metrics.yaml` (next-token accuracy + bits-per-byte for the LM stages).

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

### `transforms.py` — Tensor transforms for callback configs

Small `nn.Module` callables for the `transform_pred` / `transform_target`
slots on `MetricsCallback`. Use these when the data layout produced by the
model does not match what an existing metric expects.

- `FlattenTemporal` — folds the temporal axis of a video tensor into the
  batch dimension: `(B, T, C, H, W) -> (B*T, C, H, W)`. 4-D and 3-D tensors
  pass through unchanged. Used by
  [`configs/callback/video_reconstruction_metrics.yaml`](../../configs/callback/video_reconstruction_metrics.yaml)
  to apply the existing per-image PSNR / SSIM / MSE metrics to video
  predictions and targets, averaging per-frame across all frames in all
  clips of an epoch.

```yaml
# configs/callback/video_reconstruction_metrics.yaml
_target_: callback.metrics_callback.MetricsCallback
metrics:
  psnr: {_target_: metrics.psnr.PSNR, data_range: 2.0}
transform_pred:   {_target_: callback.transforms.FlattenTemporal}
transform_target: {_target_: callback.transforms.FlattenTemporal}
```

### `trial_summary_callback.py` — Autoresearch trial output

Writes `trial_summary.json` at run end for autoresearch-style hyperparameter and `torch.compile` search. **Opt-in** — not included in default training, so normal runs have zero overhead.

**Features**:
- Tracks final and best-ever values for every metric that passes through `log_metrics`
- Times each training step: step 0 captures compile-warmup latency, subsequent steps feed a rolling median
- Records peak GPU memory via `torch.cuda.max_memory_allocated()`
- Classifies failures (OOM, compile error, NaN, timeout) and records status in JSON

**Output schema**:
```json
{
  "status": "ok|oom|compile_error|nan|timeout|error",
  "final":  {"val/loss_total": ..., "val/PSNR": ...},
  "best":   {"val/PSNR": {"value": ..., "epoch": ...}},
  "perf":   {"steps": N, "epochs": E, "compile_warmup_ms": ...,
             "median_step_ms": ..., "peak_mem_gb": ...},
  "config_overrides": [...], "git_sha": "...", "wandb_run_id": "..."
}
```

**When to use**: Add to any trial run launched by the autoresearch agent. Pass via override:

```bash
python src/trainer.py experiment/autoencoder/compression_ae \
  '+callbacks.trial_summary._target_=callback.trial_summary_callback.TrialSummaryCallback' \
  max_steps=200 --no-remote-logging
```

### `wandb_callback.py` — Weights & Biases logging

Logs scalars, images, video clips, text, and 3D point clouds to W&B.

**Features**:
- Logs visual data at configurable `logging_interval` steps
- Logs scalar metrics each epoch
- Supports classification mode (shows target/pred labels in image captions)
- Text decoding via tokenizer for language model runs
- 3D point cloud / line logging via Plotly
- Auto-routes 5-D `(B, T, C, H, W)` tensors to `wandb.Video` (denormalises
  `video_value_range` → uint8 `[0, 255]`, configurable `video_fps`)
- Configurable `image_value_range` for images normalised outside `[0, 1]`
- Per-key modality via the top-level `to_log` (see below)
- Token tensors decoded to an accumulating `wandb.Table` (`{phase}/text_samples`)
- Warns once per key (and refuses oversized 2-D tensors such as `(seq_len, vocab)`
  logits) instead of logging a junk image
- Auto-resume: saves run ID to file, resumes W&B run on `resume_training: true`

**Choosing what to log (`to_log`)**: the top-level `to_log` controls *which*
data-container keys are logged and *how*:
- `null` (default) → `input/input`, `target`, `pred` as images.
- `list[str]` → each listed key as an image (legacy).
- `dict[str, str]` → explicit `{key: modality}`, where modality is one of
  `text | image | video | point3d | line3d | scalars`.

For `text`, set `tokenizer` (path/name) and optionally `tokenizer_type`
(`bpe`/`char`/`hf`; `bpe`/`char` inferred from a `.json` extension when unset,
`hf` — a transformers `AutoTokenizer`, e.g. `"gpt2"` — must be set explicitly).
Decoded samples are written one fresh `wandb.Table` per interval under
`{phase}/text_samples` (stepped media, no accumulation). Example
(autoregressive LM):

```yaml
to_log:
  input/input: text
  target: text
  pred: text       # logits → argmax → decode
callbacks:
  wandb:
    tokenizer: ${tokenizer_path}
    tokenizer_type: bpe
```

(`log_text_data` and `field_loggers` are deprecated aliases, still folded into
`to_log` for back-compat.)

**When to use**: Include in all runs. For video AEs the defaults
(`video_value_range: [-1, 1]`, `video_fps: 8`) match BVI-AOM and
`clip_length: 8`, so no overrides are required.

```yaml
_target_: callback.wandb_callback.WandbCallback
exp_name: ${experiment_name}
project: elemental
entity: elemental-ai
logging_interval: 100
```
