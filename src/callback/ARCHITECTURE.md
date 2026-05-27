# src/callback — Architecture

## Callback (callback.py)

Abstract base with no-op implementations. All hooks receive lightweight arguments.

```python
class Callback:
    def update_config(self, config: dict): ...
    def on_phase_start(self, phase: str): ...
    def on_phase_end(self, phase: str): ...
    def on_epoch_start(self, epoch: int, phase: str): ...
    def on_epoch_end(self, epoch: int, phase: str): ...
    def on_batch_start(self, step: int, phase: str): ...
    def on_batch_end(self, step: int, phase: str, data: dict): ...
    def on_lr_update(self, epoch: int, lr: float, scheduler_type: str = ""): ...
    def log_metrics(self, step: int, phase: str, metrics: dict[str, float]): ...
```

`CallbackList` uses `__getattr__` magic to dispatch any method call to all registered callbacks, silently skipping those that don't implement it.

## CheckpointCallback (checkpoint_callback.py)

### State

| Attribute | Purpose |
|-----------|---------|
| `best_metric` | Best monitored value seen; compared each val epoch |
| `periodic_checkpoints` | `deque` of saved periodic checkpoint paths |
| `model/optimizer/scheduler/criterion` | Set via `update_config()` from trainer |
| `remote_backup` | `RemoteBackupStrategy` instance (SSH or W&B) |

### Key Methods

- `_save_checkpoint(name, epoch, step, is_periodic)` — saves via `ckpt_utils.save_checkpoint()`, manages rolling window, triggers remote upload
- `on_epoch_start(epoch, "train")` — saves periodic epoch checkpoint (skips epoch 0)
- `on_batch_end(step, "train", data)` — saves periodic step checkpoint
- `log_metrics(step, "val", metrics)` — checks if monitored metric improved, saves "best"
- `on_phase_end("train")` — saves "last" checkpoint

Best/last checkpoints are never counted towards `max_checkpoints_saved` (only periodic ones are rotated).

## MetricsCallback (metrics_callback.py)

### Design

One `MetricCollection` clone per phase — they maintain independent internal state (accumulators). Cloned with `prefix=f"{phase}/"` so metric names are prefixed automatically (e.g. `"val/PSNR"`).

### Key Methods

- `on_epoch_start` — calls `reset()` on phase metrics if computing this epoch
- `on_batch_end` — calls `update(preds, targets)` (with optional transforms)
- `on_epoch_end` — calls `compute()`, stores scalar results in `self.last_computed`
- `log_metrics` — injects `last_computed` into the metrics dict (picked up by `WandbCallback`)

Compute frequency: epoch 0 always, then every `phases[phase]` epochs, plus always on the last epoch.

## transforms.py

Pure-tensor `nn.Module` adapters that bridge model output layouts to
metric input expectations. They are kept here (rather than in `src/utils/`)
so callback configs can reference them by a short
`_target_: callback.transforms.<Name>` path.

`FlattenTemporal` collapses the leading two dims of a 5-D tensor into one.
Inputs of rank 3 or 4 are returned unchanged so the same callback config
also works on image batches; rank 2 raises. The callback chain then sees
`(B*T, C, H, W)` and reuses the existing 4-D `BaseMetric` implementations
(PSNR / SSIM / MSE) verbatim — per-frame values are accumulated and the
returned mean is the per-frame average over the epoch.

## TrialSummaryCallback (trial_summary_callback.py)

Opt-in callback that writes one JSON file per run for machine-readable result consumption (autoresearch agent, CI checks).

### Finalization protocol

The standard `Callback` interface has no `finalize` hook. `TrialSummaryCallback` adds one:

```python
def finalize(self, exc: BaseException | None = None) -> None: ...
```

`Trainer._finalize(exc)` iterates `self.callbacks.callbacks` and calls `cb.finalize(exc)` on any callback that defines it. The `exc` value is `None` on clean exit or the raised exception otherwise — this is how status (`ok`, `oom`, `compile_error`, etc.) is determined.

### Timing strategy

- `on_batch_start` records `time.perf_counter()` for training-phase steps only.
- `on_batch_end` computes elapsed ms; step 0 (lazy compile trace) is recorded separately as `compile_warmup_ms`; steps 1+ feed a `deque(maxlen=500)` rolling window.
- No `torch.cuda.synchronize()` — CPU-side timing is sufficient for ranking compile modes; it slightly underestimates GPU time but has zero steady-state overhead.
- When the callback is not registered, `on_batch_start/end` are never called — `CallbackList.__getattr__` skips absent implementations.

### classify_exception helper

`classify_exception(exc)` in this module is also imported by `src/trainer.py` to map exceptions to subprocess exit codes (0, 10–13). Both consumers share the same logic.

## WandbCallback (wandb_callback.py)

### Initialization Flow

1. `_build_wandb_init_kwargs()` — optionally finds previous run ID for resume
2. `wandb.init(...)` — creates/resumes W&B run
3. `_save_wandb_run_id()` — writes run ID to `<results_dir>/<exp_name>/.wandb_run_id`

### on_batch_end Dispatch Chain

For each key in `_to_log` (or default keys):
1. `_dispatch_field_logger()` — custom logger for point3d / line3d
2. `_log_dict_scalars()` — dict of scalars → wandb.log()
3. `_log_text()` — tokenizer.decode → wandb table (if log_text_data)
4. `_log_image()` — auto-routes 5-D tensors to `_log_video()`; otherwise numpy conversion → wandb.Image

### Image Handling

Tensors are moved to numpy via `torch_np_utils.to_numpy()`, transposed NCHW→NHWC, first sample extracted. Multi-channel class-index outputs (when `num_classes` is set) are converted to RGB using `_class_colors`. Natural-image branches map `image_value_range` → `[0, 1]` float via `_denormalize_to_unit_float()` (no-op when range is `(0, 1)`).

### Video Handling

5-D `(B, T, C, H, W)` tensors are auto-detected at the top of `_log_image()` and forwarded to `_log_video()`. The first `max_videos_per_log` clips are denormalised from `video_value_range` (e.g. `(-1, 1)` for BVI-AOM) to `uint8 [0, 255]` via `_denormalize_to_uint8()`, then passed to `wandb.Video(arr, fps=video_fps)` in NTCHW layout. bf16 inputs are upcast to fp32 before the numpy hop.

### Unsupported Shapes

Anything that lands in `_log_image()` with `ndim ∉ {2, 3, 4, 5}` triggers `_warn_unsupported_shape(tag)` — one log line per key per run, then a silent skip. Replaces the older silent-drop behaviour.

### 3D Logging

`_log_point3d()` and `_log_line3d()` use Plotly `Scatter3d` wrapped in `wandb.Plotly`. Input shape must be `(N, 3)` or `(B, N, 3)`.

### Cross-references

- Reads `WANDB_RUN_ID_FILENAME` from `utils/wandb_utils.py`
- Uses `data_processor.tokenizer.Tokenizer` for text decode
- Uses `utils.torch_np_utils.to_numpy()` for tensor conversion
