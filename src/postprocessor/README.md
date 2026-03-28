# src/postprocessor — Post-Processor Pipeline

Transform model outputs before they reach callbacks and metrics.

## Files

### `pipeline.py` — PostProcessorPipeline + PostProcessorStep

Orchestrates a sequence of post-processing steps. Each step:
1. Reads named inputs from the unified data container via `input_mapping`
2. Calls the processor
3. Writes result back to the data container at `output_key`

```yaml
# In experiment config:
postprocessors:
  _target_: postprocessor.pipeline.PostProcessorPipeline
  steps:
    argmax:
      _target_: postprocessor.pipeline.PostProcessorStep
      processor:
        _target_: postprocessor.argmax.ArgmaxPostProcessor
        dim: 1
      input_mapping:
        "pred": "pred"        # data["pred"] → kwarg "pred"
      output_key: "post_processed/formatted_pred"
```

`unfurl_output: true` merges a dict result directly into the data container (instead of nesting under `output_key`).

### `base.py` — BasePostProcessor

Abstract base class. Subclasses implement `__call__(**kwargs)`.

### `argmax.py` — ArgmaxPostProcessor

Applies `argmax` along `dim` (default: 1, the class dimension). Used for classification to convert logit tensors to class indices for metrics/visualization.

### `channel_formatter.py` — ChannelFormatterPostProcessor

Formats 4-D tensors for visualization:
- 1 channel: squeeze → `[B,H,W]`
- 3 channels: pass through (RGB)
- C > 3 channels: argmax → class index map

### `generate.py` — GeneratePostProcessor

Calls `model.generate(prompt, max_new_tokens=...)` for autoregressive language models. Useful for logging text samples during validation.

```yaml
input_mapping:
  "model": "model"
  "input/input": "input_data"
output_key: "post_processed/generated"
```

### `running_tensor_stats.py` — RunningTensorStatsPostProcessor

Tracks running mean and std over a sliding window of steps. Returns `{"running_mean": float, "running_std": float}`. Useful for monitoring diffusion model output statistics.

### `tensor_mean_std.py` — TensorMeanStdPostProcessor

Computes per-batch mean and std of a tensor.

### `degradation.py` — DegradationPostProcessor

Runs the model N times, feeding each output back as input to measure reconstruction stability under repeated encoding/decoding. Designed for test-time evaluation via `+eval=degradation`.

```yaml
input_mapping:
  "model": "model"
  "pred": "pred"
output_key: "post_processed/degraded_pred"
```

### `tensor_slice.py` — TensorSlicePostProcessor

Slices a tensor along a dimension (e.g., select first few action dimensions).
