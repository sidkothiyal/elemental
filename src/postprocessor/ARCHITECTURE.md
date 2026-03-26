# src/postprocessor — Architecture

## PostProcessorPipeline (pipeline.py)

```python
class PostProcessorPipeline:
    steps: list[PostProcessorStep]  # or dict (Hydra-instantiated)

    def __call__(self, data: dict) -> None:
        for step in steps:
            step(data)   # modifies data in-place
```

### PostProcessorStep.__call__(data)

1. Build `kwargs` from `input_mapping`:
   - Special key `"input"` collects all `input/*` entries into a sub-dict
   - Other keys: `kwargs[kwarg_name] = data[data_path]`
2. Call `processor(**kwargs)` → `result`
3. If `unfurl_output=True` and `result` is dict: `data.update(result)`
4. Else: `data[output_key] = result`

`extract_value(source, key)` handles both dict and attribute access.

## BasePostProcessor (base.py)

Abstract with `@abstractmethod __call__(**kwargs)`. Return value stored by the pipeline step.

## ArgmaxPostProcessor

`pred.argmax(dim=self.dim)` — only applied when `pred.dim() > 1`.

## ChannelFormatterPostProcessor

Branching:
- `shape[1] == 1`: `squeeze(1)`
- `shape[1] == 3`: pass through
- `shape[1] > 3`: `argmax(dim=1, keepdim=True)`
- non-4D: pass through

## GeneratePostProcessor

Calls `model.generate(input_data[:, :prompt_length], max_new_tokens=...)`. Takes first `prompt_length` tokens as seed (default 1).

## RunningTensorStatsPostProcessor

Maintains two `deque(maxlen=window_size)` for means and stds. Each call: computes current `mean/std`, appends, returns `{"running_mean": mean_of_deque, "running_std": mean_of_deque}`. NaN/Inf stds replaced with 0.
