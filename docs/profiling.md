# Profiling Guide

Elemental integrates PyTorch's built-in profiler via Hydra-composable config presets. Profiling is **disabled by default** and can be enabled per-experiment with zero code changes.

## Quick Start

```bash
# Add profiling to any experiment via command line override
python src/trainer.py experiment/autoencoder/compression_unet +profiler=standard

# Or use a specific level
python src/trainer.py experiment/autoencoder/compression_unet +profiler=memory
```

Or add it permanently to an experiment config:

```yaml
# experiment/my_experiment.yaml
defaults:
  - /default
  - /profiler: standard    # ← add this line
  - /dataset: div2k
  - /model: unet_compression
  - ...
```

## Profiling Levels

| Preset | Activities | Overhead | Best For |
|--------|-----------|----------|----------|
| `disabled` | — | 0% | Default (no profiling) |
| `cpu_only` | CPU | ~1–3% | CI/nightly runs, quick op timing |
| `standard` | CPU + CUDA | ~5–15% | Standard GPU bottleneck analysis |
| `debug` | CPU + CUDA + stack + flops | ~15–30% | Mapping ops to source lines |
| `memory` | CPU + CUDA + memory + stack | ~10–20% | OOM debugging, memory leak hunting |
| `full` | All knobs enabled | High | Pre-release performance audit |

### Preset Details

#### `cpu_only`
Records all CPU ops (Python calls, ATen ops, autograd). Very low overhead, always safe to enable in profiling windows.

#### `standard`
Adds CUDA activity recording (GPU kernel launches, memcpy, sync events via CUPTI). Enables `record_shapes` to capture input tensor shapes per op. Essential for GPU bottleneck analysis.

#### `debug`
Adds `with_stack=True` to capture Python call stacks for each op (maps low-level ops back to your source lines) and `with_flops=True` to estimate FLOPs for matmul/conv ops.

#### `memory`
Enables `profile_memory=True` to track tensor allocations/deallocations on CPU and CUDA. Combined with `with_stack=True` to identify allocation source lines. Useful for OOM debugging and memory leak hunting.

#### `full`
All knobs enabled: shapes, memory, stack, and FLOPs. Use for a comprehensive single-run performance audit. **Do not use in production training.**

## Configuration Reference

All presets share this configuration structure:

```yaml
profiler:
  enabled: true/false          # Master switch
  activities:                  # What to trace
    - CPU                      # CPU ops (~1-3% overhead)
    - CUDA                     # GPU kernels (~5-15% overhead)
  step_per: "batch"            # When to advance profiler: "batch" or "epoch"
                               #   "batch" — profiler.step() fires after each batch (default)
                               #   "epoch" — profiler.step() fires after each epoch, capturing
                               #             inter-epoch overhead (cleanup, validation, scheduler)
  schedule:
    wait: 1                    # Steps to skip (JIT warmup noise)
    warmup: 1                  # Steps to trace but discard
    active: 3                  # Steps to actually record
    repeat: 1                  # How many cycles before stopping
  record_shapes: false         # Capture input tensor shapes (+2-5%)
  profile_memory: false        # Track allocations (+5-15%)
  with_stack: false            # Python call stacks (+10-20%)
  with_flops: false            # Estimate matmul/conv FLOPs (minimal)
  output:
    print_table: true          # Print summary table to terminal
    table_sort_by: "cpu_time_total"  # Sort column for table
    table_row_limit: 20        # Number of rows to show
    export_chrome_trace: true  # Export chrome://tracing JSON (ignored when tensorboard: true,
                               #   as TensorBoard's handler handles the export internally)
    tensorboard: false         # Write TensorBoard traces
```

### Schedule Explained

The schedule controls *when* profiling is active during training steps:

```
Step: 1    2    3    4    5    6    7    8    9    10   ...
      wait warmup ──active──  wait warmup ──active──  (idle)
      skip discard  RECORD    skip discard  RECORD
```

With `wait=1, warmup=1, active=3, repeat=2`:
- **Step 1**: Skipped (wait) — avoids JIT compilation noise
- **Step 2**: Traced but discarded (warmup) — caches are warm
- **Steps 3–5**: Actually recorded ✓
- **Steps 6–10**: Second cycle repeats the pattern
- **Step 11+**: Profiler is idle (no overhead)

Only `active × repeat` steps are actually recorded, minimizing the overall impact.

### What Gets Labeled

The standard (non-GAN) training loop annotates these phases with `record_function`:

| Label | What's Inside |
|-------|--------------|
| `forward` | Model forward pass |
| `loss` | Loss computation |
| `backward` | Gradient computation |
| `optimizer` | Gradient clipping + optimizer step + zero_grad |

These appear as named regions in TensorBoard flame graphs and chrome traces.

> **Note:** GAN mode (`mode: "gan"`) does not emit `record_function` labels — the GAN batch loop runs without named regions.

## Viewing Results

Profiler outputs are saved to `{results_dir}/profiler/`.

### Terminal Table

Printed automatically after each profiling cycle:

```
-------  ------  ------  ------  ------  ------  ------
Name     CPU %   CPU t   CUDA %  CUDA t  # Calls Memory
-------  ------  ------  ------  ------  ------  ------
forward   45.2%  120ms    62.3%  180ms       3    512MB
backward  38.1%  101ms    30.5%  88ms        3    256MB
loss       8.3%   22ms     5.1%  15ms        3     24MB
optimizer  7.4%   20ms     2.1%   6ms        3      0B
-------  ------  ------  ------  ------  ------  ------
```

### Chrome Trace

Open `trace_step_N.json` files at `chrome://tracing` (or `edge://tracing`) for a timeline view of all ops.

### TensorBoard

For presets with `tensorboard: true`:

```bash
pip install torch-tb-profiler
tensorboard --logdir={results_dir}/profiler/tensorboard
```

The TensorBoard plugin provides:
- **Flame graphs** — hierarchical view of time spent in each op
- **Memory timeline** — allocation/deallocation events over time
- **GPU utilization** — kernel execution vs. idle time
- **Op-level breakdown** — sorted by time, memory, or call count

## Important Gotchas

### `profile_memory` + `torch.compile` / CUDA Graphs

> [!CAUTION]
> If you enable `profile_memory=True` alongside CUDA graphs or `torch.compile`, the profiler's allocation hooks can break graph capture. Profile without these features or use separate runs.

### `record_function` Labels Are Free

The `record_function` labels in the training loop (forward, loss, backward, optimizer) are essentially **no-ops** when no profiler is active. They have near-zero cost and are always present.

### Profiler Only Runs During Training

The profiler is only active during the training phase. Validation and test phases run without profiling to avoid interference with timing measurements.

### Schedule vs. Always-On

**Never** keep full profiling enabled in production training. The schedule pattern is designed so you profile a small window and then stop. If you need continuous monitoring, use `record_function` labels with external tools instead.

## Recommended Configurations

| Scenario | Preset | Notes |
|----------|--------|-------|
| CI/nightly perf regression | `cpu_only` | Minimal overhead, catches major regressions |
| "Why is this slow?" | `standard` | Start here for GPU bottleneck analysis |
| "Where in my code is this op?" | `debug` | Stack traces map ops to source lines |
| OOM / memory leak | `memory` | Shows allocation sources and sizes |
| Pre-release audit | `full` | Comprehensive single-run analysis |
| Production training | `disabled` | **Always** — profiler OFF |
