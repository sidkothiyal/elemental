# src/utils — Shared Utilities

Cross-cutting helpers used by trainer, callbacks, and tools.

## Files

### `initializer.py` — Experiment Initialization

`CUDAPrefetcher` — wraps a `DataLoader` and prefetches batches to GPU using a side CUDA stream. Overlaps HtoD transfers with compute, significantly reducing GPU idle time. Behaves like a DataLoader iterator.

Key functions:
- `initialize_device(device_str)` — returns `torch.device`, handles `"gpu"` (CUDA/MPS), CPU
- `initialize_dtype(dtype_str)` — returns `torch.dtype`, falls back to float32 if bfloat16 unsupported
- `set_seeds(seed)` — seeds PyTorch, CUDA, NumPy, Python random
- `initialize_dataset(dataloader_config, ...)` — instantiates dataset + dataloader from Hydra config, wraps in `CUDAPrefetcher`, sets up `DistributedSampler` if needed; returns a `{"train", "val", "test"}` dict
- `initialize_callbacks(config)` — instantiates callbacks and returns a `CallbackList` seeded with `lr`/`num_epochs`/`seed`
- `initialize_remote_backup(config, results_dir)` — instantiates remote backup strategy
- `get_num_parameters(model)` — returns `(total, trainable, non_trainable)` counts
- `compile_model_and_optimizer_step(model, optim, compile_config)` — applies `torch.compile` if configured
- `load_from_checkpoint(...)` — resolves `"latest"` checkpoint path, loads state

### `checkpoint.py` — Checkpoint I/O

`save_checkpoint(path, model, optimizer, scheduler, criterion, epoch, step)` — saves all components to a single `.pt` file.

`load_checkpoint(path, model, optimizer, scheduler, criterion, ...)` — loads state back. Supports both standard format (with `model_state_dict` key) and raw state dict files.

`find_latest_checkpoint(results_dir, experiment_name)` — scans results directory for runs matching the experiment base name, returns path to most recent valid checkpoint. Preference order: `last.pt` → `best.pt` → `checkpoint-step-N.pt` → `checkpoint-epoch-N.pt`.

Both `save/load` handle `ModuleDict` (GAN) and dict optimizers/schedulers.

### `distributed.py` — Distributed Training

Thin wrappers around PyTorch DDP:
- `setup_distributed(rank, world_size, backend)` — initializes process group
- `cleanup_distributed()` — destroys process group
- `wrap_model_ddp(model, device)` — wraps in `DDP`, handles `ModuleDict` (GAN)
- `get_distributed_dataloader(dataloader, rank, world_size)` — creates `DistributedSampler`
- `is_main_process()` — True if rank 0 or not distributed

### `s3_cache_backend.py` — S3 Remote Cache for torch.compile

PyTorch's OSS inductor remote cache only supports Redis. This module provides S3-compatible alternatives for all five torch.compile cache artifacts.

#### The five torch.compile cache artifacts

When `torch.compile` runs, it produces five distinct cache artifacts across its compilation pipeline. Understanding their size and regeneration cost determines which ones are worth caching on a slow remote store (R2, cross-region S3):

| Cache | Size | Regen cost | Remote ROI | Default |
|---|---|---|---|---|
| **AutotuningCache** | KB–low MB | 30–300s+ (GPU benchmark per kernel config) | **Best** — tiny to transfer, very expensive to regen | ✓ enabled |
| **FXGraphCache** | 1–50 MB | 5–40s (Inductor optimization passes + codegen) | **Good** — skips all of Inductor on hit | ✓ enabled |
| **AOTAutogradCache** | 1–20 MB | 2–30s (autograd tracing + min-cut partition) | **Good** — training only; works together with FXGraphCache | ✓ enabled |
| **PGO cache** | KB | negligible | Nice-to-have — reduces recompilation count | ✗ disabled |
| **TritonCache / cubins** | 50 MB–4+ GB | 10–120s+ (Triton → PTX → cubin pipeline) | **Risky** on slow networks — transfer may be slower than regen | ✗ disabled |

**Note on Triton cubins:** PyTorch OSS bundles cubin files directly into FXGraphCache entries via `bundle_triton_into_fx_graph_cache` (enabled by default). So an FXGraphCache remote hit implicitly restores the cubins too — meaning you don't need the standalone Triton remote cache in most cases.

#### Classes

- `S3RemoteCacheBackend` — implements PyTorch inductor's `RemoteCacheBackend[bytes]` interface using boto3. Used for FXGraphCache, AutotuningCache, AOTAutogradCache, and PGO cache.
- `TritonS3RemoteCacheBackend` — implements Triton's separate `RemoteCacheBackend` interface (different API: `get(filenames) -> dict`, `put(filename, data)`). Used only for the standalone Triton cubin cache.
- `install_s3_cache_backend(fx_graph, autotune, aot_autograd, pgo, triton)` — patches inductor `RemoteCache` subclasses via `backend_override_cls` before `torch.compile` is called. Defaults enable the first three (best ROI), disable PGO and Triton.

#### Configuration

Two independent conditions must both be true:

1. **S3 backend enabled** — set `TORCH_COMPILE_S3_REMOTE_CACHE=1` (env var, takes precedence) **or** `torch_compile.remote_cache.enabled: true` (config). Either suffices.
2. **Inductor feature flags** — always required as env vars: `TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE=1`, `TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE=1`, `TORCHINDUCTOR_BUNDLED_AUTOTUNE_REMOTE_CACHE=1`.

Per-cache flags (only take effect when the S3 backend is enabled):

```yaml
torch_compile:
  remote_cache:
    enabled: false  # override with TORCH_COMPILE_S3_REMOTE_CACHE=1
    fx_graph: true       # default on
    autotune: true       # default on
    aot_autograd: true   # default on
    pgo: false           # default off
    triton: false        # default off (separate env var mechanism)
```

See `ENVIRONMENT_SETUP.md` and `beam.yaml` for the full env var list.

### `sampling.py` — Token Sampling

`sample_next_token(logits, *, temperature, top_k, top_p, generator) -> (token, log_prob)` —
single source of truth for autoregressive next-token sampling. Supports greedy
(`temperature=0`), temperature scaling, top-k truncation, and nucleus (top-p)
filtering. Returns the sampled token id **and** its log-probability under the
filtered distribution actually sampled from — the latter is what RL importance
ratios need, so rollout code does not have to recompute it. Used by
`AutoregressiveLM.generate` and `NanochatGPT.generate`.

### `torch_np_utils.py` — Tensor/NumPy Conversions

`to_numpy(tensor)` — moves to CPU, detaches, converts to numpy. Handles all dtypes including bfloat16 (cast to float32 first).

### `wandb_utils.py` — W&B Utilities

`WANDB_RUN_ID_FILENAME` — constant filename (`.wandb_run_id`) for storing run IDs.

`find_wandb_run_id_for_resume(results_dir, experiment_name, load_checkpoint_path)` — locates the run ID file for a previous run to enable W&B resume.

## src/utils/remote_backup — Remote Backup Strategies

### `base.py` — RemoteBackupStrategy

Abstract base with:
- `should_backup(checkpoint_name) → bool` — whether this strategy uploads the given checkpoint name
- `upload(local_path, checkpoint_name)` — uploads the file

### `ssh_strategy.py` — SSHBackupStrategy

Uploads checkpoints to a remote SSH server via `scp`/`rsync`.

### `wandb_strategy.py` — WandbBackupStrategy

Uploads checkpoints as W&B artifacts. Only saves checkpoints in `allowed_names` (default: `["best", "last"]`).
