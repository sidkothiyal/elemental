# src/utils — Architecture

## CUDAPrefetcher (initializer.py)

Prefetches one batch ahead using a side CUDA stream:

```
                  default stream              side stream
iter N:   consume current_batch   ||  prefetch next_batch (HtoD)
```

On each `yield`:
1. `wait_stream()` — ensures side stream is done before consuming
2. `record_stream()` — tells allocator this batch is used on default stream
3. Kicks off prefetch of the *next* batch on the side stream

`move_to_device()` and `_record_stream_recursive()` handle nested dicts/lists of tensors.

CPU/MPS fallback: `stream=None`, acts as a plain passthrough iterator.

## initialize_dataset (initializer.py)

1. Instantiates `dataset` (train/val/test) via `hy_utils.instantiate(config)`
2. Instantiates `DataLoader` with configured workers/pin_memory/etc.
3. For distributed: replaces sampler with `DistributedSampler`
4. Wraps each non-None loader in `CUDAPrefetcher`

## compile_model_and_optimizer_step

When `torch_compile` config present with `enabled: true`:
- Installs `S3RemoteCacheBackend` if `remote_cache.enabled: true` OR `TORCH_COMPILE_S3_REMOTE_CACHE=1` (must happen before `torch.compile`)
- Wraps model with `torch.compile(model, **compile_kwargs)`
- Optionally compiles `optimizer.step` if `compile_optimizer_step: true`

## S3RemoteCacheBackend (s3_cache_backend.py)

PyTorch's OSS inductor remote cache only supports Redis. This module provides S3-compatible backends for all five torch.compile cache artifacts.

### Two cache systems, two backend classes

**Inductor caches** (`S3RemoteCacheBackend`): FXGraphCache, AutotuningCache, AOTAutogradCache, PGO cache all share PyTorch's `RemoteCacheBackend[bytes]` interface. `install_s3_cache_backend()` patches `backend_override_cls` on the relevant `RemoteCache` subclasses. The `backend_override_cls` hook is PyTorch's built-in test/override mechanism — `RemoteCache.__init__` checks it and uses the override factory instead of constructing a Redis backend.

**Triton cache** (`TritonS3RemoteCacheBackend`): Triton has a completely separate cache system in `triton.runtime.cache`. Its `RemoteCacheBackend` uses a different interface (`get(filenames) -> Dict[str, bytes]`, `put(filename, data)`). Activated by setting `TRITON_CACHE_MANAGER=triton.runtime.cache.RemoteCacheManager` and `TRITON_REMOTE_CACHE_BACKEND=utils.s3_cache_backend.TritonS3RemoteCacheBackend`.

### Default cache selection

Only the three highest-ROI caches are enabled by default for slow/high-latency stores:

```
AutotuningCache   KB–low MB  best ROI: tiny + very expensive to regen (GPU benchmarks)
FXGraphCache      1–50 MB    good ROI: skips all Inductor passes on hit
AOTAutogradCache  1–20 MB    good ROI: training only
PGO cache         KB         disabled: low time savings
TritonCache/cubin 50MB–4GB+  disabled: transfer may exceed regen time on slow networks
                              (cubins are bundled into FXGraphCache entries anyway)
```

`install_s3_cache_backend(fx_graph, autotune, aot_autograd, pgo, triton)` accepts per-cache flags so experiment configs can opt into PGO or Triton caching explicitly.

### Activation — two independent conditions

Both must be true for remote caching to work:

1. **S3 backend enabled** — `TORCH_COMPILE_S3_REMOTE_CACHE=1` (env var) **or** `torch_compile.remote_cache.enabled: true` (config). OR-ed in `initializer.py` so remote machines can activate it without changing config files.
2. **Inductor feature flags** — `TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE=1`, `TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE=1`, `TORCHINDUCTOR_BUNDLED_AUTOTUNE_REMOTE_CACHE=1`. Tell inductor to call a remote backend at all; without these the S3 backend is installed but never invoked.

## Checkpoint (checkpoint.py)

### Save format

```python
{
    "epoch": int | None,
    "step": int | None,
    "model_state_dict": state_dict,
    "optimizer_state_dict": state_dict,
    "scheduler_state_dict": state_dict,
    "criterion_state_dict": state_dict,
    # + any extra kwargs
}
```

For `ModuleDict` models (GAN): state dict is `{"generator": ..., "discriminator": ...}`.

### find_latest_checkpoint algorithm

1. Strip datetime suffix from experiment name using `_DATETIME_SUFFIX_RE`
2. Scan `results_dir` for entries matching base name + datetime suffix
3. Sort by datetime string (lexicographic = chronological for ISO format)
4. For each dir, try: preferred names → step checkpoints → epoch checkpoints
5. Return first valid file found

## Distributed (distributed.py)

`MASTER_ADDR` and `MASTER_PORT` read from environment; defaults to `localhost:12355`.

`wrap_model_ddp` for `ModuleDict`: wraps each sub-model individually in `DDP`, returns new `ModuleDict`. This allows separate optimizer creation per sub-model (used in GAN).

## Remote Backup (remote_backup/)

`RemoteBackupStrategy.should_backup(name)` gate prevents uploading every checkpoint (only specified names). Implemented in subclasses:

- `WandbBackupStrategy`: uses `wandb.save(local_path)` or artifact API
- `SSHBackupStrategy`: uses `subprocess.run(["scp", ...])` or `rsync`

Both are instantiated via Hydra and passed to `CheckpointCallback` via `update_config`.
