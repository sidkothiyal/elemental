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
- Wraps model with `torch.compile(model, **compile_kwargs)`
- Optionally compiles `optimizer.step` if `compile_optimizer_step: true`

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
