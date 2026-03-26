# src/utils — Shared Utilities

Cross-cutting helpers used by trainer, callbacks, and tools.

## Files

### `initializer.py` — Experiment Initialization

`CUDAPrefetcher` — wraps a `DataLoader` and prefetches batches to GPU using a side CUDA stream. Overlaps HtoD transfers with compute, significantly reducing GPU idle time. Behaves like a DataLoader iterator.

Key functions:
- `initialize_device(device_str)` — returns `torch.device`, handles `"gpu"` (CUDA/MPS), CPU
- `initialize_dtype(dtype_str)` — returns `torch.dtype`, falls back to float32 if bfloat16 unsupported
- `set_seeds(seed)` — seeds PyTorch, CUDA, NumPy, Python random
- `initialize_dataset(dataloader_config, ...)` — instantiates dataset + dataloader from Hydra config, wraps in `CUDAPrefetcher`, sets up `DistributedSampler` if needed
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
