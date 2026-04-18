# src/ — Architecture

## trainer.py (CLI)

Thin Hydra/Typer entrypoint. Delegates to `training.builder.build_trainer(...)` and calls `.train()` or `.test()` on the returned instance. Handles all modes, including `mode: graph`, which `build_trainer` dispatches to `training.graph_trainer.GraphTrainer` via a dedicated init path.

## training/

Class hierarchy that owns the epoch loop, batch step, and initialization.

### Layout
- `helpers.py` — Stateless helpers shared by all trainers: `build_data_container`, `populate_predictions`, `collect_module_losses`, `clear_module_losses`, `cleanup_memory`, `setup_profiler`, `should_step_optimizer`, `materialize_and_average_losses`, `_accumulate_loss`, `_get_cached_signature`, `backward_step`, `optimizer_step`, `build_optim_dict_or_single`, `build_scheduler_dict_or_single`.
- `trainer.py` — `class Trainer`, the supervised base. Uses `backward_step` / `optimizer_step` helpers for the backward/step cycle; delegates `_build_optimizer` / `_build_scheduler` to the shared dict-aware helpers.
- `gan_trainer.py` — `class GANTrainer(Trainer)`. Inherits `_build_optimizer` / `_build_scheduler` from base (shared helpers auto-detect ModuleDict → dict-of-optimizers).
- `customizable_trainer.py` — `class CustomizableTrainer(Trainer)`. Overrides `train_batch` to execute a user-defined `GraphNode` per batch. Config-driven `metrics_format` for metric naming.
- `graph_trainer.py` — `class GraphTrainer`. Standalone graph-execution trainer (does not inherit from `Trainer`). Manages `DependencyResolver`, `GraphExecutor`, and epoch loop for fully declarative pipelines.
- `rl_trainer.py` — `class RLTrainer(Trainer)` skeleton (stubs raise `NotImplementedError`).
- `builder.py` — `build_trainer(config_file, local_rank, overrides) -> Trainer | GraphTrainer`; composes Hydra config, dispatches `mode: graph` to `GraphTrainer` early. Other modes go through `_trainer_class_for(config.mode)` → `cls._build_{model,optimizer,scheduler,loss,scaler}` classmethods. Also exports `build_graph_trainer()` for direct graph-mode construction.

### Trainer dispatch (`config.mode`)

| `mode` | Class | Notes |
|--------|-------|-------|
| `autoencoder`, `classification`, etc. | `Trainer` | Standard supervised |
| `gan` | `GANTrainer` | Three-phase imperative GAN |
| `customizable` | `CustomizableTrainer` | Per-batch graph, inherits full Trainer lifecycle |
| `graph` | `GraphTrainer` | Standalone graph executor, own epoch loop |
| `rl` | `RLTrainer` | Stub |

### Key methods

**`Trainer.train()`** — Full training loop over `range(start_epoch, num_epochs)`. `finally`-stops the profiler and cleans up distributed.

**`Trainer.test()`** — Single test-phase pass.

**`Trainer.run_epoch(epoch)`** — One training epoch + optional validation. Handles `set_epoch` for distributed samplers, per-epoch scheduler step, `on_phase_end`/`on_phase_start` callback hooks, and validation interval gating.

**`Trainer._run_batch_loop(phase, epoch, loader)`** — Tqdm driver. Dispatches to `train_batch` or `eval_batch`, accumulates losses, fires batch-level callbacks, runs the profiler step. Never overridden.

**`Trainer.train_batch(batch, batch_idx, epoch, step, data, *, training=True)`** — The single override point. Runs forward + loss + module-loss collection inside `torch.autocast` and `torch.set_grad_enabled(training)`. Uses `backward_step()` and `optimizer_step()` helpers. Returns a dict of detached loss tensors (components + module losses + `"total"`).

**`Trainer.eval_batch(...)`** — One-liner: `return self.train_batch(..., training=False)`.

**`GANTrainer.train_batch(...)`** — Three-phase override: disc-on-fake → disc-on-real → generator, each with its own zero_grad/backward/step cycle gated on `training`.

**`CustomizableTrainer.train_batch(...)`** — Executes `self.graph` (a `GraphNode`) with seeded variables (model, optim, loss, scaler, batch, epoch, etc.). The graph owns the full forward/backward/step choreography. Returns a detached loss dict.

**`GraphTrainer.train()`** — Initializes nodes via `DependencyResolver.topological_sort`, then runs `execute_epoch()` in a loop. Each epoch clears epoch variables, instantiates execution functions, and runs them sequentially.

### Shared helpers (`training/helpers.py`)

- `backward_step(loss, scaler, grad_accum_steps)` — Scales loss and runs backward (with or without GradScaler).
- `optimizer_step(*, optim, parameters, scaler, clip_grad_norm)` — Unscale + clip + step + scaler.update + zero_grad.
- `build_optim_dict_or_single(config, model)` — Returns a dict of optimizers when model is ModuleDict; else single optimizer.
- `build_scheduler_dict_or_single(config, optim)` — Mirrors optim shape (dict or single) for schedulers.

### Training Flow
1. `build_trainer` → compose config, init env (seeds, TF32, device, DDP), build callbacks + loaders, dispatch to `cls._build_{model,optim,scheduler,loss,scaler}`, load checkpoint, compile, wire postprocessor + profiler, seed callbacks with model/optimizer/device/dtype/config.
2. `Trainer.train()` iterates `run_epoch(epoch)` over `range(start_epoch, num_epochs)`.
3. Each `run_epoch`: `cleanup_memory` → `on_epoch_start` → `_run_batch_loop("train", ...)` → per-epoch scheduler step → `on_phase_end` → (every N epochs) `_run_batch_loop("val", ...)`.
4. Each batch: `build_data_container` → forward → `populate_predictions` → postprocess → loss → `collect_module_losses` → (if training) backward → step → per-batch scheduler → callbacks.

### Node Reference Syntax
In config YAML, `"node:model_node.parameters()"` resolves to calling `.parameters()` on the node registered as `model_node`. Nested attribute access like `"node:dataset.train_loader"` also works. Dict access on `ModuleDict` or plain dicts is supported: `"node:model.generator"`.
