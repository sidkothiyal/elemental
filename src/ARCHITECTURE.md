# src/ — Architecture

## trainer.py (CLI)

Thin Hydra/Typer entrypoint. Delegates to `training.builder.build_trainer(...)` and calls `.train()` or `.test()` on the returned Trainer instance.

## training/

Class hierarchy that owns the epoch loop, batch step, and initialization.

### Layout
- `helpers.py` — Stateless helpers shared by all trainers: `build_data_container`, `populate_predictions`, `collect_module_losses`, `clear_module_losses`, `cleanup_memory`, `setup_profiler`, `should_step_optimizer`, `materialize_and_average_losses`, `_accumulate_loss`, `_get_cached_signature`.
- `trainer.py` — `class Trainer`, the supervised base.
- `gan_trainer.py` — `class GANTrainer(Trainer)`.
- `rl_trainer.py` — `class RLTrainer(Trainer)` skeleton (stubs raise `NotImplementedError`).
- `builder.py` — `build_trainer(config_file, local_rank, overrides) -> Trainer`; composes Hydra config, sets up env/DDP/loaders/callbacks, dispatches to `_trainer_class_for(config.mode)`, then invokes the class's `_build_{model,optimizer,scheduler,loss,scaler}` classmethods.

### Key methods

**`Trainer.train()`** — Full training loop over `range(start_epoch, num_epochs)`. `finally`-stops the profiler and cleans up distributed.

**`Trainer.test()`** — Single test-phase pass.

**`Trainer.run_epoch(epoch)`** — One training epoch + optional validation. Handles `set_epoch` for distributed samplers, per-epoch scheduler step, `on_phase_end`/`on_phase_start` callback hooks, and validation interval gating.

**`Trainer._run_batch_loop(phase, epoch, loader)`** — Tqdm driver. Dispatches to `train_batch` or `eval_batch`, accumulates losses, fires batch-level callbacks, runs the profiler step. Never overridden.

**`Trainer.train_batch(batch, batch_idx, epoch, step, data, *, training=True)`** — The single override point. Runs forward + loss + module-loss collection inside `torch.autocast` and `torch.set_grad_enabled(training)`. Backward, optimizer step, grad-clip, and per-batch scheduler step are all gated on `training=True`. Returns a dict of detached loss tensors (components + module losses + `"total"`).

**`Trainer.eval_batch(...)`** — One-liner: `return self.train_batch(..., training=False)`. Eliminates duplicated val/test code.

**`GANTrainer.train_batch(...)`** — Three-phase override: disc-on-fake → disc-on-real → generator, each with its own zero_grad/backward/step cycle gated on `training`. Postprocessor runs **inside** autocast (unified with supervised path).

**`GANTrainer._build_model/_optimizer/_scheduler/_loss`** — Produce `ModuleDict` / dict-of-optimizers / dict-of-schedulers / dict-of-losses keyed by `"generator"`/`"discriminator"`.

**`RLTrainer.train_batch(...)`** — Orchestrates `_collect_rollouts` → `_compute_advantages` → `_update_policy`. All three currently raise `NotImplementedError` — bodies arrive with SIMPLER env integration.

### Training Flow
1. `build_trainer` → compose config, init env (seeds, TF32, device, DDP), build callbacks + loaders, dispatch to `cls._build_{model,optim,scheduler,loss,scaler}`, load checkpoint, compile, wire postprocessor + profiler, seed callbacks with model/optimizer/device/dtype/config.
2. `Trainer.train()` iterates `run_epoch(epoch)` over `range(start_epoch, num_epochs)`.
3. Each `run_epoch`: `cleanup_memory` → `on_epoch_start` → `_run_batch_loop("train", ...)` → per-epoch scheduler step → `on_phase_end` → (every N epochs) `_run_batch_loop("val", ...)`.
4. Each batch: `build_data_container` → forward → `populate_predictions` → postprocess → loss → `collect_module_losses` → (if training) backward → step → per-batch scheduler → callbacks.

## graph_trainer.py

### Classes

**`DependencyResolver`** — Resolves inter-node references in config.
- `extract_dependencies(config)` — Finds all `node:<name>` string references recursively
- `topological_sort(node_configs)` — Kahn's algorithm for dependency ordering
- `resolve_references(config)` — Replaces `node:<name>.<attr>` strings with actual object references (supports attribute access and `method()` calls)

**`GraphTrainer(config)`** — Manages the graph-based training lifecycle.
- `initialize_nodes()` — Sorts configs by dependency, instantiates each via Hydra, registers with `GraphExecutor`
- `create_execution_functions()` — Instantiates `ExecuteFunction` list from `execution_graph` config
- `execute_epoch(epoch)` — Clears epoch variables, runs each execution function sequentially, stores outputs
- `train()` — Runs `initialize_nodes()` then epoch loop calling `execute_epoch()`

### Node Reference Syntax
In config YAML, `"node:model_node.parameters()"` resolves to calling `.parameters()` on the node registered as `model_node`. Nested attribute access like `"node:dataset.train_loader"` also works.
