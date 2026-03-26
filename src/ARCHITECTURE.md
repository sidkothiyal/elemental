# src/ — Architecture

## trainer.py

### Key Functions

**`run_train(config_file, test, local_rank)`** — Top-level orchestrator. Loads Hydra config, sets up distributed training, initializes device/model/dataset/loss/optimizer/callbacks, runs the epoch loop (train → validate → test), handles checkpointing and remote backup.

**`batch_loop(phase, epoch, dataloader, model, loss, optim, callbacks, device, ...)`** — Standard training loop for one epoch. Runs forward pass under `torch.autocast`, computes loss via `LossContainer`, collects module-level losses, runs backward pass with gradient scaling, steps optimizer/scheduler, runs postprocessors and callbacks. Returns dict of average losses.

**`batch_loop_gan(phase, epoch, dataloader, model, loss, optim, ...)`** — GAN variant. Alternates discriminator and generator training steps. Model must be `nn.ModuleDict({"generator": ..., "discriminator": ...})`. Loss and optimizer are also dicts keyed by `"generator"`/`"discriminator"`.

**`build_data_container(batch, data, model, epoch)`** — Creates the unified data dict. Prefixes all batch keys with `input/` except `target`. Stores `model` and `epoch` references.

**`populate_predictions(pred, data)`** — Writes model output into the data container. Dict outputs get `pred/<key>` sub-keys.

**`collect_module_losses(model, dataset, pred, target, model_input_args)`** — Walks `model.named_modules()` and dataset, calls `compute_losses()` on any that define it (introspecting signature to pass matching kwargs), then collects `.losses` dicts. Returns combined loss dict with path-prefixed names.

**`setup_profiler(profiler_config, results_dir)`** — Builds `torch.profiler.profile` from config with schedule, activity selection, and TensorBoard/Chrome trace export.

**`cleanup_memory(model, optim, device, epoch, empty_cache_interval)`** — Zeros gradients, clears module losses, optionally calls `torch.cuda.empty_cache()`.

### Training Flow
1. `run_train` → init device, model, dataset, loss, optimizer, scheduler, callbacks
2. For each epoch: `batch_loop` (or `batch_loop_gan` if `mode: "gan"`)
3. Each batch: `build_data_container` → forward → `populate_predictions` → postprocess → loss → backward → step → callbacks
4. After training batches: `cleanup_memory` → validation loop → checkpoint

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
