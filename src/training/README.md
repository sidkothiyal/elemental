# src/training — Trainer Hierarchy

Owns the epoch loop, the single-batch forward/backward step, and the construction of a trainer from a Hydra config. The CLI at `src/trainer.py` is a thin entrypoint that delegates here.

## Files

### `builder.py` — `build_trainer(config_file, local_rank, overrides) -> Trainer`

Composes a Hydra config, initializes the environment (seeds, TF32, device, DDP), builds loaders and callbacks, dispatches to `_trainer_class_for(config.mode)`, then invokes that class's `_build_{model,optimizer,scheduler,loss,scaler}` classmethods.

**Mode dispatch:**
| `config.mode` | Class |
|---|---|
| `"supervised"` (default) | `Trainer` |
| `"gan"` | `GANTrainer` |
| `"customizable"` | `CustomizableTrainer` |
| `"graph"` | `GraphTrainer` (early exit, separate init path) |
| `"rl"` | `RLTrainer` |

Also exports `build_graph_trainer(config_file, overrides)` for direct graph-mode construction.

### `trainer.py` — `class Trainer`

Supervised base. Drives the full training/validation/test lifecycle.

**Public API:**
- `train()` — epoch loop over `range(start_epoch, num_epochs)`; `finally`-stops the profiler and cleans up distributed.
- `test()` — single test-phase pass.
- `run_epoch(epoch)` — one training epoch plus optional validation.
- `train_batch(batch, ..., *, training=True) -> dict[str, Tensor]` — the single override point. Runs forward + loss inside `torch.autocast` and `torch.set_grad_enabled(training)`. Backward / optimizer.step / grad-clip / per-batch scheduler step are gated on `training=True`.
- `eval_batch(...)` — one-liner: `return self.train_batch(..., training=False)`.

**`_build_*` classmethod hooks (override points):**
- `_build_model(config, device, use_distributed) -> nn.Module`
- `_build_optimizer(config, model) -> Optimizer | dict[str, Optimizer]` (auto-detects ModuleDict)
- `_build_scheduler(config, optim) -> LRScheduler | dict | None` (mirrors optim shape)
- `_build_loss(config, device) -> LossContainer`
- `_build_scaler(config, device, dtype) -> GradScaler | None`

### `gan_trainer.py` — `class GANTrainer(Trainer)`

Overrides:
- `_build_model` → `nn.ModuleDict({"generator", "discriminator"})`
- `_build_optimizer` / `_build_scheduler` → inherited from base (auto-detect ModuleDict → dict-of-optimizers/schedulers)
- `_build_loss` → dict (adversarial / feature_matching)
- `train_batch` → three phases (disc-on-fake → disc-on-real → generator), each with its own zero_grad / backward / step cycle, all gated on `training`.

Postprocessor runs **inside** autocast (unified with the supervised path).

### `rl_trainer.py` — `class RLTrainer(Trainer)` (skeleton)

Wires the override shape for RL. `_collect_rollouts`, `_compute_advantages`, and `_update_policy` all raise `NotImplementedError`. `train_batch` orchestrates them so that once the three methods have bodies, the trainer is complete.

### `customizable_trainer.py` — `class CustomizableTrainer(Trainer)`

Inherits the full Trainer lifecycle and overrides `train_batch` to execute a `GraphNode` per batch. The graph receives all trainer state as scope variables (`model`, `optim`, `loss_fn`, `scaler`, `batch`, `epoch`, `training`, etc.) and owns the full forward/backward/step choreography.

- `_build_graph(config)` classmethod — instantiates the per-batch graph from `config.graph`.
- Config-driven `metrics_format` mapping (e.g. `{generator: generator_loss}`) customizes metric names.

### `graph_trainer.py` — `class GraphTrainer`

Standalone graph-execution trainer (does NOT inherit from Trainer). Manages `DependencyResolver`, `GraphExecutor`, and epoch loop for fully declarative training pipelines defined in YAML (`nodes:` + `execution_graph:` sections).

### `helpers.py` — stateless helpers

Shared across all trainers:
- `build_data_container(batch, ...)` — namespace batch keys as `input/<key>`, stash `model` / `epoch`
- `populate_predictions(pred, data)` — copy `pred` and `pred/<key>` back into data
- `collect_module_losses(model, dataset=...)` — walk submodules, pull `.losses` or `.compute_losses()` output
- `clear_module_losses(model)`, `cleanup_memory(model, optim, device)`
- `setup_profiler(config, results_dir)` — torch.profiler wrapper
- `should_step_optimizer(batch_idx, grad_accum_steps, total_batches)` — gradient accumulation boundary logic
- `backward_step(loss, scaler, grad_accum_steps)` — scale + backward (with or without GradScaler)
- `optimizer_step(*, optim, parameters, scaler, clip_grad_norm)` — unscale + clip + step + update + zero_grad
- `build_optim_dict_or_single(config, model)` — dict-of-optimizers for ModuleDict, else single
- `build_scheduler_dict_or_single(config, optim)` — mirrors optim shape
- `materialize_and_average_losses(running, denom)`, `_accumulate_loss(running, key, tensor)`
- `_get_cached_signature(fn)` — cached `inspect.signature`

## Usage

```python
from training.builder import build_trainer

trainer = build_trainer("experiment/autoencoder/compression_unet", overrides=["lr=0.001"])
trainer.train()
```

See `src/trainer.py` for the CLI wrapper.
