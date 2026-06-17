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


### `trainer.py` — `class Trainer`

Supervised base. Drives the full training/validation/test lifecycle.

**Public API:**
- `train()` — epoch loop over `range(start_epoch, num_epochs)`; `finally`-stops the profiler and cleans up distributed.
- `test()` — single test-phase pass.
- `run_epoch(epoch)` — one training epoch plus optional validation.
- `train_batch(batch, ..., *, training=True) -> dict[str, Tensor]` — the single override point. Runs forward + loss inside `torch.autocast` and `torch.set_grad_enabled(training)`. Backward / optimizer.step / grad-clip / per-batch scheduler step are gated on `training=True`.
- `eval_batch(...)` — one-liner: `return self.train_batch(..., training=False)`.
- `_validate(epoch, *, log_step=None, max_batches=None)` — one validation pass, shared by epoch-boundary validation and mid-training step validation. Snapshots/restores transient loop state + the model's train/eval mode.

**Mid-training step validation & production logging:**
- `validation_interval_steps` (config) runs a short held-out eval every N optimizer steps (independent of the epoch-based `validation_interval`); `validation_max_batches` bounds it. Essential for streaming `IterableDataset`s whose epoch never ends. Logs `val/loss_*`, `val/perplexity`, and (with the LM metrics callback) `val/next_token_acc`, `val/bits_per_byte`.
- `_run_batch_loop(..., max_batches=None, log_step=None)` — `max_batches` caps the loop (bounded eval); `log_step` aligns the final metrics to the global optimizer step.
- Every `log_interval` steps the trainer also emits `train/grad_norm`, `throughput/tokens_per_sec`, `throughput/achieved_tflops`, `throughput/mfu` (when `peak_flops_per_sec` is set), `gpu_mem_allocated_gb`/`gpu_mem_reserved_gb`, and `perplexity`. Per-group LRs are logged as `learning_rate/<sub-optimizer>` (e.g. `learning_rate/muon`, `learning_rate/adamw`).
- `skip_nonfinite_steps` (config): a NaN/Inf loss drops the accumulation window (counted as `train/skipped_steps`) instead of raising `NaNLossError`.

**Resume:** `_global_optim_step` is saved into the checkpoint and restored on resume (the legacy `step` field is the micro-batch counter and over-advances the LR schedule under gradient accumulation). `_restore_dataloader_state()` replays a `StatefulDataLoader`'s saved stream position so a streaming dataset continues past consumed docs.

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

### `rl_trainer.py` — `class RLTrainer(Trainer)`

Algorithm-, env-, and policy-agnostic RL trainer. Holds four config-instantiated
collaborators:
- `env: BaseVecEnv` (`src/env/`) — vectorized environment
- `policy_wrapper: BasePolicyWrapper` (`src/rl/policy_wrapper.py`) — wraps
  `self.model`, owns value head + any exploration params (e.g. log_std)
- `rollout_buffer: OnPolicyRolloutBuffer` (`src/rl/rollout_buffer.py`)
- `rl_algorithm: BaseRLAlgorithm` (`src/rl/algorithm.py`, e.g. `PPOAlgorithm`)

Overrides:
- `run_epoch` → rollout phase (no-grad) + policy-update phase + periodic sim-eval.
  The algorithm drives env.step, fills the buffer, and dispatches each minibatch
  back to `self.train_batch` so all standard infrastructure (AMP, DDP, scaler,
  grad-clip, callbacks) runs unchanged.
- `train_batch` → builds the data container, calls `policy_wrapper.evaluate_actions`,
  populates `rl/log_prob_new`, `rl/value_new`, `rl/entropy`, runs `self.loss(data)`,
  then `backward_step`/`optimizer_step` from `helpers`.
- `test` → `_run_sim_eval(0)` (deterministic rollout, no updates).

`_build_env` / `_build_policy_wrapper` / `_build_rollout_buffer` / `_build_algorithm`
classmethod hooks instantiate the RL collaborators from Hydra; `_attach_policy_params`
adds extra wrapper-owned params (value head, log_std) to the optimizer.

`self.loaders` stays `{}` in RL mode — the builder gates supervised dataloader
init on `config.dataloader`.

### `customizable_trainer.py` — `class CustomizableTrainer(Trainer)`

Inherits the full Trainer lifecycle and overrides `train_batch` to execute a `GraphNode` per batch. The graph receives all trainer state as scope variables (`model`, `optim`, `loss_fn`, `scaler`, `batch`, `epoch`, `training`, etc.) and owns the full forward/backward/step choreography.

- `_build_graph(config)` classmethod — instantiates the per-batch graph from `config.graph`.
- Config-driven `metrics_format` mapping (e.g. `{generator: generator_loss}`) customizes metric names.

Reusable graphs live in [`configs/graph/`](../../configs/graph/) and are composed via Hydra `defaults: - /graph: <name>`. Example experiments:
- `configs/experiment/gan/customizable_gan.yaml` — GAN via `configs/graph/gan_per_batch.yaml`.
- `configs/experiment/distill/distill_dinov3_to_vit.yaml` — feature distillation (frozen DINOv3 teacher → smaller ViT student) via `configs/graph/distill_per_batch.yaml`.

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
- `build_optim_dict_or_single(config, model)` — three branches in priority:
  (1) `config.optimizer_groups` set → `MultiOptimizer` with one sub-optimizer
  per named partition; (2) `model` is `ModuleDict` → dict-of-optimizers;
  (3) single optimizer over all trainable params. See
  [src/optimizer/](../optimizer/README.md).
- `build_scheduler_dict_or_single(config, optim)` — accepts either a string
  (looks up `torch.optim.lr_scheduler.<name>`) or a dict with `_target_:`
  (Hydra-instantiate with `optimizer=` injected). Mirrors optim shape.
- `materialize_and_average_losses(running, denom)`, `_accumulate_loss(running, key, tensor)`
- `_get_cached_signature(fn)` — cached `inspect.signature`

### `model_exporter.py` — `class ModelExporter` + `build_model_exporter`

Exports a trained model to ONNX. Supports two extraction methods per module entry:

- **PyTorch-level** (`input_layer` / `output_layer`): resolves a dotted submodule path
  (e.g. `"model.encoder"`) via `_get_submodule`, then calls `torch.onnx.export` on
  that submodule directly.
- **ONNX-level** (`source_onnx` + `input_tensor` / `output_tensor`): calls
  `onnx.utils.extract_model` to slice an already-exported ONNX graph by tensor name.

`build_model_exporter` accepts three config sources:

| Config type | Example |
|---|---|
| Full experiment config | `experiment/autoencoder/compression_ae` |
| Model-only config | `model/vq_vae_transformer` |
| Dedicated export config | `export/vq_vae_transformer` |

It reuses `Trainer._build_model` and `initializer.load_from_checkpoint`, skipping
optimizer / scheduler / loss / callbacks entirely.

## Usage

```python
from training.builder import build_trainer
from training.model_exporter import build_model_exporter

# Training
trainer = build_trainer("experiment/autoencoder/compression_unet", overrides=["lr=0.001"])
trainer.train()

# Export
exporter = build_model_exporter(
    "export/vq_vae_transformer",
    output_dir="exports/vqvae/",
    checkpoint="checkpoints/vqvae.pt",
)
exporter.export()
```

See `src/trainer.py` for the training CLI and `src/exporter.py` for the export CLI.
