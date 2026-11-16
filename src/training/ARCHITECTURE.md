# src/training — Architecture

## Package layout

```
training/
├── __init__.py                # Re-exports all trainers + build_trainer
├── builder.py                 # build_trainer() + dispatch
├── trainer.py                 # class Trainer (supervised base)
├── gan_trainer.py             # class GANTrainer(Trainer) — three-phase GAN step
├── customizable_trainer.py    # class CustomizableTrainer(Trainer) — per-batch graph execution
├── graph_trainer.py           # class GraphTrainer — standalone graph-execution trainer
├── rl_trainer.py              # class RLTrainer(Trainer) — rollout + PPO update via train_batch
├── model_exporter.py          # class ModelExporter + build_model_exporter() — ONNX export
└── helpers.py                 # Stateless helpers shared across trainers
```

## Class hierarchy

```
Trainer (supervised base)
├── GANTrainer              # train_batch overridden with 3-phase step
├── CustomizableTrainer     # train_batch delegates to a user-defined GraphNode
└── RLTrainer               # run_epoch drives rollout+update; train_batch does one PPO minibatch

GraphTrainer                # standalone, does NOT inherit from Trainer
```

Subclasses override:
1. **`train_batch(..., *, training)`** — the per-batch step. The `training` flag gates backward/step.
2. **`_build_*` classmethods** — how the model, optimizer, scheduler, loss, scaler are constructed from config.

Everything else — epoch loop, callback wiring, dataloader iteration, profiler step, checkpoint load, DDP wrap — lives on the base.

## Build flow (`build_trainer`)

```
build_trainer(config_file, local_rank, overrides) -> Trainer
  1. hydra.compose → DictConfig
  2. _init_env(config, local_rank) → device, dtype, seeds, DDP setup
     (DDP is active iff `config.distributed` is true AND local_rank >= 0, i.e.
     launched under torchrun; `distributed` defaults to false in default.yaml)
  3. initializer.initialize_dataset(config.dataloader, ...) → {"train": DataLoader, "val": DataLoader, "test": DataLoader}
  4. remote_backup = initializer.initialize_remote_backup(config, results_dir)
  5. callbacks = initializer.initialize_callbacks(config)
  6. if config.mode == "graph": return GraphTrainer(config)  # early exit for graph mode
     cls = _trainer_class_for(config.mode)                # Trainer | GANTrainer | RLTrainer | CustomizableTrainer
  7. model     = cls._build_model(config, device, use_distributed)
     optim     = cls._build_optimizer(config, model)
     scheduler = cls._build_scheduler(config, optim)
     loss      = cls._build_loss(config, device)
     scaler    = cls._build_scaler(config, device, dtype)
  8. checkpoint_data = _maybe_load_checkpoint(model, optim, scheduler, loss, config)
  9. model = initializer.compile_model_and_optimizer_step(model, optim, config.get("torch_compile"))
 10. postprocessor = _build_postprocessor(config)
 11. profiler, profiler_step_per = _build_profiler(config, results_dir)
 12. callbacks.update_config(...)                         # one consolidated call
 13. return cls(config=..., model=..., optim=..., loss=..., scheduler=..., scaler=...,
                callbacks=..., loaders=..., postprocessor=..., profiler=..., ...)
```

The `_build_*` classmethods are the extension point. The base `_build_optimizer` uses `build_optim_dict_or_single`, which auto-detects `ModuleDict` models and returns dict-of-optimizers. `_build_scheduler` mirrors the optim shape. `GANTrainer` inherits this directly (no override needed). For `CustomizableTrainer`, the builder additionally calls `cls._build_graph(config)` and passes the result to the constructor. Shared per-batch graphs live in [`configs/graph/`](../../configs/graph/) (e.g. `gan_per_batch.yaml`, `distill_per_batch.yaml`) and are composed into experiments via `defaults: - /graph: <name>`.

## Training flow (`Trainer.train`)

```
Trainer.train()
  callbacks.on_phase_start("train")
  for epoch in range(start_epoch, num_epochs):
    run_epoch(epoch)
      cleanup_memory(model, optim, device, epoch=epoch, empty_cache_interval=...)
      callbacks.on_epoch_start(epoch, phase="train")
      model.train()
      _run_batch_loop("train", epoch, loaders["train"])
      (per-epoch scheduler step if scheduler_step_per == "epoch")
      callbacks.on_phase_end("train")
      if (epoch + 1) % validation_interval == 0:
        _validate(epoch)            # epoch-boundary eval (shared helper)
  (finally) stop profiler + cleanup distributed
```

`_validate(epoch, *, log_step=None, max_batches=None)` is the single eval path: it snapshots/restores transient loop state (`_total_batches`/`_current_loader`/`_current_phase`) and the model's train/eval mode, runs `_run_batch_loop("val", ..., max_batches, log_step)`, and restores `model.train()`. Mid-training **step** validation calls the same helper from inside the training loop (see below).

## Batch loop (`_run_batch_loop`)

```
for batch_idx, batch in enumerate(tqdm(loader)):
  step = epoch * total_batches + batch_idx
  callbacks.on_batch_start(step, phase)
  if phase == "train":
    losses = self.train_batch(batch, batch_idx, epoch, step, data, training=True)
  else:
    losses = self.eval_batch(batch, batch_idx, epoch, step, data)    # → train_batch(..., training=False)
  for k, v in losses.items():
    _accumulate_loss(running, k, v)
  if phase == "train":
    self._total_tokens += batch["input"].numel() * world_size  # cumulative GLOBAL tokens
  if phase == "train" and should_step_optimizer(...):
    self._global_optim_step += 1
    if validation_interval_steps and step % validation_interval_steps == 0:
      self._validate(epoch, log_step=self._global_optim_step, max_batches=validation_max_batches)
  if max_batches is not None and batch_idx >= max_batches: break   # bounded eval
  if phase == "train":                                  # post-step stamp for on_batch_end consumers
    data["global_optim_step"] = self._global_optim_step  # checkpoint resume position (== scheduler step)
    data["total_tokens"] = self._total_tokens
  callbacks.on_batch_end(step, phase, data)
  if phase == "train" and log_interval > 0 and step % log_interval == 0:
    # live curves + diagnostics: loss_*, perplexity, grad_norm, throughput/*,
    # gpu_mem_*, skipped_steps, optim_step (= _global_optim_step), tokens_consumed
    callbacks.log_metrics(step, phase, batch_metrics)
  if profiler and profiler_step_per == "batch":
    profiler.step()
callbacks.on_epoch_end(epoch, phase)
metrics = materialize_and_average_losses(running, total_batches)
callbacks.log_metrics(log_step or step, phase, metrics)  # epoch-end average (log_step aligns mid-train eval)
if profiler and profiler_step_per == "epoch":
  profiler.step()
```

**`log_interval`** (default 50; 0 disables) controls the per-batch `log_metrics` call. The epoch-end average alone never reaches W&B for never-completing streaming epochs (`validation_interval: 0`, a single unbounded `IterableDataset` epoch), so the live batch losses are pushed every `log_interval` training steps under the same `train/loss_*` keys.

## Single-batch step (`train_batch`, supervised)

```
with torch.set_grad_enabled(training), torch.autocast(device.type, dtype=dtype):
  build_data_container(batch, data=data, model=model, epoch=epoch)
  pred = model(batch["input"], **model_input_args)
  populate_predictions(pred, data)
  if postprocessor: postprocessor(data)
  loss_total, components = call_loss_with_epoch(loss, data, epoch)
  module_losses = collect_module_losses(model, dataset=dataset_if_training, pred=..., target=...)
  batch_loss = loss_total + sum(module_losses.values())

if training:
  scaled = batch_loss / grad_accum_steps
  (scaler.scale(scaled).backward() | scaled.backward())
  if should_step_optimizer(batch_idx, grad_accum_steps, total_batches):
    (scaler.unscale_(optim); clip_grad_norm_(model.parameters(), clip_grad_norm))
    (scaler.step(optim) + scaler.update() | optim.step())
    optim.zero_grad(set_to_none=False)
    if scheduler_step_per == "batch": scheduler.step()

return {"total": batch_loss.detach(), **{k: v.detach() for k, v in components.items()}, **module_losses}
```

## Single-batch step (`train_batch`, GAN)

Three sequential phases inside one autocast scope:

```
Phase 1 — Discriminator on fake:
  with autocast, set_grad_enabled(training):
    fake = model["generator"](batch["input"])
    d_fake = model["discriminator"](fake.detach())
    loss_d_fake = ...
  if training: zero_grad(D); (scaled)backward(loss_d_fake); optim["discriminator"].step()

Phase 2 — Discriminator on real:
  with autocast, set_grad_enabled(training):
    d_real = model["discriminator"](batch["target"])
    loss_d_real = ...
  if training: zero_grad(D); (scaled)backward(loss_d_real); optim["discriminator"].step()

Phase 3 — Generator:
  with autocast, set_grad_enabled(training):
    d_fake_for_g = model["discriminator"](fake)
    loss_g = adversarial + feature_matching + ...
  if training: zero_grad(G); (scaled)backward(loss_g); optim["generator"].step()

return {"generator_loss": ..., "discriminator_loss": ..., ...}
```

The postprocessor runs **inside** autocast (unified with the supervised path).

## Single-batch step (`train_batch`, RL)

The RL flow inverts the driver: `run_epoch` calls `rl_algorithm.collect_rollouts`
(no-grad env.step loop that fills `rollout_buffer`), then
`rl_algorithm.update(trainer=self, ...)`. The algorithm iterates minibatches from
the buffer and dispatches each one back to `trainer.train_batch`. Every minibatch
still flows through the standard autocast → backward → optimizer-step plumbing.

```
with torch.set_grad_enabled(training), torch.autocast(device.type, dtype=dtype):
  build_data_container(batch, data=data, model=model, epoch=epoch)
  evals = policy_wrapper.evaluate_actions(obs=batch_obs, action=batch["rl/action"],
                                          action_info=batch_action_info)
  data["rl/log_prob_new"] = evals["log_prob"]
  data["rl/value_new"]    = evals["value"]
  data["rl/entropy"]      = evals["entropy"]
  batch_loss, components  = self.loss(data)     # LossContainer(PPOClip + Value + Entropy)

if training:
  backward_step(batch_loss, scaler, grad_accum_steps)
  if should_step_optimizer(batch_idx, grad_accum_steps, total_batches):
    optimizer_step(optim=optim, parameters=model.parameters(),
                   scaler=scaler, clip_grad_norm=clip_grad_norm)
return {"total": batch_loss.detach(), **{k: v.detach() for k, v in components.items()}}
```

Buffer batches are keyed to match the data container exactly: `input/<obs>`,
`rl/action`, `rl/log_prob_old`, `rl/value_old`, `rl/advantage`, `rl/return`,
plus `action_info/<key>` for impl-specific tensors the wrapper needs to
recompute log-probs (e.g. a diffusion policy's noise trajectory).

## Export flow (`build_model_exporter`)

```
build_model_exporter(config_file, output_dir, checkpoint, overrides) -> ModelExporter
  1. hydra.compose → DictConfig
     config_file may be:
       experiment/autoencoder/compression_ae   (full experiment config)
       model/vq_vae_transformer                (model-only config)
       export/vq_vae_transformer               (dedicated export config)
  2. initializer.initialize_device(config.device) → device
  3. Trainer._build_model(config, device, use_distributed=False) → model
  4. if config.load_checkpoint_path:
       initializer.load_from_checkpoint(model, optimizer=None, ...) → weights only
  5. model.eval()
  6. return ModelExporter(model, config, device, output_dir)
```

```
ModelExporter.export()
  for each entry in config.export.modules:
    if entry.source_onnx is set:            # ONNX-level extraction
      onnx.utils.extract_model(
        source_path  = output_dir / entry.source_onnx,
        output_path  = output_dir / entry.onnx_filename,
        input_names  = entry.input_tensor,
        output_names = entry.output_tensor,
      )
    else:                                   # PyTorch-level extraction
      module = _get_submodule(model, entry.input_layer) if entry.input_layer else model
      dummy_inputs = zeros(entry.inputs[*].shape, dtype=entry.inputs[*].dtype)
      torch.onnx.export(
        module, dummy_inputs,
        output_dir / entry.onnx_filename,
        opset_version   = config.export.opset_version,
        input_names     = [inp.name for inp in entry.inputs],
        output_names    = [out.name for out in entry.outputs],
        dynamic_axes    = {name: {0: "batch_size"}} if config.export.dynamic_axes,
      )
```

Export configs live in `configs/export/`. `configs/export/default.yaml` is the base
(device, opset_version, dynamic_axes, empty modules list). Dedicated export configs
compose it alongside a model config:

```yaml
defaults:
  - /export/default
  - /model: vq_vae_transformer
  - _self_
```

## Override points for future extension

To add a new mode (e.g. `RL`, `contrastive`, `multi-task`):

1. Subclass `Trainer`.
2. Override `train_batch(..., *, training)`. `eval_batch` inherits automatically as `train_batch(..., training=False)`.
3. Override the subset of `_build_*` classmethods that differ from supervised.
4. Register in `_trainer_class_for(config.mode)` in `builder.py`.

See `rl_trainer.py` for a worked example (RL mode) and `gan_trainer.py` (GAN).

## Numerics invariants

Preserved exactly across the refactor (pinned by `tests/integration/test_golden_training.py` to 1e-5):

1. `set_seeds → instantiate model → .to(device) → channels_last → DDP → optimizer → scheduler → loss → scaler → load_checkpoint → torch.compile`.
2. `torch.autocast` + `torch.set_grad_enabled(training)` wrap forward + loss (not backward).
3. `batch_loss.detach()` accumulated; component losses detached; module losses **not** detached (they flow through backward).
4. `scaled_loss = batch_loss / grad_accum_steps` before `.backward()`.
5. `zero_grad(set_to_none=False)` (non-default).
6. Per-batch scheduler step after `optim.step()` inside the accumulation-boundary branch; per-epoch after `_run_batch_loop`.
7. `collect_module_losses(dataset=... if training else None)` — diffusion datasets update state only during training.
8. `torch.set_float32_matmul_precision("high")` runs once before any matmul.
