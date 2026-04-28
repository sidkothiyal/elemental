# Architecture

High-level architecture reference for the elemental ML training library.

## Project Structure

```
src/
├── trainer.py            # Thin Hydra/Typer CLI — delegates to training.builder (handles all modes, incl. graph)
├── exporter.py           # Thin Hydra/Typer CLI — delegates to training.model_exporter for ONNX export
├── training/             # Trainer class hierarchy + builder + helpers
├── callback/             # Lifecycle hooks for training
├── data_processor/       # Dataset download + preprocessing
├── dataset/              # torch.utils.data.Dataset subclasses
├── env/                  # Vectorized RL environments (BaseVecEnv + DummyVecEnv / SimplerEnv)
├── loss/                 # Composable loss modules (incl. PPO / value / entropy for RL)
├── metrics/              # TorchMetrics-based metric modules (incl. episode-metric trackers)
├── model/                # Neural network architectures
│   └── blocks/           # Reusable building blocks
├── nodes/                # Computation graph nodes (GraphTrainer / CustomizableTrainer)
├── postprocessor/        # Data-container post-processing pipeline
├── rl/                   # RL algorithms, policy wrappers, rollout buffer, GAE (PPO today)
├── tools/                # Standalone utility scripts
└── utils/                # Shared helpers (checkpoint, distributed, init)
    └── remote_backup/    # Remote backup strategies
```

## Data Flow: Standard Training Run

```
1. training.builder.build_trainer(config_file) → Trainer
   ├── hydra.compose → DictConfig
   ├── _init_env() → device, dtype, seeds, DDP setup
   ├── initializer.initialize_dataset() → {"train", "val", "test": DataLoader}
   │     └── CUDAPrefetcher wraps each DataLoader
   ├── cls = _trainer_class_for(config.mode)   # Trainer | GANTrainer | RLTrainer | CustomizableTrainer
   │     (mode == "graph" → GraphTrainer via separate init path)
   ├── cls._build_model / _optimizer / _scheduler / _loss / _scaler
   ├── _maybe_load_checkpoint() → checkpoint_data
   ├── _build_postprocessor() / _build_profiler()
   ├── callbacks.update_config(...)
   └── return cls(config=..., model=..., optim=..., ...)

2. Trainer.train()
   for epoch in range(start_epoch, num_epochs):
     run_epoch(epoch)
       ├── cleanup_memory()
       ├── model.train() + _run_batch_loop("train", ...)
       └── (every N epochs) model.eval() + _run_batch_loop("val", ...)

3. Trainer._run_batch_loop(phase, epoch, loader)
   for batch in loader:
     if phase == "train": train_batch(batch, ..., training=True)
     else:                 eval_batch(batch, ...)    # = train_batch(..., training=False)

4. Trainer.train_batch(batch, ..., *, training)
     build_data_container(batch, data)           # namespace batch keys
     model(batch["input"], **model_input_args) → pred
     populate_predictions(pred, data)            # store pred/* keys
     postprocessor_pipeline(data)                # post_processed/*
     loss(data) → (total, components)            # LossContainer
     collect_module_losses(model, dataset, ...)  # submodule losses
     if training: backward() / optimizer.step() / per-batch scheduler
     callbacks.on_batch_end(step, phase, data)
   callbacks.on_epoch_end(epoch, phase)
   callbacks.log_metrics(step, phase, metrics)
```

## Key Architectural Patterns

### 1. Unified Data Container

The trainer builds a single `dict` per batch that flows through the entire pipeline:

```python
data = {
    "input/<key>": tensor,           # batch fields (except "target")
    "target": tensor,                # ground truth
    "model": nn.Module,              # model reference
    "epoch": int,
    "pred": tensor,                  # model output
    "pred/<key>": tensor,            # dict output entries
    "post_processed/<key>": value,   # post-processor outputs
    "model/<key>": value,            # model.loss_args entries
}
```

This allows losses, post-processors, and callbacks to address any data by key without coupling.

### 2. LossContainer / LossComponent

```
LossContainer
  ├── LossComponent(loss_fn=MSELoss(), pred_key="pred", target_key="target")
  ├── LossComponent(loss_fn=SSIMLoss(), weight=0.3)
  └── ... aggregated via sum()
```

Each `LossComponent` resolves its inputs from the data container. Extra context (e.g. mask tensors) is forwarded via `context_mapping`.

### 3. Callback Lifecycle

```
CallbackList dispatches to each Callback:
  update_config() → on_phase_start() → on_epoch_start() →
  on_batch_start() → on_batch_end() → on_epoch_end() →
  log_metrics() → on_lr_update() → on_phase_end()
```

Implemented callbacks: `CheckpointCallback`, `MetricsCallback`, `WandbCallback`.

### 4. PostProcessorPipeline

```
PostProcessorPipeline
  └── list[PostProcessorStep]
        ├── input_mapping: maps data-container keys → processor kwargs
        ├── processor: BasePostProcessor.__call__(**kwargs)
        └── output_key / unfurl_output → writes back to data dict
```

Processors: `ArgmaxPostProcessor`, `ChannelFormatterPostProcessor`, `GeneratePostProcessor`, `RunningTensorStatsPostProcessor`, `TensorMeanStdPostProcessor`, `TensorSlicePostProcessor`.

### 5. Remote Dataset System

**Approach A — Remote filesystem (fsspec)**:
`BaseDataset` → `resolve_path()` transparently downloads/caches from SSH/S3/HTTP.

**Approach B — ZeroMQ dataset server**:
`DatasetServer` (on data machine) serves via ZeroMQ REQ/REP.
`RemoteDataset` (on training machine) fetches samples by index.
Protocol: HANDSHAKE / GETITEM / GETBATCH / PING.

### 6. Graph-Based Trainer (`mode: graph`)

Standalone graph execution trainer for fully declarative training pipelines:
```
GraphTrainer (src/training/graph_trainer.py)
  ├── DependencyResolver (src/nodes/dependency_resolver.py) — topological sort of node configs
  ├── GraphExecutor — executes nodes, manages global/epoch variables
  └── nodes:
        ├── BaseNode — abstract base
        ├── TorchModuleNode — wraps nn.Module
        ├── VariableNode — holds arbitrary values
        ├── GraphNode — nested statement graph
        ├── ForLoopNode — iterator loop
        └── ExecuteFunction — callable wrapper with input/output mapping
```

Configs reference nodes via `node:nodename.attr` strings resolved at runtime.
Dispatched from `build_trainer` when `config.mode == "graph"`.

### 7. Customizable Trainer (`mode: customizable`)

Inherits the full `Trainer` lifecycle (callbacks, profiler, DDP, checkpoint/resume,
scaler, grad accumulation) and overrides `train_batch` to run a user-defined
execution graph per batch. This lets users express non-standard forward/backward
choreography (e.g. GAN three-phase training) declaratively in YAML while reusing
all existing trainer infrastructure.

```
CustomizableTrainer(Trainer) (src/training/customizable_trainer.py)
  ├── self.graph: GraphNode — per-batch execution graph
  ├── _batch_executor: GraphExecutor — scoped to one batch
  └── train_batch() → executes graph with seeded variables (model, optim, loss, batch, ...)
```

Config-driven `metrics_format` mapping customizes metric names. Dispatched from
`build_trainer` when `config.mode == "customizable"`.

### 8. RL Mode (`mode: "rl"`)

Algorithm-, env-, and policy-agnostic RL training that reuses the `Trainer` base for
callbacks, AMP, DDP, checkpointing, and gradient plumbing. Every gradient step still
flows through `train_batch` so the supervised infrastructure is unchanged.

```
RLTrainer(Trainer) (src/training/rl_trainer.py)
  ├── env:            BaseVecEnv           (src/env/base_vec_env.py)
  ├── policy_wrapper: BasePolicyWrapper    (wraps self.model; owns value head + log_std)
  ├── rollout_buffer: OnPolicyRolloutBuffer (src/rl/rollout_buffer.py)
  └── rl_algorithm:   BaseRLAlgorithm     (PPOAlgorithm in src/rl/ppo.py)
```

Per-epoch control flow (`RLTrainer.run_epoch`):
1. **Rollout phase** (no grad): algorithm drives `env.step` for `steps_per_rollout`,
   stores `obs/*`, `action`, `log_prob`, `value`, `reward`, `done`, `action_info/*`
   in the buffer; finalizes with GAE to add `advantage`/`return`.
2. **Update phase**: for `num_policy_epochs × minibatches`, each minibatch is
   dispatched to `trainer.train_batch(...)`. The trainer calls
   `policy_wrapper.evaluate_actions`, populates `rl/log_prob_new`, `rl/value_new`,
   `rl/entropy`, then runs `self.loss(data)` and standard backward/optimizer step.
3. **Sim-eval** every `validation_interval` epochs: deterministic rollout with no
   gradient updates.

RL-only data-container keys live under an `rl/*` namespace and plug into the
standard `LossContainer`/`LossComponent` machinery via `context_mapping` — PPO loss
is a config, not new loss infrastructure. Swapping env (`configs/env/*.yaml`),
algorithm (`configs/rl/*.yaml`), policy wrapper (`configs/policy_wrapper/*.yaml`),
and loss (`configs/loss/ppo.yaml`) are all Hydra edits. The same skeleton drives
VLA+SIMPLER PPO and future LLM/VLM RLHF (GRPO or PPO over an autoregressive policy
wrapper) without `RLTrainer` code changes.

### 9. GAN Mode

When `mode: "gan"`:
- `model` is a `ModuleDict` with `"generator"` and `"discriminator"`
- `optim` and `scheduler` are dicts keyed the same way (auto-built via `build_optim_dict_or_single`)
- `GANTrainer.train_batch` runs a three-phase step (disc-fake → disc-real → generator) in place of the supervised single-phase step
- Can also be expressed declaratively via `mode: customizable` with a GAN graph (see `configs/experiment/gan/customizable_gan.yaml`)

## Export Flow

```
build_model_exporter(config_file, output_dir, checkpoint, overrides) -> ModelExporter
  1. hydra.compose → DictConfig
     config_file accepts three forms:
       experiment/autoencoder/compression_ae   (full experiment config)
       model/vq_vae_transformer                (model-only config)
       export/vq_vae_transformer               (dedicated export config in configs/export/)
  2. Trainer._build_model(config, device, use_distributed=False) → model
  3. initializer.load_from_checkpoint(model, optimizer=None, ...) if checkpoint configured
  4. model.eval()
  5. return ModelExporter(model, config, device, output_dir)

ModelExporter.export()
  for each module entry in config.export.modules:
    if source_onnx set → ONNX-level:  onnx.utils.extract_model(source, output, input_tensors, output_tensors)
    else              → PyTorch-level: _get_submodule(model, input_layer) → torch.onnx.export(...)
```

Export configs live in `configs/export/`. They compose `configs/export/default.yaml` +
a model config via `defaults:`, then add an `export.modules` list. See
`configs/export/vq_vae_transformer.yaml` for a complete example with PyTorch-level and
commented ONNX-level entries.

## Feature-to-Directory Mapping

| Feature | Primary Location |
|---------|-----------------|
| Training loop | `src/training/` (hierarchy); `src/trainer.py` (unified CLI — all modes incl. `mode: graph`) |
| ONNX export | `src/training/model_exporter.py`; `src/exporter.py` (CLI); `configs/export/` |
| Hydra config | `configs/` |
| Model architectures | `src/model/` |
| Building blocks | `src/model/blocks/` |
| Dataset loading | `src/dataset/` |
| Data downloading | `src/data_processor/` |
| Loss functions | `src/loss/` |
| Metrics | `src/metrics/` |
| Checkpointing | `src/callback/checkpoint_callback.py`, `src/utils/checkpoint.py` |
| W&B logging | `src/callback/wandb_callback.py` |
| TorchMetrics integration | `src/callback/metrics_callback.py` |
| Post-processing | `src/postprocessor/` |
| Distributed training | `src/utils/distributed.py` |
| Device/dtype/seed init | `src/utils/initializer.py` |
| Remote backup | `src/utils/remote_backup/` |
| Graph nodes | `src/nodes/` |
| RL algorithms / rollout buffers / GAE | `src/rl/` (PPO + GRPO; `text_rollout_buffer.py` for variable-length completions) |
| Vectorized RL environments | `src/env/` (incl. `prompt_iterator_env.py` for GRPO) |
| Policy wrappers (diffusion / autoregressive) | `src/rl/policy_wrapper.py`, `src/rl/diffusion_policy_wrapper.py`, `src/rl/autoregressive_policy_wrapper.py` |
| RL reward verifiers (GRPO) | `src/rl/verifiers/` (e.g. `gsm8k_verifier.py`) |
| PPO / value / entropy losses | `src/loss/policy_losses.py` |
| GRPO loss | `src/loss/grpo_loss.py` |
| Episode metric trackers | `src/metrics/episode_metrics.py` |
| Rollout callbacks (video, metrics) | `src/callback/rollout_video_callback.py`, `src/callback/rollout_metrics_callback.py` |
| Custom optimizers / schedulers / param-group partition | `src/optimizer/` (Muon, MultiOptimizer, WarmupCosineLR, partition predicates) |
| BPE tokenizer + chat template | `src/data_processor/bpe_tokenizer.py` |
| Token sampling (top-k/p/temperature) | `src/utils/sampling.py` |
| Tokenizer training entry point | `src/tools/train_tokenizer.py` |
| Checkpoint surgery | `src/tools/checkpoint_surgeon.py` |
| Nanochat full pipeline | `configs/experiment/nanochat/{pretrain,sft,grpo_gsm8k}.yaml` |
