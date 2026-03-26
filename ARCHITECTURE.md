# Architecture

High-level architecture reference for the elemental ML training library.

## Project Structure

```
src/
├── trainer.py            # Standard training loop (Hydra + fire)
├── graph_trainer.py      # Node-graph alternative trainer (Hydra @hydra.main)
├── callback/             # Lifecycle hooks for training
├── data_processor/       # Dataset download + preprocessing
├── dataset/              # torch.utils.data.Dataset subclasses
├── loss/                 # Composable loss modules
├── metrics/              # TorchMetrics-based metric modules
├── model/                # Neural network architectures
│   └── blocks/           # Reusable building blocks
├── nodes/                # Computation graph nodes (graph_trainer)
├── postprocessor/        # Data-container post-processing pipeline
├── tools/                # Standalone utility scripts
└── utils/                # Shared helpers (checkpoint, distributed, init)
    └── remote_backup/    # Remote backup strategies
```

## Data Flow: Standard Training Run

```
1. trainer.run_train(config_file)
   ├── hydra.compose → DictConfig
   ├── initializer.initialize_dataset() → (train_loader, val_loader, test_loader)
   │     └── CUDAPrefetcher wraps each DataLoader
   ├── hydra.utils.instantiate(config.model) → nn.Module
   ├── optimizer / scheduler / loss instantiation
   ├── callbacks.update_config(...)
   └── Training loop:
         for epoch in range(num_epochs):
           cleanup_memory()
           model.train()
           batch_loop("train", ...)
           batch_loop("val", ...)

2. batch_loop(phase, epoch, dataloader, ...)
   for batch in dataloader:
     build_data_container(batch, data)           # namespace batch keys
     model(batch["input"], **model_input_args) → pred
     populate_predictions(pred, data)            # store pred/* keys
     postprocessor_pipeline(data)               # post_processed/*
     loss(data) → (total, components)           # LossContainer
     collect_module_losses(model, dataset, ...)  # submodule losses
     backward() / optimizer.step()
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

### 6. Graph-Based Trainer

Alternative to the standard trainer for complex pipelines:
```
GraphTrainer
  ├── DependencyResolver — topological sort of node configs
  ├── GraphExecutor — executes nodes, manages global/epoch variables
  └── nodes:
        ├── BaseNode — abstract base
        ├── TorchModuleNode — wraps nn.Module
        ├── VariableNode — holds arbitrary values
        ├── GraphNode — nested statement graph
        └── ForLoopNode — iterator loop
```

Configs reference nodes via `node:nodename.attr` strings resolved at runtime.

### 7. GAN Mode

When `mode: "gan"`:
- `model` is a `ModuleDict` with `"generator"` and `"discriminator"`
- `optim` and `scheduler` are dicts keyed the same way
- `batch_loop_gan()` is used instead of `batch_loop()`

## Feature-to-Directory Mapping

| Feature | Primary Location |
|---------|-----------------|
| Training loop | `src/trainer.py` |
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
| Checkpoint surgery | `src/tools/checkpoint_surgeon.py` |
