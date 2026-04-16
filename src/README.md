# src/ — Python Source Root

This is the Python source root (`PYTHONPATH`). Import modules as `from model.transformer import Transformer`.

## Entry Points

### trainer.py — Standard Training Loop
The main entry point for training all model types (autoencoders, classifiers, diffusion, GANs, LMs).

**Usage:**
```bash
uv run python src/trainer.py experiment/autoencoder/compression_unet
uv run python src/trainer.py experiment/classification/imagenet lr=0.001
torchrun --nproc_per_node=2 src/trainer.py experiment/... distributed=true
```

**Features:**
- Trainer class hierarchy in `training/`: `Trainer` (supervised), `GANTrainer`, `RLTrainer`
- Mode dispatch by `config.mode` — CLI calls `build_trainer(...).train()` / `.test()`
- Single-batch forward/backward step factored into `train_batch(..., training=True|False)` so `eval_batch` reuses the same code path with grads disabled
- Unified data container passed through all components (model, loss, postprocessor, callbacks)
- Module-level loss collection — submodules can contribute custom losses via `compute_losses()` hooks
- Distributed training via `torchrun` with NCCL/Gloo backends
- Checkpoint resume with `load_checkpoint_path: "latest"` and `resume_training: true`
- PyTorch profiler integration (CPU/CUDA tracing, TensorBoard export)
- Channels-last memory format and cuDNN benchmark auto-tuning

### graph_trainer.py — Graph-Based Training
An alternative trainer that executes a DAG of nodes with automatic dependency resolution.

**Usage:**
```bash
uv run python src/graph_trainer.py --config-name node_train_autoencoder
```

**Features:**
- Nodes defined in config, instantiated via Hydra with `_target_`
- Automatic topological sort of node dependencies
- `node:<name>.<attr>` reference syntax for inter-node wiring
- Per-epoch execution graph with separate global and epoch-scoped variables

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `model/` | Model architectures (Transformer, ViT, UNet, Diffusion, VQ-VAE, MAE, HF wrappers) |
| `model/blocks/` | Reusable building blocks (conv, resnet, linear, upsample) |
| `dataset/` | Dataset classes (DIV2K, Cityscapes, ImageNet, WikiText, remote) |
| `loss/` | Loss functions and containers |
| `callback/` | Checkpoint, metrics, and W&B callbacks |
| `postprocessor/` | Post-processing pipeline |
| `metrics/` | Evaluation metrics (PSNR, MSE, IoU) |
| `nodes/` | Graph-based trainer nodes |
| `training/` | Trainer hierarchy (`Trainer`, `GANTrainer`, `RLTrainer`), builder, and stateless helpers |
| `utils/` | Device setup, checkpoint, distributed helpers |
| `tools/` | Standalone tools (checkpoint surgery) |
| `data_processor/` | Dataset download/config utilities |
