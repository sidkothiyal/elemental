# CLAUDE.md — elemental

**elemental** is a modular PyTorch ML training library for image compression autoencoders, classifiers, diffusion models, GANs, masked autoencoders, and autoregressive LMs. Config via [Hydra](https://hydra.cc/); tracking via W&B.

## Tooling

- **Python env**: `uv` (defined in `pyproject.toml`). Always use `uv run` to execute Python.
- **Linting/formatting**: Pre-commit hooks (Black, isort, flake8, pylint, mypy). Run `uv run pre-commit run --all-files` before committing. Don't manually fix style — let the formatters handle it.
- **Deployment**: Docker for remote runs (see `deploy/`).

## Commands

```bash
uv run python src/trainer.py experiment/autoencoder/compression_unet   # train
uv run python src/trainer.py ... lr=0.001 num_epochs=500               # override
torchrun --nproc_per_node=2 src/trainer.py ... distributed=true        # distributed
uv run python src/trainer.py experiment/node/node_train_autoencoder   # mode: graph → GraphTrainer
uv run pytest tests/                                                    # all tests
uv run pytest tests/test_loss.py -k "test_ssim" -v                     # single test
```

## Non-Obvious Patterns

### Hydra Configs
All configs live in `configs/` using hierarchical Hydra composition. Experiment configs in `configs/experiment/` compose sub-configs (dataset, model, loss, optimizer, scheduler, callback) via `defaults:` lists. Entry point: `configs/config.yaml` → `configs/default.yaml`.

### Import Convention
`src/` is on `PYTHONPATH`. Import as `from model.transformer import Transformer` — **never** `from src.model...`.

### Unified Data Container
The trainer passes a single `dict` through every component. Keys: `input/<key>`, `target`, `pred`, `pred/<key>`, `post_processed/<key>`, `model`, `epoch`. **Always read/write via this container** — never pass tensors directly between components.

### GAN Mode
When `mode: "gan"`: model must be `ModuleDict({"generator": ..., "discriminator": ...})`, trainer auto-switches to `batch_loop_gan()`.

## Adding New Components

| Component | Steps |
|-----------|-------|
| **Model** | `src/model/my_model.py` + `configs/model/my_model.yaml` (`_target_: model.my_model.MyModel`) |
| **Dataset** | `src/dataset/my_dataset.py` (extend `BaseDataset`) + `configs/dataset/my_dataset.yaml` |
| **Loss** | `src/loss/my_loss.py` + config with `pred_key`/`target_key` |
| **Callback** | Implement `Callback` interface in `src/callback/` |
| **Post-processor** | Implement `BasePostProcessor.__call__(**kwargs)`, add to `PostProcessorPipeline` |

## Documentation

Hierarchical docs in every `src/` subdirectory:
- `README.md` — features and usage (for users)
- `ARCHITECTURE.md` — implementation details (for developers)

Outermost `README.md` is a GitHub project overview. Outermost `ARCHITECTURE.md` is a directory-to-feature lookup map.

**Update both files in every directory you modify.** Also update the outermost `README.md` and `ARCHITECTURE.md` if changes affect project-level docs. If information is now inaccurate after your change, correct it — outdated docs are worse than no docs.

## Testing

Tests mirror `src/` structure under `tests/`. When adding new functionality, add tests that verify the new behavior — don't add tests for unchanged code. Run only relevant tests during development; full suite before commit.

## Planning

If a task looks like it will have more than 3 steps, create a plan first and get alignment before executing. Break complex work into discrete steps and execute them one by one.
