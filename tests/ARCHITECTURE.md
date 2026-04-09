# Tests Architecture

## Design Principles

### 1. Mirror `src/` Layout

Every file in `src/<submodule>/<file>.py` has a corresponding test at
`tests/src/<submodule>/test_<file>.py`.  This 1:1 mapping lets the
`run_changed.py` script automatically find tests for any changed source file.

### 2. Three Tiers of Tests

```
Unit (tests/src/)         — fast, no I/O, pure PyTorch, mocked externals
Integration (tests/integration/)  — config loading, Hydra instantiation, component wiring
E2E (tests/integration/test_training_runs.py)  — 2-epoch runs on synthetic data
```

### 3. Marker-Controlled Execution

Markers are declared in `tests/conftest.py` and govern what runs by default:

- `unit` + `integration` → default `pytest` run
- `e2e`, `slow`, `hf` → opt-in via `-m` flag

### 4. Changed-File Runner

`tests/run_changed.py` maps `src/**/*.py` → `tests/src/**/test_*.py` via a simple
path transformation. Config changes (`configs/**`) map to
`tests/integration/test_config_loading.py`. This enables pre-commit hooks or
feature-branch CI to run only relevant tests.

## Key Files

| File | Role |
|------|------|
| `tests/conftest.py` | Root fixtures, marker declarations, path setup |
| `tests/integration/conftest.py` | `load_experiment_config()` helper, `config_root` fixture |
| `tests/run_changed.py` | Git-diff-based test selector |

## Fixture Strategy

### Unit Tests
- `data_container` — pre-populated unified data dict matching trainer contract
- `tiny_encoder` / `tiny_decoder` — simple 2-layer conv modules for model composition tests
- `image_batch` / `seg_batch` — ready-to-use random tensors
- `device` — session-scoped, defaults to CPU

### Integration Tests
- `config_root` — session-scoped path to `configs/`
- `clear_hydra` — autouse fixture that clears `GlobalHydra` before/after each test

## Synthetic Data Pattern

Unit and integration tests never touch disk for data. Instead:
- Image tests use `torch.randn(B, C, H, W)`
- Sequence tests use `torch.randint(0, V, (B, T))`
- Dataset tests use small inline `Dataset` subclasses (`_SimpleImageDataset`, `_SequenceDataset`)
- E2E training tests override the dataset `_target_` config key with `_SyntheticImageDataset`

## Diffusion Model Tests

Since commit `1289bc7`, `DenoisingDiffusionModel.forward()` returns a dict:
- Training mode: `{"pred_noise": Tensor, "pred_x0": Tensor}`
- Eval mode: `{"pred_x0": Tensor}`

`tests/src/model/test_diffusion.py` explicitly tests for this dict contract.

## LoRA Tests

`tests/src/model/test_lora.py` creates a tiny `TinyModel` with `q_proj` and
`v_proj` Linear layers (matching the default `LoRAConfig.target_modules`)
rather than wrapping a real HuggingFace model, keeping the test fast and
network-free.

## VLA Module Tests

`tests/src/model/test_vla_modules.py` uses `_FakeVLM` — a tiny `nn.Module`
with the same interface as `HFVLM` but returning fixed-size zero tensors. This
lets `VLAConditioner`, `ActionDenoisingMLP`, and `ActionDenoisingTransformer`
be tested without downloading Qwen3-VL.

## Auto-Discovery Config Tests

`tests/integration/test_config_loading.py` uses glob-based discovery to find
all YAML configs under `configs/`. Adding a new config file automatically
includes it in the test suite — no hardcoded lists to maintain.

Categories tested: experiment, model, dataset, loss, optimizer, scheduler,
callback, profiler.

## Golden Value Tests

Golden value tests pin exact numeric outputs for deterministic inputs to catch
silent behavioral regressions. They exist at three levels:

- **Function-level**: `tests/src/loss/test_loss_golden_values.py` — each loss
  function with `torch.manual_seed(42)` + known inputs.
- **Class-level**: `tests/src/model/test_model_golden_values.py` — model
  forward passes with seeded weights + inputs.
- **Integration-level**: `tests/integration/test_golden_training.py` — full
  training pipeline, loss values captured via `_LossRecorderCallback`.

## Convergence Tests

`tests/integration/test_convergence.py` runs short manual training loops
(no Hydra) and asserts that loss decreases. Uses fixed synthetic data so
models can overfit, proving the training loop works end-to-end.

## Coverage Enforcement

- `pytest-cov` is configured in `pyproject.toml` with `--cov=src` and
  `fail_under = 30`.
- `tests/test_coverage_completeness.py` is a meta-test that globs all
  `src/**/*.py` files and asserts each has a corresponding test file.
  Files that are intentionally untested (downloaders, CLI tools, etc.)
  are listed in `_EXEMPT`.

## Coverage Targets

| Module | Target |
|--------|--------|
| `src/loss/` | 90%+ |
| `src/metrics/` | 90%+ |
| `src/model/blocks/` | 85%+ |
| `src/model/` | 75%+ |
| `src/postprocessor/` | 85%+ |
| `src/callback/` | 70%+ |
| `src/dataset/` | 60%+ |
| `src/utils/` | 70%+ |
| Integration config coverage | 100% (all experiment YAMLs) |
