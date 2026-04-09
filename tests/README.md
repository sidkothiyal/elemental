# Tests

Comprehensive test suite for the Elemental ML training library.

## Structure

```
tests/
├── conftest.py              # Root fixtures: markers, device, data containers, tiny encoders/decoders
├── run_changed.py           # Run only tests for files changed vs a git branch
├── src/                     # Unit tests — mirrors src/ layout exactly
│   ├── callback/            # Callback lifecycle, checkpoint saving, metrics, W&B
│   ├── data_processor/      # Tokenizer and preprocessing utilities
│   ├── dataset/             # DiffusionDataset, FramePredictionDataset, BaseDataset, serialization
│   ├── loss/                # All loss functions: container, phased, weighted, masked, SSIM, etc.
│   ├── metrics/             # BaseMetric, PSNR, MSE, SSIM, IoU, action metrics
│   ├── model/
│   │   ├── blocks/          # ConvNormAct, ResNetBlock, LinearBlock, UpsampleNormAct, ConditionalConvBlock
│   │   ├── test_autoencoder.py
│   │   ├── test_cached_module.py
│   │   ├── test_combiner.py
│   │   ├── test_diffusion.py
│   │   ├── test_frame_predictor.py
│   │   ├── test_lora.py
│   │   ├── test_model_utils.py
│   │   ├── test_patcher_unpatcher.py
│   │   ├── test_positional_embeddings.py
│   │   ├── test_quantizer.py
│   │   ├── test_timestep_embedding.py
│   │   ├── test_transformer.py
│   │   ├── test_unet.py
│   │   └── test_vla_modules.py
│   ├── nodes/               # GraphExecutor, VariableNode
│   ├── postprocessor/       # Pipeline, argmax, channel formatter, tensor slice
│   └── utils/               # Checkpoint I/O, device utils, numpy↔tensor
├── integration/             # Integration tests — config loading, instantiation
│   ├── test_config_loading.py       # Auto-discovered experiment + component configs
│   ├── test_loss_configs.py         # All loss YAMLs instantiate + forward
│   ├── test_model_instantiation.py  # Lightweight model YAMLs instantiate + forward
│   ├── test_training_runs.py        # E2E: 2-epoch runs on synthetic data
│   ├── test_convergence.py          # E2E: verify loss decreases during training
│   └── test_golden_training.py      # E2E: pin training loss values for regression detection
├── test_coverage_completeness.py    # Meta-test: every src/ module has a test file
└── fixtures/                # (empty) placeholder for future static test fixtures
```

## Running Tests

### Default (fast) — all unit tests, no HF downloads

```bash
uv run pytest tests/src/
```

### Integration tests only (config loading, component wiring)

```bash
uv run pytest tests/integration/ -m integration
```

### E2E training runs (slow)

```bash
uv run pytest tests/integration/test_training_runs.py -m e2e
```

### Single submodule

```bash
uv run pytest tests/src/model/
uv run pytest tests/src/loss/
uv run pytest tests/src/dataset/
```

### By marker

```bash
uv run pytest -m unit           # all fast unit tests (default CI)
uv run pytest -m integration    # config loading / wiring tests
uv run pytest -m "e2e or slow"  # training runs
uv run pytest -m hf             # tests requiring HF model downloads
uv run pytest -m golden         # All golden tests (loss + model, runs in default suite)
```

### Convergence tests — verify training actually reduces loss

```bash
uv run pytest tests/integration/test_convergence.py -m e2e -v
```

These run short manual training loops (no Hydra) and assert that loss
decreases over multiple steps. Useful for verifying the optimizer and
backprop pipeline work end-to-end.

### Coverage — check test completeness

Coverage is **not** included in the default `pytest` run. Use `--cov` explicitly:

```bash
# Run tests with line-level coverage report
uv run pytest tests/ --cov=src --cov-report=term-missing --cov-config=pyproject.toml

# HTML report for detailed browsing
uv run pytest tests/ --cov=src --cov-report=html --cov-config=pyproject.toml

# Run the meta-test that checks every src/ module has a test file
uv run pytest tests/test_coverage_completeness.py -v
```

Coverage thresholds are configured in `pyproject.toml` under `[tool.coverage.report]`
(`fail_under = 30`). The completeness meta-test reports all `src/` modules
missing a corresponding test file. Add intentionally untested files to
`_EXEMPT` in `test_coverage_completeness.py`.

### Changed-file runner (fastest — only tests relevant to your changes)

```bash
# Compare HEAD against main (default)
uv run python tests/run_changed.py

# Compare HEAD against previous commit
uv run python tests/run_changed.py HEAD~1
```

## Markers

| Marker | Meaning | Included in default `pytest`? |
|--------|---------|------------------------------|
| `unit` | Fast unit tests, no I/O | Yes |
| `integration` | Config loading and component wiring | Yes |
| `e2e` | End-to-end short training runs | No (`-m e2e` required) |
| `slow` | Tests taking > 10 s | No |
| `hf` | Requires HuggingFace model downloads | No |
| `golden` | Pins exact numeric outputs for regression detection | Yes |

## Adding a New Test

1. Create `tests/src/<submodule>/test_<file>.py` to mirror `src/<submodule>/<file>.py`.
2. Mark every test class or function with at least one marker (`@pytest.mark.unit`).
3. Use synthetic tensors — never depend on real data files in unit tests.
4. Add the corresponding entry to `tests/run_changed.py` mapping if needed.
5. Config tests auto-discover YAML files — no need to update any hardcoded list.
   New configs are automatically tested when added to `configs/`.

## Conventions

- No network calls in unit or integration tests (mock or skip with `@pytest.mark.hf`).
- Use `pytest.approx` for all float comparisons.
- Use `tmp_path` fixture for file I/O.
- Each test file has a module docstring explaining what it covers.
