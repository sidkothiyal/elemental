# autoresearch — LLM-driven experiment search for elemental

Automated propose → trial → observe loop that uses an LLM to search over:

1. **torch.compile settings** — finds the mode/backend/fullgraph/dynamic/compile_optimizer_step combination that maximises training throughput (`perf.median_step_ms`).
2. **Hyperparameters** — searches lr, batch size, grad accumulation, weight decay, etc. to maximise reconstruction quality (`best.val/PSNR`).

The search is model-agnostic: any Anthropic model or any model available on [OpenRouter](https://openrouter.ai/models) can drive it.

## Quick start

### 1. Install dependencies

```bash
uv sync --group autoresearch
```

Set your API key depending on the provider:

```bash
# Anthropic (default)
export ANTHROPIC_API_KEY=sk-ant-...

# OpenRouter
export OPENROUTER_API_KEY=sk-or-...
```

### 2. Run a torch.compile sweep

```bash
uv run python -m autoresearch.agent \
  --config experiment/autoencoder/compression_ae \
  --space autoresearch/search_spaces/torch_compile.yaml \
  --max-trials 16
```

### 3. Run a hyperparameter search

```bash
uv run python -m autoresearch.agent \
  --config experiment/autoencoder/compression_ae \
  --space autoresearch/search_spaces/compression_ae_hparams.yaml \
  --max-trials 24
```

### 4. Use a different LLM (OpenRouter)

```bash
uv run python -m autoresearch.agent \
  --config experiment/autoencoder/compression_ae \
  --space autoresearch/search_spaces/torch_compile.yaml \
  --provider openrouter \
  --model google/gemini-2.5-flash \
  --max-trials 16

# Or DeepSeek R1 for stronger reasoning
uv run python -m autoresearch.agent \
  --provider openrouter \
  --model deepseek/deepseek-r1 \
  --config experiment/autoencoder/compression_ae \
  --space autoresearch/search_spaces/compression_ae_hparams.yaml
```

## CLI reference

```
Options:
  --config       TEXT     Hydra config name [required]
  --space        PATH     Search space YAML [required]
  --provider     TEXT     'anthropic' or 'openrouter'  [default: anthropic]
  --model        TEXT     Model ID (provider-specific; see defaults below)
  --max-trials   INT      Maximum trials to run          [default: 16]
  --study        TEXT     Study name (auto-generated if blank)
  --results-dir  TEXT     Base results directory         [default: results/autoencoder]
```

Default models:

| Provider | Default model |
|----------|--------------|
| `anthropic` | `claude-sonnet-4-6` |
| `openrouter` | `google/gemini-2.5-flash` |

Any [OpenRouter model slug](https://openrouter.ai/models) is accepted for `--model` when using `--provider openrouter`.

## How it works

Each trial:
1. The LLM receives the search space definition + the history of all previous trials (last 10 summaries).
2. It responds with a JSON object: `{"reasoning": "...", "overrides": ["lr=0.001", ...]}`.
3. The agent launches `src/trainer.py` in a subprocess with those overrides plus `max_steps` and `--no-remote-logging` from the search space's `fixed_overrides`.
4. The subprocess writes `trial_summary.json` (via `TrialSummaryCallback`) containing status, metrics, and throughput data.
5. The summary is appended to `autoresearch/runs/<study>/trials.jsonl`.
6. Repeat until `max_trials` or until the LLM emits `"overrides": "<STOP>"`.

## Output

```
autoresearch/runs/<study>/
└── trials.jsonl       # one JSON line per trial
```

Each line contains the full `trial_summary.json` plus `trial_idx`, `agent_overrides`, `agent_reasoning`, `llm_provider`, and `llm_model`.

At the end the agent prints a ranked summary sorted by the objective metric.

## Adding a new search space

Create a YAML file under `autoresearch/search_spaces/` following this schema:

```yaml
name: my_search
objective: maximise           # or minimise
metric: "best.val/PSNR.value" # dot-separated path into trial_summary.json
constraint: "status == 'ok'"

fixed_overrides:              # applied to every trial
  - max_steps=300
  - --no-remote-logging

parameters:
  lr:
    type: float_log
    low: 1.0e-5
    high: 1.0e-2
  batch_size:
    type: categorical
    values: [2, 4, 8]
```

The LLM receives this YAML verbatim as context; it does not enforce the schema mechanically — it reads the comments and proposes overrides accordingly.

## Environment variables

| Variable | Required for |
|----------|-------------|
| `ANTHROPIC_API_KEY` | `--provider anthropic` |
| `OPENROUTER_API_KEY` | `--provider openrouter` |
