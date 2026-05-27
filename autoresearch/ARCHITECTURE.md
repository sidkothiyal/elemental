# autoresearch — Architecture

## Directory layout

```
autoresearch/
├── agent.py              # CLI entry point and main loop
├── search_spaces/
│   ├── torch_compile.yaml        # compile mode/fullgraph/dynamic search
│   └── compression_ae_hparams.yaml  # lr/batch/optimizer hparam search
├── runs/                 # created at runtime
│   └── <study>/
│       └── trials.jsonl  # append-only log of trial results
├── README.md
└── ARCHITECTURE.md
```

## Component overview

### `LLMClient`

Thin abstraction over two LLM backends, both lazily imported:

| Provider | SDK | Auth env var | Message format |
|----------|-----|-------------|----------------|
| `anthropic` | `anthropic.Anthropic()` | `ANTHROPIC_API_KEY` | `system` + `messages` (Anthropic native) |
| `openrouter` | `openai.OpenAI(base_url=…)` | `OPENROUTER_API_KEY` | `messages` with `role=system` prefix (OpenAI chat completions) |

`LLMClient.complete(system, user_msg) -> str` normalises both to a single string return.  Lazy imports mean the package does not fail at import time if only one SDK is installed — the error surfaces only when `LLMClient.__init__` is called.

### `_run_trial`

Launches `src/trainer.py` in a subprocess:

```python
subprocess.run(
    ["uv", "run", "python", "src/trainer.py", config, *all_overrides],
    capture_output=True, check=False
)
```

Key design choices:
- **Always subprocess** — `hydra.initialize()` is a global singleton; re-entering it in-process on a second trial would raise `GlobalHydra is already initialized`.
- **Exit-code mapping** — exit codes 10–13 are always added to `fixed_overrides` via the `TrialSummaryCallback` so the agent can distinguish OOM/compile-error/NaN from noise.  If the subprocess crashes before writing `trial_summary.json`, the fallback dict is synthesised from the exit code and stderr.
- **Timestamp-based JSON discovery** — `experiment_name` contains `${now:%H-%M-%S}` so the exact output path isn't known before launch; the agent finds the most-recently modified `trial_summary.json` under `results_dir` after each run.

### Search loop (`main`)

```
for trial_idx in range(len(history), max_trials):
    user_msg = _build_user_message(space_cfg, history[-10:], ...)
    raw_text  = client.complete(_SYSTEM_PROMPT, user_msg)
    overrides = parse_json(raw_text)["overrides"]
    summary   = _run_trial(config, fixed_overrides + overrides, ...)
    history.append(summary)
    append trials.jsonl
```

History is bounded to the last 10 trials in the prompt to keep token usage manageable.  The full history is always preserved in `trials.jsonl` for post-hoc analysis.

### Prompt design

The system prompt gives the LLM three constraints in plain English:

1. Output format: JSON `{"reasoning": "...", "overrides": [...]}`.
2. Domain constraints: no `reduce-overhead + compile_optimizer_step` (known CUDA graph conflict); effective batch ≤ 32.
3. Stop signal: `"overrides": "<STOP>"` to terminate early.

The user message includes the full search space YAML (so the LLM can read parameter types and ranges) and the JSON history (so it can reason about trends).

### `_nested_get`

Resolves dot-separated paths like `"best.val/PSNR.value"` into a nested dict.  Used to extract the objective metric value from `trial_summary.json` for logging and ranking.  Note: keys containing `/` (e.g. `val/PSNR`) are treated as a single segment because the split is on `.` only.

## Integration with `src/`

`autoresearch/` is entirely standalone — it never imports from `src/`. The integration is one-directional:

```
autoresearch/agent.py
    → subprocess: src/trainer.py + Hydra overrides
        → TrialSummaryCallback writes trial_summary.json
    ← reads trial_summary.json
```

The key infra in `src/` that enables this:
- `max_steps` / `max_wall_time_sec` config keys (added to `configs/default.yaml`) for short-budget trials.
- `--no-remote-logging` CLI flag on `src/trainer.py` to skip W&B during trials.
- `TrialSummaryCallback` (opt-in via `+callbacks.trial_summary._target_=callback.trial_summary_callback.TrialSummaryCallback`) for machine-readable JSON output.
- Exit codes 10–13 from `src/trainer.py` for failure classification.
