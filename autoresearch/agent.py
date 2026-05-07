"""LLM-driven autoresearch agent for elemental.

Runs a propose → trial → observe loop using any supported LLM to search over
torch.compile settings or hyperparameters.

Supported providers
-------------------
anthropic  (default) — uses the Anthropic SDK; requires ANTHROPIC_API_KEY.
openrouter            — uses the OpenAI-compatible API at openrouter.ai;
                        requires OPENROUTER_API_KEY; accepts any model ID
                        that OpenRouter hosts (e.g. google/gemini-2.5-flash,
                        meta-llama/llama-3.3-70b-instruct, deepseek/r1, …).

Usage
-----
    # Anthropic (default)
    uv run --group autoresearch python -m autoresearch.agent \\
        --config experiment/autoencoder/compression_ae \\
        --space autoresearch/search_spaces/torch_compile.yaml \\
        --max-trials 16

    # OpenRouter
    uv run --group autoresearch python -m autoresearch.agent \\
        --config experiment/autoencoder/compression_ae \\
        --space autoresearch/search_spaces/compression_ae_hparams.yaml \\
        --provider openrouter \\
        --model google/gemini-2.5-flash \\
        --max-trials 24

Each trial:
  1. Renders the search space + history of past trial summaries into a prompt.
  2. Asks the LLM for the next set of Hydra overrides to try.
  3. Runs the trainer in a subprocess (short-budget, --no-remote-logging).
  4. Reads trial_summary.json and appends it to trials.jsonl.
  5. Repeats until max-trials is reached or the model emits "<STOP>".

Results are written to autoresearch/runs/<study>/trials.jsonl.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import hydra
import typer
import yaml
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

app = typer.Typer(pretty_exceptions_enable=False)
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent
_RUNS_DIR = _REPO_ROOT / "autoresearch" / "runs"
_CONFIG_PATH = Path(__file__).parent / "config.yaml"

# Hydra override that injects TrialSummaryCallback into any experiment config.
# Config-group override syntax (+/callback@...) is invalid in Hydra's override
# grammar; direct value injection is the only command-line approach that works.
_TRIAL_SUMMARY_OVERRIDE = (
    "+callbacks.trial_summary._target_=callback.trial_summary_callback.TrialSummaryCallback"
)


def _load_autoresearch_config() -> dict:
    """Load autoresearch/config.yaml, returning an empty dict on failure."""
    if not _CONFIG_PATH.exists():
        return {}
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_SYSTEM_PROMPT = """\
You optimize torch.compile + batch-size settings for an image compression autoencoder.
Each turn you propose Hydra overrides for the next short training trial; you then \
receive a JSON summary back. You minimize perf.ms_per_sample (lower is better).

OUTPUT FORMAT — STRICT:
- Reply with EXACTLY one JSON object. No prose before or after. No markdown fences.
- Put your reasoning INSIDE the "reasoning" field (1–3 sentences). Do NOT think out \
loud before the JSON; the JSON IS your answer.
- Schema: {"reasoning": str, "overrides": list[str] | "<STOP>", \
"model_suggestions": list[str] (optional)}
- Example: {"reasoning": "Try max-autotune with shape padding to amortize compile \
over 200 steps.", "overrides": ["batch_size=8", "gradient_accumulation_steps=4", \
"torch_compile.mode=max-autotune", "+torch_compile.options.shape_padding=true"]}
- Use "<STOP>" only when further search is hopeless.

CONSTRAINTS:
- batch_size * gradient_accumulation_steps <= 32.
- Never set torch_compile.mode=reduce-overhead together with compile_optimizer_step=true.
- torch_compile.options.* keys go through torch.compile(options={...}); set them \
only when the mode supports them (e.g. coordinate_descent_tuning ∈ max-autotune*).
- New keys (anything under torch_compile.options.*) need the leading "+" in the \
override string, e.g. "+torch_compile.options.shape_padding=true".

OPTIONAL "model_suggestions": tiny, behavior-preserving edits to the model that \
remove graph breaks or improve compile fusion (e.g. replace .item()/Python-if-on-tensor \
with torch.where, avoid in-place on views). Advisory only — logged, never auto-applied.
"""


# ---------------------------------------------------------------------------
# Multi-provider LLM client
# ---------------------------------------------------------------------------


class LLMClient:  # pylint: disable=too-few-public-methods
    """Thin abstraction over Anthropic SDK and OpenRouter (OpenAI-compatible).

    Anthropic SDK is used for Anthropic models; the openai package (with a
    custom base_url) is used for OpenRouter.  Both paths are imported lazily
    so the package does not fail at import time if only one SDK is installed.
    """

    def __init__(self, provider: str, models: list[str]) -> None:
        self.provider = provider
        self.models = models
        self.model = models[0]
        self.last_used_model = models[0]

        if provider == "anthropic":
            try:
                import anthropic  # pylint: disable=import-outside-toplevel

                self._client = anthropic.Anthropic()
            except ImportError as exc:
                raise ImportError(
                    "anthropic package is required for provider='anthropic'. "
                    "Install with: uv sync --group autoresearch"
                ) from exc

        elif provider == "openrouter":
            try:
                import openai  # pylint: disable=import-outside-toplevel

                api_key = os.environ.get("OPENROUTER_API_KEY")
                if not api_key:
                    raise EnvironmentError("OPENROUTER_API_KEY environment variable is not set.")
                self._client = openai.OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key,
                    default_headers={"HTTP-Referer": "https://github.com/elemental"},
                )
            except ImportError as exc:
                raise ImportError(
                    "openai package is required for provider='openrouter'. "
                    "Install with: uv sync --group autoresearch"
                ) from exc

        else:
            raise ValueError(f"Unknown provider '{provider}'. Supported: anthropic, openrouter")

    def complete(self, system: str, user_msg: str, max_tokens: int = 4096) -> str:
        """Return the model's text response, trying fallback models on failure.

        Output is forced into a JSON object two ways:
          * OpenRouter: ``response_format={"type": "json_object"}`` (server-side).
          * Anthropic: prefill the assistant turn with ``{`` so the response must
            continue from there.
        Combined, this keeps chatty / "thinking" models from spending their token
        budget on prose and never reaching the JSON.
        """
        last_exc: Exception | None = None
        for model in self.models:
            try:
                result = self._complete_with(model, system, user_msg, max_tokens)
                self.last_used_model = model
                return result
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Model %s failed (%s); trying next fallback.", model, exc)
                last_exc = exc
        raise RuntimeError(f"All models failed. Last error: {last_exc}") from last_exc

    def _complete_with(self, model: str, system: str, user_msg: str, max_tokens: int) -> str:
        if self.provider == "anthropic":
            # Prefill "{" — Anthropic continues from the partial assistant turn,
            # so the response is guaranteed to start as a JSON object.
            response = self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": "{"},
                ],
            )
            return ("{" + response.content[0].text).strip()

        # openrouter — OpenAI-compatible chat completions.
        # response_format={"type":"json_object"} is widely supported on OpenRouter and
        # silently ignored by models that don't support it (the JSON-extraction fallback
        # in main() handles those).
        response = self._client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
        )
        return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Trial execution
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_override_value(overrides: list[str], key: str) -> str | None:
    """Return the rightmost value assigned to ``key`` in a list of Hydra overrides.

    Handles both ``key=value`` and ``+key=value`` forms; ignores flag-style entries
    (e.g. ``--no-remote-logging``). Last write wins.
    """
    found: str | None = None
    for ov in overrides:
        if "=" not in ov or ov.startswith("--"):
            continue
        lhs, _, rhs = ov.partition("=")
        lhs = lhs.lstrip("+~")
        if lhs == key:
            found = rhs
    return found


def _resolve_int_override(overrides: list[str], key: str, default: int) -> int:
    raw = _parse_override_value(overrides, key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Could not parse override %s=%r as int; using default %d", key, raw, default)
        return default


def _inject_per_sample_perf(
    summary: dict[str, Any],
    overrides: list[str],
    defaults: dict[str, int],
) -> None:
    """Compute and inject ``perf.ms_per_sample`` and ``perf.effective_batch_size``.

    ``effective_batch_size`` = ``batch_size * gradient_accumulation_steps``, where the
    values come from agent overrides if present, else from the resolved-config defaults.
    ``ms_per_sample`` = ``median_step_ms / effective_batch_size`` when both are known.
    """
    perf = summary.setdefault("perf", {})
    bs = _resolve_int_override(overrides, "batch_size", defaults.get("batch_size", 1))
    ga = _resolve_int_override(
        overrides, "gradient_accumulation_steps", defaults.get("gradient_accumulation_steps", 1)
    )
    eff = bs * ga
    perf["batch_size"] = bs
    perf["gradient_accumulation_steps"] = ga
    perf["effective_batch_size"] = eff
    median_step_ms = perf.get("median_step_ms")
    if isinstance(median_step_ms, (int, float)) and eff > 0:
        perf["ms_per_sample"] = round(median_step_ms / eff, 4)


def _run_trial(
    config_file: str,
    fixed_overrides: list[str],
    agent_overrides: list[str],
    results_dir: str,
    defaults: dict[str, int],
) -> dict[str, Any]:
    """Run one training trial in a subprocess and return the trial_summary dict."""
    all_overrides = fixed_overrides + agent_overrides
    cmd = [
        "uv",
        "run",
        "python",
        str(_REPO_ROOT / "src" / "trainer.py"),
        config_file,
        *all_overrides,
    ]
    logger.info("Running: %s", " ".join(cmd))

    t0 = time.perf_counter()
    t0_wall = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_REPO_ROOT), check=False)
    wall_time = time.perf_counter() - t0

    logger.info("Trial finished in %.1fs — exit code %d", wall_time, result.returncode)
    if result.returncode not in (0, 10, 11, 12, 13):
        # Full stderr stays in the debug log only; stdout-level handler is at
        # WARNING+, so emitting at INFO keeps the noisy Inductor key dumps off
        # the user's terminal.
        logger.info(
            "Unexpected exit code %d; stderr (last 2k):\n%s",
            result.returncode,
            result.stderr[-2000:],
        )

    # Locate this trial's trial_summary.json. results_dir accumulates files from
    # earlier runs (and earlier trials in this study), so we must restrict to
    # files written after this subprocess started — otherwise a trial that fails
    # before the trainer creates its own summary will silently inherit the
    # previous trial's metrics.
    summary_files = sorted(
        (
            p
            for p in Path(results_dir).rglob("trial_summary.json")
            if p.stat().st_mtime >= t0_wall - 1.0
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not summary_files:
        status_map = {10: "oom", 11: "compile_error", 12: "nan", 13: "timeout"}
        status = status_map.get(result.returncode, "error")
        return {
            "status": status,
            "error": result.stderr[-2000:] if result.stderr else "no trial_summary.json found",
            "final": {},
            "best": {},
            "perf": {},
            "config_overrides": agent_overrides,
        }

    summary: dict[str, Any] = json.loads(summary_files[0].read_text())
    _inject_per_sample_perf(summary, agent_overrides, defaults)
    return summary


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def _failure_hint(err: str) -> str | None:
    """One-line plain-English diagnosis for common Hydra/trainer failure modes."""
    if not err:
        return None
    if "Ambiguous value" in err and "requires --multirun" in err:
        return (
            "Hydra rejected a comma-separated value. Wrap as a list "
            "(key=[a,b,c]) or quote the string (key='a,b,c')."
        )
    if "is not in struct" in err or "Could not override" in err:
        return "Override key is unknown to the schema. New keys need a leading '+'."
    if "ConfigKeyError" in err:
        return "Config key not found. Check the key path and whether it needs '+' or '~'."
    # torch._inductor.config rejects unknown keys by enumerating valid ones —
    # the stderr ends with a long list of names like 'test_configs.*'.
    if "test_configs.bisect_keep_custom_backend_for_inductor" in err or (
        "_save_config_ignore" in err and "test_configs." in err
    ):
        return (
            "torch._inductor.config rejected an unknown key. One of the "
            "+torch_compile.options.* overrides is not valid in this PyTorch "
            "version (likely 'triton.cudagraphs' or another nested key) — "
            "drop it and retry."
        )
    if "CUDAGraphs that has been overwritten" in err:
        return (
            "CUDAGraphs tensor-overwrite during training. Switch to "
            "torch_compile.mode=max-autotune-no-cudagraphs (or set "
            "+torch_compile.options.triton.cudagraphs=false)."
        )
    if "CUDA out of memory" in err or "out of memory" in err.lower():
        return "Out of memory — reduce batch_size or effective batch size."
    return None


def _summarise_trial_for_history(trial: dict) -> dict:
    """Return a slimmed-down trial record for inclusion in the LLM prompt."""
    perf = trial.get("perf", {}) or {}
    err = trial.get("error") or ""
    return {
        "trial_idx": trial.get("trial_idx"),
        "status": trial.get("status"),
        "error": err[-1500:] if err else None,
        "failure_hint": _failure_hint(err),
        "agent_overrides": trial.get("agent_overrides", []),
        "perf": {
            "median_step_ms": perf.get("median_step_ms"),
            "ms_per_sample": perf.get("ms_per_sample"),
            "effective_batch_size": perf.get("effective_batch_size"),
            "peak_mem_gb": perf.get("peak_mem_gb"),
            "compile_warmup_ms": perf.get("compile_warmup_ms"),
        },
        "best": trial.get("best", {}),
    }


_TUNING_HINTS = (
    "Graph-break risks to scan for in the model source: control-flow on tensor "
    "values, .item()/.cpu()/.tolist()/.numpy() inside forward, in-place ops on "
    "views, calls into non-torch libs, custom autograd Functions. Either tune "
    "compile flags around them or emit a model_suggestions entry."
)


def _extract_json_object(text: str) -> dict | None:
    """Best-effort: return the first balanced top-level ``{...}`` block as a dict.

    Tolerates models that wrap their JSON in prose or markdown fences. Returns
    ``None`` if no parseable JSON object is found.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break  # try next start
        start = text.find("{", start + 1)
    return None


def _build_user_message(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    space: dict,
    history: list[dict],
    trial_idx: int,
    max_trials: int,
    experiment_config_yaml: str,
    model_source: str,
    model_source_path: str,
) -> str:
    """Render the per-trial user message.

    Heavy context (resolved config + model source + tuning hints) is sent only on
    trial 0 so that it lands in the prompt cache. Subsequent trials get a compact
    history-only message — keeps the prompt small enough that thinking-style models
    can't burn their token budget on prose.
    """
    lines = [
        f"## Trial {trial_idx + 1} / {max_trials}",
        f"### Objective: {space['objective']} `{space['metric']}`  "
        f"(constraint: {space.get('constraint', 'none')})",
        "",
        "### Search space",
        f"```yaml\n{yaml.dump(space.get('parameters', {}), default_flow_style=False)}```",
    ]

    if trial_idx == 0:
        lines += [
            "",
            "### Resolved experiment config (one-time context)",
            f"```yaml\n{experiment_config_yaml}```",
            "",
            f"### Model source — `{model_source_path}` (one-time context)",
            f"```python\n{model_source}```",
            "",
            f"### Tuning hints\n{_TUNING_HINTS}",
        ]

    lines.append("")
    if history:
        slim = [_summarise_trial_for_history(t) for t in history[-10:]]
        lines += [
            "### Trial history (most recent last)",
            "```json",
            json.dumps(slim, indent=2),
            "```",
        ]
    else:
        lines.append("No trials run yet — propose the first configuration to try.")

    lines.append("")
    lines.append("Reply with the JSON object now. No prose, no markdown fences.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def _nested_get(d: dict, key: str) -> Any:
    """Resolve a dot-separated key like 'best.val/PSNR.value' into d."""
    cur: Any = d
    for part in key.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _print_ranking(history: list[dict], space_cfg: dict) -> None:  # pylint: disable=too-many-locals
    metric_key = space_cfg["metric"]
    objective = space_cfg.get("objective", "maximise")
    valid = [
        t for t in history if t.get("status") == "ok" and _nested_get(t, metric_key) is not None
    ]
    if not valid:
        logger.info("No successful trials to rank.")
        return

    reverse = objective == "maximise"
    ranked = sorted(valid, key=lambda t: _nested_get(t, metric_key), reverse=reverse)

    show_perf_cols = metric_key.startswith("perf.")
    separator = "=" * 100
    print(f"\n{separator}")
    print(f"Ranking by {metric_key} ({objective})")
    print(separator)
    if show_perf_cols:
        print(f"  {'#':>3}  {'metric':>10}  {'step_ms':>8}  {'eff_bs':>6}  overrides")
    for rank, trial in enumerate(ranked, 1):
        val = _nested_get(trial, metric_key)
        overrides = trial.get("agent_overrides", [])
        if show_perf_cols:
            perf = trial.get("perf", {}) or {}
            step_ms = perf.get("median_step_ms")
            eff_bs = perf.get("effective_batch_size")
            step_ms_s = f"{step_ms:.4g}" if isinstance(step_ms, (int, float)) else "-"
            eff_bs_s = f"{eff_bs}" if eff_bs is not None else "-"
            print(f"  #{rank:2d}  {val:>10.4g}  {step_ms_s:>8}  {eff_bs_s:>6}  {overrides}")
        else:
            print(f"  #{rank:2d}  {val:.4g}  {overrides}")
    print(f"{separator}\n")


# ---------------------------------------------------------------------------
# Experiment + model-source resolution (sent as static context to the LLM)
# ---------------------------------------------------------------------------


def _resolve_experiment_context(  # pylint: disable=too-many-locals
    config_file: str,
    fixed_overrides: list[str],
) -> tuple[str, str, str, dict[str, int]]:
    """Compose the experiment Hydra config and read the model source file.

    Returns (experiment_yaml, model_source, model_source_path, defaults) where
    defaults is a dict with ``batch_size`` and ``gradient_accumulation_steps`` taken
    from the resolved config (used to compute ``ms_per_sample`` when the agent
    doesn't override them).
    """
    sys.path.insert(0, str(_REPO_ROOT / "src"))

    # Strip flag-style entries (e.g. --no-remote-logging) — those are Typer flags
    # consumed by src/trainer.py, not Hydra overrides.
    hydra_overrides = [o for o in fixed_overrides if not o.startswith("--")]

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    config_path = str(_REPO_ROOT / "configs")
    with hydra.initialize_config_dir(version_base=None, config_dir=config_path):
        cfg = hydra.compose(config_name=config_file, overrides=hydra_overrides)

    experiment_yaml = OmegaConf.to_yaml(cfg, resolve=False)

    # Resolve model source file from cfg.model._target_.
    model_target = OmegaConf.select(cfg, "model._target_") or ""
    model_source = ""
    model_source_path = ""
    if model_target:
        # e.g. "model.residual_autoencoder.ResidualAutoencoder" → src/model/residual_autoencoder.py
        module_path = model_target.rsplit(".", 1)[0]
        candidate = _REPO_ROOT / "src" / Path(*module_path.split("."))
        candidate_py = candidate.with_suffix(".py")
        if candidate_py.exists():
            model_source_path = str(candidate_py.relative_to(_REPO_ROOT))
            text = candidate_py.read_text(encoding="utf-8")
            src_lines = text.splitlines()
            limit = 250
            if len(src_lines) > limit:
                model_source = (
                    "\n".join(src_lines[:limit])
                    + f"\n... (truncated, {len(src_lines)} lines total)"
                )
            else:
                model_source = text
        else:
            model_source_path = f"<not found: {candidate_py}>"
            model_source = f"# Model source not found at {candidate_py}"

    defaults = {
        "batch_size": int(OmegaConf.select(cfg, "batch_size") or 1),
        "gradient_accumulation_steps": int(
            OmegaConf.select(cfg, "gradient_accumulation_steps") or 1
        ),
    }
    return experiment_yaml, model_source, model_source_path, defaults


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _setup_logging(run_dir: Path) -> Path:
    """Route verbose logs to ``run_dir/debug.log``; keep stdout at WARNING+.

    All existing ``logger.*`` calls land in the debug file; clean per-trial
    progress is emitted via plain ``print()`` from ``main()``.
    """
    debug_log = run_dir / "debug.log"
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(debug_log, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.addHandler(console_handler)

    return debug_log


@app.command()
def main(  # pylint: disable=too-many-locals,too-many-arguments,too-many-branches
    # pylint: disable=too-many-positional-arguments,too-many-statements
    config: str = typer.Option(
        ..., help="Hydra config name (e.g. experiment/autoencoder/compression_ae)"
    ),
    space: Path = typer.Option(..., help="Path to search space YAML"),
    provider: str = typer.Option(
        "",
        help=(
            "LLM provider: 'anthropic' or 'openrouter'. "
            "Falls back to autoresearch/config.yaml when omitted."
        ),
    ),
    model: str = typer.Option(
        "",
        help=(
            "Model ID (provider-specific). Falls back to providers.<provider>.model "
            "in autoresearch/config.yaml when omitted. "
            "OpenRouter accepts any slug from openrouter.ai/models."
        ),
    ),
    max_trials: int = typer.Option(16, help="Maximum number of trials"),
    study: str = typer.Option("", help="Study name (auto-generated if omitted)"),
    results_dir: str = typer.Option("results/autoencoder", help="Base results directory"),
) -> None:
    """Run an LLM-driven search over Hydra overrides for elemental experiments."""
    ar_cfg = _load_autoresearch_config()
    resolved_provider = provider or ar_cfg.get("provider", "")
    if not resolved_provider:
        raise typer.BadParameter(
            "No provider set; pass --provider or set 'provider' in autoresearch/config.yaml."
        )

    provider_defaults = ar_cfg.get("providers", {}).get(resolved_provider, {})
    if model:
        resolved_models = [model]
    elif "models" in provider_defaults:
        resolved_models = provider_defaults["models"]
    elif "model" in provider_defaults:
        resolved_models = [provider_defaults["model"]]
    else:
        resolved_models = []

    if not resolved_models:
        raise typer.BadParameter(
            f"No model set for provider '{resolved_provider}'; pass --model or add "
            f"'providers.{resolved_provider}.models' to autoresearch/config.yaml."
        )

    client = LLMClient(provider=resolved_provider, models=resolved_models)
    logger.info("Provider: %s | Models: %s", resolved_provider, resolved_models)

    if not study:
        study = f"{space.stem}_{uuid.uuid4().hex[:6]}"

    run_dir = _RUNS_DIR / study
    run_dir.mkdir(parents=True, exist_ok=True)
    trials_log = run_dir / "trials.jsonl"

    debug_log = _setup_logging(run_dir)
    print(f"Study: {study} → {run_dir}")
    print(f"Debug log: {debug_log}")
    print(f"Trials:    {trials_log}")
    logger.info("Study: %s → %s", study, run_dir)

    space_cfg = _load_yaml(space)
    fixed_overrides: list[str] = space_cfg.get("fixed_overrides", [])
    if not any("trial_summary" in o for o in fixed_overrides):
        fixed_overrides.append(_TRIAL_SUMMARY_OVERRIDE)

    history: list[dict] = []
    if trials_log.exists():
        for line in trials_log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                history.append(json.loads(line))
        logger.info("Loaded %d existing trials from %s", len(history), trials_log)

    experiment_yaml, model_source, model_source_path, defaults = _resolve_experiment_context(
        config, fixed_overrides
    )
    logger.info(
        "Resolved experiment defaults: batch_size=%d, gradient_accumulation_steps=%d "
        "(model source: %s)",
        defaults["batch_size"],
        defaults["gradient_accumulation_steps"],
        model_source_path,
    )

    trial_idx = len(history)
    max_parse_retries = 3
    while trial_idx < max_trials:
        user_msg = _build_user_message(
            space_cfg,
            history,
            trial_idx,
            max_trials,
            experiment_yaml,
            model_source,
            model_source_path,
        )

        # Re-prompt on malformed JSON / empty overrides without burning the
        # trial slot. A bounded retry budget prevents a stuck model from
        # spinning forever.
        parsed: dict | None = None
        agent_overrides: list[str] | str = []
        reasoning = ""
        model_suggestions: list[str] = []
        for parse_attempt in range(max_parse_retries):
            raw_text = client.complete(_SYSTEM_PROMPT, user_msg)
            logger.info("Model response (%d chars):\n%s", len(raw_text), raw_text[:2000])

            parsed = _extract_json_object(raw_text)
            if parsed is None:
                logger.warning(
                    "Model returned no parseable JSON (attempt %d/%d). "
                    "Raw response (truncated):\n%s",
                    parse_attempt + 1,
                    max_parse_retries,
                    raw_text[:500],
                )
                continue

            reasoning = parsed.get("reasoning", "")
            agent_overrides = parsed.get("overrides", [])
            model_suggestions = parsed.get("model_suggestions", [])
            if agent_overrides == "<STOP>":
                break
            if isinstance(agent_overrides, list) and len(agent_overrides) > 0:
                break
            logger.warning(
                "Overrides missing or empty (attempt %d/%d). Got: %r",
                parse_attempt + 1,
                max_parse_retries,
                agent_overrides,
            )
        else:
            logger.error(
                "Giving up on trial %d after %d failed model responses; advancing.",
                trial_idx + 1,
                max_parse_retries,
            )
            print(
                f"Trial {trial_idx + 1}/{max_trials} — skipped "
                f"(model returned malformed JSON {max_parse_retries}x)"
            )
            trial_idx += 1
            continue

        if agent_overrides == "<STOP>":
            logger.info("Model signalled STOP. Reasoning: %s", reasoning)
            break

        if not isinstance(agent_overrides, list) or len(agent_overrides) == 0:
            # Should be unreachable given the retry loop above, but guard anyway.
            logger.error("Unexpected overrides payload after retries: %r", agent_overrides)
            trial_idx += 1
            continue

        logger.info("Overrides: %s", agent_overrides)
        logger.info("Reasoning: %s", reasoning)
        if model_suggestions:
            logger.info("Model suggestions: %s", model_suggestions)

        summary = _run_trial(config, fixed_overrides, agent_overrides, results_dir, defaults)
        summary["trial_idx"] = trial_idx
        summary["agent_overrides"] = agent_overrides
        summary["agent_reasoning"] = reasoning
        summary["agent_model_suggestions"] = model_suggestions
        summary["llm_provider"] = resolved_provider
        summary["llm_model"] = client.last_used_model

        history.append(summary)
        with open(trials_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")

        metric_val = _nested_get(summary, space_cfg["metric"])
        metric_str = f"{metric_val:.4g}" if isinstance(metric_val, (int, float)) else "n/a"
        logger.info(
            "Trial %d/%d — status=%s, %s=%s",
            trial_idx + 1,
            max_trials,
            summary.get("status"),
            space_cfg["metric"],
            metric_str,
        )
        print(
            f"Trial {trial_idx + 1}/{max_trials} — status={summary.get('status')}  "
            f"{space_cfg['metric']}={metric_str}  overrides={agent_overrides}"
        )
        trial_idx += 1

    _print_ranking(history, space_cfg)
    print(f"Done. Results: {trials_log}")
    logger.info("Done. Results: %s", trials_log)


if __name__ == "__main__":
    app()
