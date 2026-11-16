#!/usr/bin/env python3
"""Value-for-money analyzer for GPU training benchmarks.

Reads one or more ``benchmark.json`` files produced by ``BenchmarkCallback`` (and/or
W&B run summaries), plugs in a per-node-hour price, and ranks candidate GPUs by the
estimated **total cost to train** a target token budget — cheapest first, with the
estimated wall-clock hours alongside.

Usage:
    # Local benchmark.json files, one price per GPU family:
    uv run python src/tools/benchmark_value.py \
        --records "results/**/benchmark.json" \
        --price "rtx6000=9.25,a100=9.25,h100=18.5"

    # From W&B run summaries (benchmark/* keys):
    uv run python src/tools/benchmark_value.py \
        --wandb_runs "elemental-ai/elemental/<id1>,elemental-ai/elemental/<id2>" \
        --price h100=18.5

    # Project a single-GPU measurement to an 8-GPU node (assumes linear scaling):
    uv run python src/tools/benchmark_value.py --records "results/**/benchmark.json" \
        --price a100=9.25 --num_gpus 8

Price keys match a GPU loosely (case-insensitive, punctuation-ignored substring against
the device name / short tag), so ``h100``, ``a100``, ``6000``/``rtx_pro_6000`` all work.
A bare number (``--price 12.5``) applies to every record.
"""

import glob
import json
from pathlib import Path
from typing import Any

import fire

# Compute-optimal (~Chinchilla 20×) horizon for the ~186M elemental LM arms:
# total_steps 7500 × 0.524M tokens/optstep ≈ 3.9B tokens. Override with --target_tokens.
DEFAULT_TARGET_TOKENS = 3.9e9


# ---------------------------------------------------------------------------
# Loading & normalisation
# ---------------------------------------------------------------------------


def _short_device(name: str | None) -> str:
    """Canonical short tag for a CUDA device name (used as a row label / price key)."""
    if not name:
        return "unknown"
    upper = name.upper()
    if "H100" in upper:
        return "h100"
    if "H200" in upper:
        return "h200"
    if "A100" in upper:
        return "a100"
    if "RTX PRO 6000" in upper or ("RTX" in upper and "6000" in upper):
        return "rtx_pro_6000"
    slug = "".join(c if c.isalnum() else "_" for c in name.lower())
    return slug.strip("_") or "unknown"


def _extract(record: dict[str, Any], source: str) -> dict[str, Any]:
    """Flatten a nested local record or a flat W&B summary into the fields we need."""
    if "throughput" in record:  # nested local benchmark.json
        thr, dev, mdl, batch = (
            record["throughput"],
            record["device"],
            record["model"],
            record.get("batching", {}),
        )
        view = {
            "status": record.get("status"),
            "tps_agg": thr.get("tokens_per_sec_aggregate"),
            "tps_pg": thr.get("tokens_per_sec_per_gpu"),
            "mfu": thr.get("mfu"),
            "num_gpus": dev.get("num_gpus") or 1,
            "device_name": dev.get("name"),
            "model_target": mdl.get("target"),
            "tokens_per_optstep_global": batch.get("tokens_per_optstep_global"),
        }
    else:  # flat wandb summary: benchmark/<name>
        view = {
            "status": record.get("benchmark/status"),
            "tps_agg": record.get("benchmark/tokens_per_sec_aggregate"),
            "tps_pg": record.get("benchmark/tokens_per_sec_per_gpu"),
            "mfu": record.get("benchmark/mfu"),
            "num_gpus": record.get("benchmark/num_gpus") or 1,
            "device_name": record.get("benchmark/name"),
            "model_target": record.get("benchmark/target"),
            "tokens_per_optstep_global": record.get("benchmark/tokens_per_optstep_global"),
        }
    view["source"] = source
    view["device_tag"] = _short_device(view["device_name"])
    view["label"] = f"{view['model_target'] or '?'} @ {view['device_tag']}"
    return view


def _load_local(records: str) -> list[dict[str, Any]]:
    views: list[dict[str, Any]] = []
    for pattern in str(records).split(","):
        pattern = pattern.strip()
        if not pattern:
            continue
        matches = sorted(glob.glob(pattern, recursive=True))
        if not matches:
            print(f"WARNING: no files matched {pattern!r}")
        for path in matches:
            try:
                data = json.loads(Path(path).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                print(f"WARNING: could not read {path}: {exc}")
                continue
            views.append(_extract(data, source=path))
    return views


def _load_wandb(wandb_runs: str) -> list[dict[str, Any]]:
    try:
        import wandb  # pylint: disable=import-outside-toplevel
    except ImportError:
        print("WARNING: wandb not installed — skipping --wandb_runs")
        return []
    api = wandb.Api()
    views: list[dict[str, Any]] = []
    for spec in str(wandb_runs).split(","):
        spec = spec.strip()
        if not spec:
            continue
        try:
            summary = dict(api.run(spec).summary)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"WARNING: could not fetch wandb run {spec}: {exc}")
            continue
        views.append(_extract(summary, source=spec))
    return views


# ---------------------------------------------------------------------------
# Pricing & value math
# ---------------------------------------------------------------------------


def _parse_price(price: Any) -> dict[str, float]:
    if price is None:
        return {}
    if isinstance(price, (int, float)):
        return {"*": float(price)}
    if isinstance(price, dict):
        return {str(k): float(v) for k, v in price.items()}
    text = str(price)
    if "=" not in text:
        return {"*": float(text)}
    out: dict[str, float] = {}
    for part in text.split(","):
        if not part.strip():
            continue
        key, val = part.split("=", 1)
        out[key.strip()] = float(val)
    return out


def _norm(text: str) -> str:
    return "".join(c for c in text.lower() if c.isalnum())


def _match_price(prices: dict[str, float], view: dict[str, Any]) -> float | None:
    if "*" in prices:
        return prices["*"]
    haystacks = [_norm(view["device_tag"]), _norm(view.get("device_name") or "")]
    for key, val in prices.items():
        nkey = _norm(key)
        if any(nkey and (nkey in hay or hay in nkey) for hay in haystacks if hay):
            return val
    return None


def _compute(
    view: dict[str, Any], price: float | None, target_tokens: float, num_gpus: int | None
) -> dict[str, Any]:
    tps_agg = view["tps_agg"]
    measured_gpus = int(view["num_gpus"] or 1)
    projected = False
    eff_price = price

    if tps_agg and num_gpus and num_gpus != measured_gpus and view.get("tps_pg"):
        tps_agg = view["tps_pg"] * num_gpus  # linear-scaling assumption
        projected = True
        if price is not None:
            eff_price = (price / measured_gpus) * num_gpus

    row = dict(view)
    row.update(
        {
            "tps_used": tps_agg,
            "num_gpus_used": num_gpus if projected else measured_gpus,
            "projected": projected,
            "price": eff_price,
            "est_wall_hours": None,
            "cost_to_train": None,
            "tokens_per_dollar": None,
            "cost_per_1m_tokens": None,
        }
    )
    if tps_agg and tps_agg > 0:
        row["est_wall_hours"] = target_tokens / (tps_agg * 3600.0)
        if eff_price is not None and eff_price > 0:
            row["cost_to_train"] = row["est_wall_hours"] * eff_price
            row["tokens_per_dollar"] = tps_agg * 3600.0 / eff_price
            row["cost_per_1m_tokens"] = eff_price * 1e6 / (tps_agg * 3600.0)
    return row


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt(value: Any, spec: str) -> str:
    if value is None:
        return "-"
    return format(value, spec)


def _money(value: Any, spec: str) -> str:
    return "-" if value is None else "$" + format(value, spec)


def _print_table(rows: list[dict[str, Any]], target_tokens: float) -> None:
    header = (
        f"{'label':<34} {'GPUs':>4} {'tok/s(agg)':>12} {'MFU':>6} "
        f"{'$/hr':>8} {'wall(h)':>9} {'cost-to-train':>14} {'$/1M-tok':>10}"
    )
    print(f"\nTarget budget: {target_tokens:.3g} tokens\n")
    print(header)
    print("-" * len(header))
    for r in rows:
        cost = _money(r["cost_to_train"], ",.2f")
        per_1m = _money(r["cost_per_1m_tokens"], ".4f")
        print(
            f"{r['label'][:34]:<34} {r['num_gpus_used']:>4} "
            f"{_fmt(r['tps_used'], ',.0f'):>12} {_fmt(r['mfu'], '.1%'):>6} "
            f"{_fmt(r['price'], ',.2f'):>8} {_fmt(r['est_wall_hours'], ',.2f'):>9} "
            f"{cost:>14} {per_1m:>10}"
        )


def analyze(
    records: str | None = None,
    wandb_runs: str | None = None,
    price: Any = None,
    target_tokens: float = DEFAULT_TARGET_TOKENS,
    num_gpus: int | None = None,
) -> None:
    """Rank benchmarked GPUs by estimated cost to train a target token budget.

    Args:
        records: Comma-separated glob(s) of local ``benchmark.json`` files (supports ``**``).
        wandb_runs: Comma-separated W&B run specs (``entity/project/run_id``).
        price: ``name=USD_per_node_hour`` pairs (comma-separated) or a bare number for all.
        target_tokens: Token budget to cost out (default ≈3.9B, the compute-optimal horizon).
        num_gpus: Project each per-GPU rate to this many GPUs (assumes LINEAR scaling).
    """
    views: list[dict[str, Any]] = []
    if records:
        views += _load_local(records)
    if wandb_runs:
        views += _load_wandb(wandb_runs)
    if not views:
        print("No benchmark records found. Pass --records and/or --wandb_runs.")
        return

    prices = _parse_price(price)
    target_tokens = float(target_tokens)
    rows = [_compute(v, _match_price(prices, v), target_tokens, num_gpus) for v in views]

    for r in rows:
        if r["status"] not in (None, "ok"):
            print(f"WARNING: {r['label']} has status={r['status']!r} ({r['source']})")
        if prices and r["price"] is None:
            print(f"WARNING: no --price matched {r['label']} ({r['device_name']}) — cost omitted")
    if num_gpus and any(r["projected"] for r in rows):
        print(
            f"WARNING: throughput projected to {num_gpus} GPUs assuming LINEAR scaling. "
            "Real scaling is sublinear on PCIe-only cards (e.g. RTX PRO 6000 Blackwell "
            "has no NVLink/InfiniBand) — benchmark on the real N-GPU node for accuracy."
        )

    # Headline ranking: cheapest total cost to train first (rows without a cost sink last).
    def _cost_key(r: dict[str, Any]) -> tuple[int, float]:
        c = r["cost_to_train"]
        return (1, 0.0) if c is None else (0, c)

    ranked = sorted(rows, key=_cost_key)
    _print_table(ranked, target_tokens)

    priced = [r for r in ranked if r["cost_to_train"] is not None]
    if priced:
        best = priced[0]
        print(
            f"\nBest value:  {best['label']}  "
            f"→ ${best['cost_to_train']:,.2f} to train ({best['est_wall_hours']:,.2f} h)"
        )
    timed = [r for r in rows if r["tps_used"]]
    if timed:
        fastest = max(timed, key=lambda r: r["tps_used"])
        print(
            f"Fastest:     {fastest['label']}  "
            f"→ {fastest['tps_used']:,.0f} tok/s aggregate "
            f"({(fastest['est_wall_hours'] or 0):,.2f} h)"
        )


def main() -> None:
    """Entry point for the benchmark value analyzer."""
    fire.Fire(analyze)


if __name__ == "__main__":
    main()
