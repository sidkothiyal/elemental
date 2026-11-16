# Benchmark Configs

Composable bundle for measuring GPU training throughput, so you can pick the best
value-for-money hardware before committing a long pretraining run. Runs the **real**
training loop for a short warmup + measurement window (via `BenchmarkCallback`) and writes
`results/<mode>/<experiment>/benchmark.json` (plus a `benchmark/*` block in
`wandb.run.summary` when W&B is on). No training-code changes — it's a pure observer.

## Files

| File | Description |
|------|-------------|
| `default.yaml` | Adds `BenchmarkCallback`; sets a short `max_steps`; disables checkpoints, remote backup, generation postprocessors, and validation to isolate pure-training throughput. |

## Usage

`default.yaml` is a **bundle, not an entry point**. Hydra can't select a root group with
`+group=` from a nested experiment config, so compose it via a thin wrapper experiment
(precedent: `experiment/vla/vla_rl_eval_sim.yaml`). Wrappers already exist for the three
LM pretrain arms:

```
configs/experiment/{nanochat,memory_bank,recursive_memory}/pretrain_benchmark.yaml
```

Each is just:

```yaml
# @package _global_
defaults:
  - /experiment/<arm>/pretrain
  - /benchmark: default
  - _self_
```

Run one and then rank candidate GPUs by estimated cost-to-train:

```bash
uv run python src/trainer.py experiment/recursive_memory/pretrain_benchmark
uv run python src/tools/benchmark_value.py \
  --records "results/**/benchmark.json" --price rtx6000=9.25 a100=9.25 h100=18.5
```

## Key knobs

- `callbacks.benchmark.warmup_steps` / `measure_steps` — window sizing (default 20 / 50).
- `max_steps` — **must be `>= warmup_steps + measure_steps + 1`** (the +1 lets the closing
  optimizer step be observed). Kept in sync in `default.yaml` as `71`.
- `peak_flops_per_sec` — optional MFU denominator override; otherwise a device-name lookup
  in `BenchmarkCallback` supplies it (H100 / A100 / RTX PRO 6000). MFU is diagnostic only —
  the cost ranking needs just tokens/sec + price.

## Tips

- Set `skip_nonfinite_steps=false` for a clean fail-fast benchmark.
- A large `p90_step_ms` / `median_step_ms` gap suggests a mid-window recompile — raise
  `warmup_steps`.
- Benchmark on the **actual N-GPU node** you'll rent. The analyzer's `--num_gpus`
  projection assumes *linear* scaling, which overstates PCIe-only cards with no
  NVLink/InfiniBand (e.g. RTX PRO 6000 Blackwell).
