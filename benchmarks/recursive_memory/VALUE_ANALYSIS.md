# recursive_memory pretrain — value-for-money analysis

Cost/throughput comparison for the `recursive_memory` pretrain arm
(`configs/experiment/recursive_memory/pretrain_benchmark.yaml`, model `RecursiveMemoryGPT`,
186M params) across two 8-GPU nodes.

- **Raw records:** [`8xA100/benchmark.json`](8xA100/benchmark.json),
  [`8xH100/benchmark.json`](8xH100/benchmark.json) (W&B run summaries).
- **Analyzer:** `src/tools/benchmark_value.py`
- **Target token budget:** 3,932,160,000 tokens — the full `pretrain.yaml` horizon
  (`total_steps` 7500 × `tokens_per_optstep_global` 524,288 ≈ 3.93B, ~21×/param, Chinchilla-optimal).

Reproduce (per price point):

```bash
uv run python src/tools/benchmark_value.py \
  --records "benchmarks/recursive_memory/8xA100/benchmark.json" \
  --price 3 --target_tokens 3932160000
```

## Measured throughput (as benchmarked, bf16, seq_len 2048, global batch 524,288 tok/optstep)

| Node    | tok/s (aggregate) | tok/s/GPU | MFU  | median step | peak mem |
|---------|------------------:|----------:|-----:|------------:|---------:|
| 8×A100  |            64,216 |     8,027 | 2.9% |    8,167 ms |  37.4 GB |
| 8×H100  |            99,344 |    12,418 | 1.4% |    5,283 ms |  37.5 GB |

The H100 node is **~1.55× faster** in wall-clock (99,344 / 64,216). MFU is low on both and
notably lower on H100 (1.4% vs 2.9%) — the PEER per-token expert gather is memory-bandwidth /
gather bound, so the H100's much larger peak FLOPs (989 vs 312 TF/s) go underutilized. This is
the crux of the value story: you pay for H100 FLOPs you can't feed.

## Cost to train the full 3.93B-token pretrain run

Ranked cheapest total cost first. Wall-clock is fixed per node (~17.0 h A100, ~11.0 h H100);
only price moves cost.

| Rank | Node    | $/hr | wall (h) | cost-to-train | $/1M tokens |
|-----:|---------|-----:|---------:|--------------:|------------:|
| 1    | 8×A100  |    3 |    17.01 |    **$51.03** |     $0.0130 |
| 2    | 8×A100  |    4 |    17.01 |        $68.04 |     $0.0173 |
| 3    | 8×A100  |    7 |    17.01 |       $119.06 |     $0.0303 |
| 4    | 8×A100  |   10 |    17.01 |       $170.09 |     $0.0433 |
| 5    | 8×H100  |   16 |    10.99 |       $175.92 |     $0.0447 |
| 6    | 8×H100  |   22 |    10.99 |       $241.88 |     $0.0615 |

## Takeaways

- **Cheapest overall: 8×A100 at $3/hr → ~$51 to train the whole run** (17.0 h wall-clock).
  At every listed A100 price the A100 node is cheaper per token than either H100 price.
- **Fastest: 8×H100 → 11.0 h** (~6 h sooner), but the cheapest way to get that speed
  ($16/hr → $176) still costs more than the *most expensive* A100 option ($10/hr → $170).
- **Price breakeven (H100 vs A100).** Because H100 finishes in 0.646× the A100 wall-clock,
  H100 wins on cost only once the A100 spot price climbs past `price_H100 × 0.646`:
  - vs H100 @ $16/hr → A100 is the better value below **$10.34/hr** (so A100 @ $10 ≈ tie).
  - vs H100 @ $22/hr → A100 is the better value below **$14.21/hr**.
  Within the given price grid, A100 always wins on cost; H100 always wins on time.

## Recommendation

To run the full `configs/experiment/recursive_memory/pretrain.yaml` as cheaply as possible:
**use the 8×A100 node at the $3/hr interruptible price — ~$51.03 for the full 3.93B-token run
(~17 h).** Pay up for H100 only when the ~6-hour-earlier finish is worth the premium, or if
A100 spot pricing spikes above ~$10/hr (vs H100 @ $16) — above that crossover the H100 is both
faster *and* cheaper.

---

## Update — PEER memory revision + batch-16 (measured on 8×A100)

After a change to the PEER memory, two fresh 8×A100 benchmarks were run. The revision **cut peak
memory sharply** — 37.4 GB → **27.8 GB** at the original `batch_size: 8` — which freed enough room
to actually run **`batch_size: 16 / gradient_accumulation_steps: 2`** (same 0.524M-token global
batch). Both fit; batch 16 peaks at 51.9 GB allocated (70.2 GB reserved). `gradient_checkpointing`
stays **on** (turning it off is not an option for now), so Lever 2 below is deferred.

Raw records: [`8xA100/peer_updated_bs8_ga4/benchmark.json`](8xA100/peer_updated_bs8_ga4/benchmark.json),
[`8xA100/peer_updated_bs16_ga2/benchmark.json`](8xA100/peer_updated_bs16_ga2/benchmark.json).

### Throughput & memory (8×A100, bf16, global batch 0.524M tokens)

| Config (PEER rev)         | peak mem (alloc / reserved) | tok/s (agg) | MFU  | median step | wall (h) |
|---------------------------|----------------------------:|------------:|-----:|------------:|---------:|
| bs 8 / ga 4               |          27.8 / 35.9 GB     |      65,127 | 2.9% |    8,049 ms |    16.77 |
| **bs 16 / ga 2**          |          51.9 / 70.2 GB     |  **68,588** | 3.1% |    7,643 ms |**15.93** |
| _(old PEER, bs 8 / ga 4)_ |          37.4 / 38.0 GB      |      64,216 | 2.9% |    8,167 ms |    17.01 |

Batch 16 is **+5.3%** throughput over the revised bs-8 run and **+6.8%** over the original — the
gather-bound ceiling still caps the gain, but it's a free win now that the memory fits.

### Cost to train the full 3.93B-token run (8×A100, batch 16 / ga 2 — the new cheapest)

| $/hr | wall (h) | cost-to-train | $/1M tokens | vs. original bs-8 |
|-----:|---------:|--------------:|------------:|------------------:|
|    3 |    15.93 |    **$47.78** |     $0.0121 |  −$3.25 (−6.4%)   |
|    4 |    15.93 |        $63.70 |     $0.0162 |  −$4.34 (−6.4%)   |
|    7 |    15.93 |       $111.48 |     $0.0283 |  −$7.58 (−6.4%)   |
|   10 |    15.93 |       $159.25 |     $0.0405 | −$10.84 (−6.4%)   |

For reference, the revised **bs 8 / ga 2** run costs $50.31 / $67.09 / $117.40 / $167.71 at the same
four prices — i.e. batch 16 saves another ~5% on top.

**Updated recommendation:** run the full pretrain on **8×A100 at $3/hr with `batch_size: 16`,
`gradient_accumulation_steps: 2` → ~$47.78 for the full 3.93B-token run (~15.9 h).** This is the
cheapest configuration measured, ~6% under the original bs-8 baseline, and it's a confirmed fit
(not an extrapolation). The H100 cost-crossover math from the original analysis is unchanged.

---

## Additional notes — can we go faster / cheaper?

> Status after the PEER revision above: **Lever 1 (batch 16) is now confirmed and adopted.**
> **Lever 2 (disable checkpointing) is off the table for now.** Lever 3 (the PEER gather) remains
> the real ceiling. The prose below is the original headroom analysis, kept for rationale.

Both benchmarks peaked at only **~37.4 GB of the 80 GB** on each card (`benchmark/peak_mem_gb`),
i.e. **~42 GB (~53%) idle headroom** on both the A100 and H100 node. That headroom is real money
left on the table. Two independent levers use it, and they compete for the same memory — pick one
and **re-run `pretrain_benchmark` to confirm the fit before committing to the full run** (the
numbers below are extrapolated from a single 37.4 GB data point, so treat them as starting points,
not guarantees).

### Lever 1 — increase `batch_size` (keep the global batch fixed)

Global batch = `batch_size × gradient_accumulation_steps × 8 GPUs = 0.524M tokens` is a *training*
hyperparameter and should stay fixed (it's matched across the nanochat / memory_bank / recursive
arms). So raise `batch_size` and drop `gradient_accumulation_steps` by the same factor — same
optimizer math, fewer but larger micro-batches (less launch/gather overhead per optimizer step).
`batch_size` must divide the per-GPU product of 32, so the clean step up is:

```
batch_size: 16
gradient_accumulation_steps: 2   # 16 × 2 × 8 = 0.524M tokens, unchanged
```

Activation memory scales ~linearly with `batch_size`. With ~3 GB of fixed state (params + grads +
Muon/AdamW moments) the ~34 GB of activations at batch 8 imply **~71 GB projected at batch 16** —
fits 80 GB but tight, so benchmark it. If it OOMs, this is the aggressive edge; there's no clean
intermediate that keeps the global batch fixed (24 doesn't divide 32), so fall back to Lever 2.
Expected upside is **modest** — this workload is gather-bound, not matmul-bound (see below), so
bigger micro-batches mostly trim overhead rather than lift utilization.

### Lever 2 — turn off `gradient_checkpointing` (deferred — not an option for now)

The model config flags this directly (`configs/model/recursive_memory_gpt_d12eq.yaml:31-36`):
`gradient_checkpointing: true` recomputes the PEER expert gather in backward — *"trades ~30%
wall-clock for the memory needed to fit a 24GB card."* On an 80 GB card that trade is unnecessary.
Setting `model.gradient_checkpointing=false` should recover **~30% throughput → ~23% lower cost**
(cost ×1/1.30). That dwarfs Lever 1, so try it first.

The catch: the saved gather tables `(B, T, peer_heads, peer_top_k, d)` across all 12
(`n_trunk_layers × num_loops`) PEER calls are exactly the dominant activation cost, so disabling
checkpointing adds *tens* of GB. It plausibly fits at the **current** `batch_size: 8` (consuming the
42 GB headroom) but **cannot be combined with the batch-16 bump** — you get one or the other.
Recommended order: (a) benchmark `gradient_checkpointing=false` at batch 8; if it fits, take the
~30%; (b) only if it still OOMs, keep checkpointing and use Lever 1. Applied to the winner, ~23%
off brings the A100 @ $3/hr run from ~$51 down to **~$39**.

### Lever 3 — the real ceiling is the PEER gather (MFU is 1.4–2.9%)

MFU is strikingly low and *drops* on the faster card (2.9% A100 → 1.4% H100). The recursive trunk
is bottlenecked on the per-token expert gather (`peer_heads × peer_top_k = 8 × 16 = 128` active
single-neuron experts/token), which is memory-bandwidth/gather bound, not FLOP bound — so the
H100's 3.2× peak FLOPs buy only ~1.55× real speedup, and that's why A100 wins on cost. Genuinely
lifting utilization means attacking the gather itself: lowering `peer_top_k` or `peer_heads`, or a
fused/grouped gather kernel. These change model capacity, so they're **research knobs, not free
throughput** — flag for the modeling side, not a config toggle for this run.

### Already covered / non-levers

- **`torch.compile`** is already on (`/torch_compile` in the pretrain defaults) — no action.
- **More GPUs / nodes** cut wall-clock but not cost (scaling is ~linear, price scales with it) —
  won't make the run cheaper, only faster.
- **FP8 on H100** won't help a gather/bandwidth-bound workload enough to overturn the A100 cost
  advantage; not worth the numerical risk here.

**Bottom line (updated):** `batch_size: 16 / gradient_accumulation_steps: 2` is now measured and
adopted (**A100 @ $3/hr: ~$51 → ~$47.78**, see the Update section). `gradient_checkpointing=false`
is deferred (not viable for now); if it later becomes an option it's the bigger lever (~23% off),
but the two compete for the same memory — you'd trade the batch-16 bump for it, so re-benchmark
before switching. Beyond these, don't expect large speedups without changing the PEER gather itself.
