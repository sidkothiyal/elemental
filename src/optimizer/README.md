# src/optimizer — Optimizers, Schedulers, and Param Partitioning

Custom optimizer machinery used by the nanochat pipeline (and reusable
elsewhere). The trainer's standard path through
[helpers.py](../training/helpers.py) recognizes a new `optimizer_groups:` key
and routes through these modules.

## Files

### `muon.py` — Muon

Faithful single-rank port of Karpathy's Muon: SGD-with-momentum followed by a
five-step Newton-Schulz orthogonalization on the (Nesterov-lookahead)
momentum buffer, plus an aspect-ratio scale and decoupled weight decay.

```python
Muon(params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0)
```

Operates only on 2-D matrix parameters — passing a 1-D tensor raises at
`step()`. Pair with AdamW (for embeddings, norm scales, scalars) via
[`MultiOptimizer`](#multi_optimizerpy--multioptimizer).

`zeropower_via_newtonschulz5(g, steps=5)` is exposed as a pure function for
unit tests; it runs in bfloat16 internally and casts back.

### `multi_optimizer.py` — MultiOptimizer

Composes an ordered dict of `torch.optim.Optimizer` instances behind a single
facade that PyTorch's LR schedulers (`isinstance(_, Optimizer)` check) accept.

```python
MultiOptimizer({"muon": Muon(...), "adamw": AdamW(...)}, default_group="adamw")
```

`param_groups` is a flat list of references to the sub-optimizers' groups, so
schedulers that mutate `group["lr"]` propagate to the right sub-optimizer.
`add_param_group(...)` accepts an `optimizer_name` field to route to a specific
sub-optimizer; otherwise routes to `default_group`.

### `param_groups.py` — Predicate-based partitioning

`partition_params(model, predicates)` walks `model.named_parameters()` and
assigns each requires-grad param to the first matching predicate. Unmatched
params raise — silent drops would waste GPU memory.

`nanochat_default()` returns the standard recipe:

| Group | Predicate |
|---|---|
| `muon` | 2-D weights NOT named `*embedding*`, `*embed*`, `*lm_head*`, `*wte*`, `*wpe*` |
| `adamw` | catch-all (norm scales, embeddings, lm_head, scalars) |

Model-specific variants extend the Muon exclusion list with 2-D tables that must
stay off Newton-Schulz:
- `recursive_memory_default()` — also excludes the PEER bank / sub-keys / loop-emb.
- `hierarchical_default()` — also excludes Perceiver compressor latents and any
  positional-embedding / gate / null-context tables (used by `HierarchicalLM` and
  `MultiScaleLM`).

### `schedulers.py` — WarmupCosineLR

Linear warmup → cosine decay → flat `min_lr_ratio` floor. Built on `LambdaLR`
so the standard `state_dict`/`load_state_dict` works. Compatible with
`MultiOptimizer`.

```python
WarmupCosineLR(optimizer, warmup_steps=1000, total_steps=100000, min_lr_ratio=0.1)
```

## Configs

| File | Purpose |
|---|---|
| [`muon_adamw.yaml`](../../configs/optimizer/muon_adamw.yaml) | Muon + AdamW pair (nanochat default) |
| [`muon_adamw_peer.yaml`](../../configs/optimizer/muon_adamw_peer.yaml) | Muon + AdamW with `recursive_memory_default` partition |
| [`muon_adamw_hierarchical.yaml`](../../configs/optimizer/muon_adamw_hierarchical.yaml) | Muon + AdamW with `hierarchical_default` partition |
| [`nanochat_warmup_cosine.yaml`](../../configs/scheduler/nanochat_warmup_cosine.yaml) | warmup-cosine scheduler with `_target_:` |

The trainer's optimizer builder branches on `config.optimizer_groups`:
- present → return a `MultiOptimizer`
- absent → fall back to the legacy `getattr(torch.optim, ...)` path

The scheduler builder accepts either a string (legacy: looks up
`torch.optim.lr_scheduler.<name>`) or a dict with `_target_:` (Hydra-instantiate
with `optimizer=` injected).
