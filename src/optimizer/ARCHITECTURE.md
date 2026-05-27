# src/optimizer — Architecture

## Muon (muon.py)

Step formula on each 2-D matrix param `p` with gradient `g`:

1. `buf ← momentum * buf + g`           (float32 momentum buffer)
2. `update ← buf + momentum * g` (Nesterov) or `buf` (plain)
3. `update ← zeropower_via_newtonschulz5(update, steps=ns_steps)`
4. `update ← update * sqrt(max(1, fan_out / fan_in))`
5. `p ← p * (1 - lr * weight_decay) - lr * update`

Newton-Schulz polynomial coefficients `(a, b, c) = (3.4445, -4.7750, 2.0315)`
are tuned for fast convergence on typical neural-net gradient spectra. The
iteration runs in bfloat16 internally for speed; the result is cast back to
the parameter dtype. Tall matrices are transposed before iteration so the
inner products run on the smaller dim.

Single-rank only in this first version. Cross-rank gather (used in upstream's
distributed Muon for very large matrices) is a follow-up; for the nanochat d12
/ d20 / d26 tiers each parameter fits comfortably in a single GPU's working
set, so per-rank Muon already matches the speedrun behavior.

## MultiOptimizer (multi_optimizer.py)

Lifecycle:

1. `__init__` calls `super().__init__(placeholder_param, {})` to satisfy
   `Optimizer.__init__`'s "non-empty params" precondition. The placeholder is
   added to `self.param_groups` by the parent.
2. We set `self._optimizers` AFTER `super().__init__` so our overridden
   `add_param_group` (which routes by `optimizer_name`) doesn't intercept the
   parent's placeholder insertion.
3. We then overwrite `self.param_groups` with a flat list of references to
   the sub-optimizers' groups, discarding the placeholder.

After init, `self.param_groups[i] is sub_optimizer.param_groups[j]` for some
`(j, sub_optimizer)` pair, so `group["lr"] = ...` mutations from a scheduler
propagate to the right place.

`state_dict` / `load_state_dict` round-trip works by storing each
sub-optimizer's `state_dict` under its name. After load we re-aggregate
`param_groups` so any updated lr / weight-decay values flow into the new
shape.

## partition_params (param_groups.py)

Predicates are tried in insertion order; first match wins. If a requires-grad
parameter doesn't match any predicate, we raise — never silently drop. This
catches typos in custom partition specs (e.g. forgetting a catch-all).

`is_matrix_block_param`'s exclusion list contains the substrings used by
nanochat-style and GPT-style models for the embedding / output-projection
parameters: `embedding`, `embed`, `lm_head`, `wte`, `wpe`. A custom model with
a different naming convention either renames its params or supplies a custom
predicate via Hydra `_target_:`.

## WarmupCosineLR (schedulers.py)

The lambda factor at step `s` (where `total_steps = T`, `warmup_steps = W`,
`min_lr_ratio = m`):

- `s < W`:        `factor = (s + 1) / max(1, W)`         (linear ramp; +1 so step 0 is non-zero)
- `W ≤ s ≤ T`:    `factor = m + (1 - m) * 0.5 * (1 + cos(pi * (s - W) / (T - W)))`
- `s > T`:        `factor = m`                            (floor)

`LambdaLR.state_dict()` saves `last_epoch`, so resuming from a checkpoint
continues the schedule exactly.
