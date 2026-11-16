# src/loss — Loss Functions

Composable loss modules for all training tasks.

## Files

### `loss_container.py` — LossContainer + LossComponent

The primary way to configure losses via Hydra. `LossContainer` orchestrates multiple named `LossComponent` instances and aggregates their outputs.

**When to use**: For all standard training. Configure in `configs/loss/`.

```python
# Equivalent to configs/loss/mse.yaml
container = LossContainer(losses={
    "mse": LossComponent(loss_fn=nn.MSELoss(), pred_key="pred", target_key="target"),
    "ssim": LossComponent(loss_fn=SSIMLoss(), weight=0.3),
})
total, components = container(data)
```

`context_mapping` allows forwarding extra keys (e.g. mask) from the data container to the loss function as kwargs.

### `ssim.py` — SSIM Loss

Structural similarity loss using `pytorch_msssim`.

```python
SSIMLoss(multi_scale=True)  # default: MS-SSIM
SSIMLoss(multi_scale=False) # single-scale SSIM
```

Returns `1 - ssim_value` so it can be minimized like other losses.

### `vgg_loss.py` — VGG Perceptual Loss

Perceptual loss using frozen VGG16/VGG19 feature layers.

```python
VGGLoss(model="vgg16", layers=[3, 8, 15], reduction="mean")
```

Valid layer indices: 3 (relu1_2), 8 (relu2_2), 15 (relu3_3), 22 (relu4_3). Multi-layer losses are spatially upscaled to match the lowest layer before averaging.

### `gradient_loss.py` — Gradient Loss

L1 loss on spatial gradients (horizontal + vertical). Encourages sharp edges.

```python
GradientLoss()
```

### `masked_loss.py` — Masked Loss

Applies any wrapped loss with spatial masking (e.g., for masked autoencoder reconstruction).

```python
MaskedLoss(
    loss_function=nn.MSELoss(reduction="none"),
    mask_weight=1.0,
    neg_mask_weight=0.0,
    auto_scale_loss=False,  # True to auto-interpolate if loss shape differs from mask
)
```

### `phased_loss.py` — Phased Loss

Switches between different loss functions based on percentage of training elapsed.

```python
PhasedLoss(total_epochs=200, phases=[
    {"run_until": 0.25, "loss": nn.L1Loss()},     # epochs 0-50
    {"run_until": 0.75, "loss": nn.MSELoss()},    # epochs 50-150
    {"run_until": 1.0,  "loss": SSIMLoss()},      # epochs 150-200
])
```

Requires `epoch` argument in `forward()`.

### `weighted_loss.py` — Weighted Loss

Fixed-weight combination of multiple loss functions.

```python
WeightedLoss(
    loss_fns=[nn.MSELoss(), SSIMLoss()],
    weights=[0.7, 0.3],
    normalize_weights=True,
    scale_losses=False,  # True to normalize each loss by running max
)
```

### `moving_weighted_loss.py` — Moving Weighted Loss

Like `WeightedLoss` but weights interpolate over training. Useful for curriculum learning.

```python
MovingWeightedLoss(
    loss_fns=[nn.L1Loss(), SSIMLoss()],
    start_weights=[0.9, 0.1],
    end_weights=[0.1, 0.9],
    total_epochs=200,
    start_percentage=0.1,   # start transition at 10% training
    end_percentage=0.9,     # finish at 90% training
    easing="cosine_in",     # easing curve: linear/cosine/cosine_in/sigmoid/quadratic
    scale_losses=True,      # EMA-normalize each loss magnitude
)
```

### `shifted_teacher_forcing.py` — Shifted Teacher Forcing

Cross-entropy loss for autoregressive models with teacher forcing. Shifts predictions and targets by `shift` positions.

```python
ShiftedTeacherForcing(shift=1, ignore_index=-100)
```

Use `shift=0` when the dataset already returns shifted `(input, target)` pairs
(see `FineWebEduDataset`, `SmolTalkSFTDataset`).

### `masked_cross_entropy.py` — Masked Cross-Entropy

Cross-entropy averaged over positions selected by a boolean mask. Used for
masked-language-modelling and LLaDA-style masked-discrete-diffusion training
where loss should fire only at corrupted positions. Pulls `mask` from the
data container via `LossComponent.context_mapping`.

```python
MaskedCrossEntropy(ignore_index=-100)
# forward(pred: (B, T, V), target: (B, T), mask: (B, T)) -> scalar
```

Internally rewrites unmasked target positions to `ignore_index` and calls
`F.cross_entropy(pred.transpose(1, 2), ...)`. Returns `0` (with gradient
zero) when no positions are masked. The hierarchical-thinking compressor MAE and joint
masked-diffusion configs both use this — see
`configs/loss/hierarchical_thinking_compressor_mae.yaml` and
`configs/loss/hierarchical_thinking_masked_diffusion.yaml`.

### `deep_supervision_ce.py` — Deep-Supervision Cross-Entropy

Next-token cross-entropy averaged over per-loop logits for the recursive LM
(`RecursiveMemoryGPT` with `deep_supervision=True`), which emits logits stacked
as `(K, B, T, V)`. Applies CE at every loop step against the same targets (the
TRM/HRM deep-supervision recipe). Only used in the Phase-4 recursion ablations
— see `configs/loss/lm_pretrain_deep_supervision.yaml`; the lean V0 run returns
a plain tensor and uses `ShiftedTeacherForcing`.

```python
DeepSupervisionCE(ignore_index=-100)
# forward(pred: (K, B, T, V), target: (B, T)) -> scalar
```

### `grpo_loss.py` — GRPO clipped surrogate (per-token, masked)

Per-token clipped surrogate for GRPO with optional KL-to-reference. Pulls
`log_prob_new`, `log_prob_old`, `advantage`, `completion_mask`, and (optional)
`log_prob_ref` from the trainer's data container via `LossComponent`'s
`context_mapping`. The KL term uses Schulman's k3 estimator
`exp(log_prob_ref - log_prob_new) - (log_prob_ref - log_prob_new) - 1`, which
is positive and low-variance and matches upstream nanochat / DeepSeek-Math.

```python
GRPOLoss(kl_coef=0.0)
```

The PG term is masked over completion tokens and divided by the completion
token count, so longer completions don't dominate the gradient.

### `policy_losses.py` — PPO-style policy losses

`PPOClipLoss`, `ValueLoss`, `EntropyBonus` — standard PPO components used by
the SIMPLER-VLA RL flow. GRPO doesn't use these (see `grpo_loss.py`) but they
share the same `LossContainer` plumbing.

### `utils.py` — Loss Utilities

`call_loss_with_epoch(loss_fn, pred, target, epoch)` — calls a loss with `epoch` kwarg if the loss function's signature accepts it. Used internally by phased/weighted losses.
