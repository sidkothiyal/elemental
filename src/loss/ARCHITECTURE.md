# src/loss — Architecture

## LossContainer / LossComponent (loss_container.py)

### LossComponent

```python
class LossComponent(nn.Module):
    loss_fn: nn.Module
    pred_key: str          # data["pred_key"] → pred
    target_key: str        # data["target_key"] → target
    context_mapping: dict  # {data_path: kwarg_name} → extra kwargs for loss_fn
    weight: float          # scalar multiplier

    def forward(self, data: dict) -> torch.Tensor
```

`_resolve(path, data)` does simple dict lookup (supports nested paths via `data[path]` directly).

### LossContainer

```python
class LossContainer(nn.Module):
    losses: nn.ModuleDict   # name → LossComponent
    aggregate: Callable     # default: sum()

    def forward(self, data: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]
```

Returns `(total_loss, component_losses_dict)`. The trainer accumulates component losses separately for logging.

## SSIMLoss (ssim.py)

Thin wrapper around `pytorch_msssim.MS_SSIM` or `pytorch_msssim.SSIM`. Returns `1 - ssim_value`. Constructor kwargs forwarded to underlying module.

## VGGLoss (vgg_loss.py)

### Architecture

- Builds `nn.ModuleDict` of `vgg.features[:layer+1]` sub-networks, one per requested layer
- All extractors set to `eval()` with `requires_grad_(False)` at init
- Forward: concatenates input+target batches, splits features, computes per-layer MSE
- Multi-layer: upscales higher-layer losses to match lowest layer spatial resolution using bilinear interpolation

### Key methods

- `get_features(input)` — extract feature list at all requested layers
- `_scale_to_target_shape(tensor, target_shape)` — bilinear interpolation for feature alignment

## GradientLoss (gradient_loss.py)

Finite differences: `pred_dx = pred[:,:,:,1:] - pred[:,:,:,:-1]`, same for y. L1 loss between pred and target gradient components. No learnable parameters.

## MaskedLoss (masked_loss.py)

- Requires `reduction="none"` from wrapped `loss_function` to get per-element values
- When `mask` is `None`, derives from `neg_mask` and vice versa
- Loss averaging: `(loss * mask).sum() / (mask.sum() + 1e-8)` — numerically stable
- `auto_scale_loss=True` uses bilinear interpolation if loss and mask spatial dims differ (warns once)

## PhasedLoss (phased_loss.py)

- `phases` sorted by `end_epoch` at init; uses `_get_phase_for_epoch()` for O(n) lookup
- Loss functions stored in `nn.ModuleList` for proper parameter registration
- Epoch-to-phase mapping: first phase where `epoch < end_epoch`
- Delegates to `call_loss_with_epoch()` to optionally forward `epoch` kwarg

## WeightedLoss (weighted_loss.py)

- Weights stored as plain tensor (not `register_buffer`) — moved to device in `forward()`
- `scale_losses=True`: tracks max loss value per loss function (`loss_scales` buffer); divides each loss by its max before weighting (prevents one loss dominating)
- `_scales_initialized` bool buffer tracks first-seen status per loss

## MovingWeightedLoss (moving_weighted_loss.py)

### Weight Interpolation

```python
# In transition window [start_epoch, end_epoch):
linear_progress = (epoch - start_epoch) / (end_epoch - start_epoch)
progress = easing_fn(linear_progress)  # apply curve
weights = start_weights + (end_weights - start_weights) * progress
if normalize_weights:
    weights = weights / weights.sum()
```

Easing functions in `EASING_FUNCTIONS` dict:
- `"linear"`: `t`
- `"cosine"`: `(1 - cos(π·t)) / 2` (S-curve)
- `"cosine_in"`: `1 - cos(π/2·t)` (slow start)
- `"sigmoid"`: `1/(1+e^(-12*(t-0.5)))` (sharp S-curve)
- `"quadratic"`: `t²` (accelerating)

`scale_losses=True` uses EMA instead of max: `scale = decay * scale + (1-decay) * raw_loss`.

## ShiftedTeacherForcing (shifted_teacher_forcing.py)

Transposes predictions `(B, T, vocab) → (B, vocab, T)` for CrossEntropyLoss, then slices: `pred = pred[:,:,:-shift]`, `target = target[:,shift:]`.

## utils.py

```python
def call_loss_with_epoch(loss_fn, pred, target, epoch, **kwargs):
    sig = inspect.signature(loss_fn.forward)
    if "epoch" in sig.parameters:
        return loss_fn(pred, target, epoch=epoch, **kwargs)
    return loss_fn(pred, target, **kwargs)
```

Used by `WeightedLoss`, `MovingWeightedLoss`, `PhasedLoss` to pass `epoch` only when needed.
