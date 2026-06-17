# src/metrics — Metrics

TorchMetrics-based metric implementations for evaluation.

## Files

### `iou.py` — IoU (Intersection over Union)

Semantic segmentation metric. Computes per-class and mean IoU.

### `mse.py` — MSE (Mean Squared Error)

Reconstruction metric.

### `psnr.py` — PSNR (Peak Signal-to-Noise Ratio)

Image quality metric. Higher is better. Common threshold: >30 dB is good compression quality.

### `ssim.py` — SSIM (Structural Similarity Index)

Perceptual image quality metric. Range [0, 1]; higher is better.

### `next_token_accuracy.py` — NextTokenAccuracy (LM)

Top-1 next-token accuracy from `(B, T, vocab)` logits vs `(B, T)` target ids, masking
`ignore_index`. A `BaseMetric` (per-sample accuracy over valid positions). Wire via the
LM metrics callback on the `val` phase.

### `bits_per_byte.py` — BitsPerByte (LM)

Tokenizer-independent bits-per-byte = `(Σ cross_entropy_bits) / (Σ decoded UTF-8 bytes)`.
A sum-based `torchmetrics.Metric` (a ratio of sums, not a mean of per-sample ratios), so it
takes a `tokenizer_path` to count bytes. Lets follow-up experiments be compared across tokenizers.

## Usage

Metrics are instantiated in `MetricsCallback` via Hydra:

```yaml
# configs/callback/image_reconstruction_metrics.yaml
_target_: callback.metrics_callback.MetricsCallback
metrics:
  psnr:
    _target_: torchmetrics.image.PeakSignalNoiseRatio
  ssim:
    _target_: torchmetrics.image.StructuralSimilarityIndexMeasure
phases:
  val: 1
pred_key: post_processed/formatted_pred
target_key: target
```

Direct usage:

```python
from metrics.psnr import PSNR
metric = PSNR()
metric.update(pred, target)
value = metric.compute()
```
