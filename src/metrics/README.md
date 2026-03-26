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
