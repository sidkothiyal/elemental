"""Peak Signal-to-Noise Ratio (PSNR) metric."""

import warnings

import torch

from metrics.base_metric import BaseMetric


class PSNR(BaseMetric):  # pylint: disable=too-few-public-methods
    """Peak Signal-to-Noise Ratio metric for image reconstruction tasks.

    PSNR is computed as: 10 * log10(data_range^2 / MSE)
    Higher values indicate better reconstruction quality.
    Typical values for 8-bit images: 20-40 dB.

    Inherits optional extended statistics (min, max, std, median, percentiles,
    histogram) from :class:`~metrics.base_metric.BaseMetric`.
    """

    higher_is_better = True

    def __init__(
        self,
        data_range: float = 1.0,
        eps: float = 1e-8,
        reduction: str = "elementwise_mean",
        **kwargs,
    ):
        """Initialize the PSNR metric.

        Args:
            data_range: The data range of the input images (max - min).
                Default is 1.0 for normalized images [0, 1].
                Use 255.0 for uint8 images [0, 255].
            eps: Small constant to avoid log(0).
            reduction: Deprecated. Previously controlled batch reduction;
                now always uses elementwise_mean via the base class.
            **kwargs: Additional keyword arguments passed to
                :class:`~metrics.base_metric.BaseMetric` (e.g.
                ``compute_min``, ``compute_histogram``, ``num_buckets``).
        """
        super().__init__(**kwargs)
        self.data_range = data_range
        self.eps = eps

        if reduction != "elementwise_mean":
            warnings.warn(
                "The 'reduction' parameter of PSNR is deprecated and has no effect. "
                "Per-sample PSNR values are always accumulated and the mean is returned.",
                DeprecationWarning,
                stacklevel=2,
            )

    def compute_sample_metric(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute per-sample PSNR.

        Args:
            preds: Predicted tensor of shape (N, C, H, W) or (N, H, W).
            target: Target tensor of same shape as preds.

        Returns:
            1-D tensor of shape (N,) with per-sample PSNR values in dB.
        """
        if preds.shape != target.shape:
            raise ValueError(
                f"Predictions and targets must have the same shape. "
                f"Got preds: {preds.shape}, target: {target.shape}"
            )

        preds = preds.float()
        target = target.float()

        batch_size = preds.shape[0]
        preds_flat = preds.reshape(batch_size, -1)
        target_flat = target.reshape(batch_size, -1)

        mse = torch.mean((preds_flat - target_flat) ** 2, dim=1)
        return 10 * torch.log10((self.data_range**2) / (mse + self.eps))
