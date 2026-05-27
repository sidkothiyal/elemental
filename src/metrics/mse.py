"""Mean Squared Error (MSE) metric."""

import torch

from metrics.base_metric import BaseMetric


class MSE(BaseMetric):  # pylint: disable=too-few-public-methods
    """Mean Squared Error metric for image reconstruction tasks.

    Computes the mean squared error between predictions and targets.
    Lower values indicate better reconstruction quality.

    Inherits optional extended statistics (min, max, std, median, percentiles,
    histogram) from :class:`~metrics.base_metric.BaseMetric`.
    """

    higher_is_better = False

    def __init__(
        self,
        data_range: float | None = None,
        normalize: bool = False,
        **kwargs,
    ):
        """Initialize the MSE metric.

        Args:
            data_range: The data range of the input images (max - min).
                If provided and normalize=True, inputs will be normalized.
            normalize: Whether to normalize inputs by data_range before
                computing MSE. Useful when comparing images with different
                dynamic ranges.
            **kwargs: Additional keyword arguments passed to
                :class:`~metrics.base_metric.BaseMetric` (e.g.
                ``compute_min``, ``compute_histogram``, ``num_buckets``).
        """
        super().__init__(**kwargs)
        self.data_range = data_range
        self.normalize = normalize

    def compute_sample_metric(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute per-sample MSE.

        Args:
            preds: Predicted tensor of shape (N, C, H, W) or (N, H, W).
            target: Target tensor of same shape as preds.

        Returns:
            1-D tensor of shape (N,) with per-sample MSE values.
        """
        if preds.shape != target.shape:
            raise ValueError(
                f"Predictions and targets must have the same shape. "
                f"Got preds: {preds.shape}, target: {target.shape}"
            )

        preds = preds.float()
        target = target.float()

        if self.normalize and self.data_range is not None:
            preds = preds / self.data_range
            target = target / self.data_range

        batch_size = preds.shape[0]
        preds_flat = preds.reshape(batch_size, -1)
        target_flat = target.reshape(batch_size, -1)

        return torch.mean((preds_flat - target_flat) ** 2, dim=1)
