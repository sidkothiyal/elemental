"""Mean Squared Error (MSE) metric."""

import torch
from torchmetrics import Metric


class MSE(Metric):
    """Mean Squared Error metric for image reconstruction tasks.

    Computes the mean squared error between predictions and targets.
    Lower values indicate better reconstruction quality.
    """

    higher_is_better = False
    full_state_update = False

    def __init__(
        self,
        data_range: float | None = None,
        normalize: bool = False,
    ):
        """Initialize the MSE metric.

        Args:
            data_range: The data range of the input images (max - min).
                If provided and normalize=True, inputs will be normalized.
            normalize: Whether to normalize inputs by data_range before
                computing MSE. Useful when comparing images with different
                dynamic ranges.
        """
        super().__init__()
        self.data_range = data_range
        self.normalize = normalize

        self.add_state("sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric state with predictions and targets.

        Args:
            preds: Predicted tensor of shape (N, C, H, W) or (N, H, W).
            target: Target tensor of same shape as preds.
        """
        if preds.shape != target.shape:
            raise ValueError(
                f"Predictions and targets must have the same shape. "
                f"Got preds: {preds.shape}, target: {target.shape}"
            )

        # Convert to float for computation
        preds = preds.float()
        target = target.float()

        # Normalize if requested
        if self.normalize and self.data_range is not None:
            preds = preds / self.data_range
            target = target / self.data_range

        # Compute squared error per sample
        # Flatten spatial dimensions and compute mean per sample
        batch_size = preds.shape[0]
        preds_flat = preds.reshape(batch_size, -1)
        target_flat = target.reshape(batch_size, -1)

        squared_error = torch.mean((preds_flat - target_flat) ** 2, dim=1)

        self.sum_squared_error += squared_error.sum()
        self.total_samples += batch_size

    def compute(self) -> torch.Tensor:
        """Compute the final MSE value.

        Returns:
            Mean Squared Error averaged over all samples.
        """
        return self.sum_squared_error / self.total_samples
