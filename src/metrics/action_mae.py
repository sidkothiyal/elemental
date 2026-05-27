"""Mean Absolute Error metric for action predictions."""

import torch

from metrics.base_metric import BaseMetric


class ActionMAE(BaseMetric):
    """Mean Absolute Error for action chunk predictions.

    Computes the mean absolute error between predicted and target action
    chunks, averaged first across all dimensions within each sample, then
    across samples.

    Accepts tensors of shape `(B, num_chunks, action_dim)` or `(B, D)`.

    Args:
        **kwargs: Forwarded to :class:`BaseMetric` (e.g. `compute_min`,
            `compute_std`, `compute_histogram`).
    """

    higher_is_better = False

    def compute_sample_metric(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute per-sample mean absolute error.

        Args:
            preds: Predicted actions of shape `(B, ...)` .
            target: Target actions, same shape as `preds`.

        Returns:
            1-D tensor of shape `(B,)` with one MAE value per sample.
        """
        batch_size = preds.shape[0]
        preds_flat = preds.reshape(batch_size, -1).float()
        target_flat = target.reshape(batch_size, -1).float()
        return torch.mean(torch.abs(preds_flat - target_flat), dim=1)
