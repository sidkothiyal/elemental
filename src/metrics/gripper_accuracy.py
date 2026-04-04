"""Gripper L1 error metric for VLA action predictions."""

import torch

from metrics.base_metric import BaseMetric


class GripperL1(BaseMetric):
    """L1 error on the gripper closedness dimension.

    Extracts the gripper dimension from predicted and target action tensors
    and computes the mean absolute error across all chunks for each sample.

    Accepts tensors of shape `(B, num_chunks, action_dim)` or `(B, action_dim)`.

    Args:
        gripper_idx: Index of the gripper dimension in the action vector.
            Default `-1` (last dimension).
        **kwargs: Forwarded to :class:`BaseMetric` (e.g. `compute_min`,
            `compute_std`, `compute_histogram`).
    """

    higher_is_better = False

    def __init__(self, gripper_idx: int = -1, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.gripper_idx = gripper_idx

    def compute_sample_metric(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute per-sample gripper L1 error.

        Args:
            preds: Predicted actions of shape `(B, ...)` where the last
                dimension contains the gripper index.
            target: Target actions, same shape as `preds`.

        Returns:
            1-D tensor of shape `(B,)` with mean L1 error per sample.
        """
        pred_gripper = preds[..., self.gripper_idx].float()
        target_gripper = target[..., self.gripper_idx].float()

        # Per-sample mean L1: flatten non-batch dims then average
        errors = torch.abs(pred_gripper - target_gripper).reshape(preds.shape[0], -1)
        return errors.mean(dim=1)
