"""Displacement error metrics for action chunk predictions.

ADE (Average Displacement Error): Mean L2 distance across all chunks.
FDE (Final Displacement Error): L2 distance at the last chunk.

Both operate on the position component of actions (first 3 dims by default).
"""

import torch

from metrics.base_metric import BaseMetric


def _prepare_displacement_inputs(
    preds: torch.Tensor, target: torch.Tensor, pos_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ensure 3-D shape and slice to position dimensions.

    Args:
        preds: Predicted actions of shape `(B, num_chunks, action_dim)` or `(B, action_dim)`.
        target: Target actions, same shape as `preds`.
        pos_dim: Number of leading action dimensions to keep. `-1` keeps all.

    Returns:
        Tuple of `(preds, target)` each with shape `(B, num_chunks, D)`.
    """
    if preds.ndim == 2:
        preds = preds.unsqueeze(1)
        target = target.unsqueeze(1)

    preds = preds.float()
    target = target.float()

    if pos_dim > 0:
        preds = preds[..., :pos_dim]
        target = target[..., :pos_dim]

    return preds, target


class ActionADE(BaseMetric):
    """Average Displacement Error across all action chunks.

    Computes per-chunk L2 distance between predicted and target actions,
    then averages across chunks for each sample. Uses only the position
    dimensions (first `pos_dim` dims) by default.

    Args:
        pos_dim: Number of position dimensions to use for L2 (default: `3`,
            i.e. world_vector). Set to `-1` to use all action dims.
        **kwargs: Forwarded to :class:`BaseMetric` (e.g. `compute_min`,
            `compute_std`, `compute_histogram`).
    """

    higher_is_better = False

    def __init__(self, pos_dim: int = 3, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.pos_dim = pos_dim

    def compute_sample_metric(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute per-sample average displacement error.

        Args:
            preds: Predicted actions of shape `(B, num_chunks, action_dim)` or `(B, action_dim)`.
            target: Target actions, same shape as `preds`.

        Returns:
            1-D tensor of shape `(B,)` with ADE per sample.
        """
        preds, target = _prepare_displacement_inputs(preds, target, self.pos_dim)
        per_chunk_l2 = torch.norm(preds - target, dim=-1)  # (B, num_chunks)
        return per_chunk_l2.mean(dim=-1)  # (B,)


class ActionFDE(BaseMetric):
    """Final Displacement Error at the last action chunk.

    Computes L2 distance between predicted and target at the final
    chunk position. Uses only the position dimensions by default.

    Args:
        pos_dim: Number of position dimensions (default: `3`). Set to `-1`
            to use all action dims.
        **kwargs: Forwarded to :class:`BaseMetric` (e.g. `compute_min`,
            `compute_std`, `compute_histogram`).
    """

    higher_is_better = False

    def __init__(self, pos_dim: int = 3, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.pos_dim = pos_dim

    def compute_sample_metric(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute per-sample final displacement error.

        Args:
            preds: Predicted actions of shape `(B, num_chunks, action_dim)` or `(B, action_dim)`.
            target: Target actions, same shape as `preds`.

        Returns:
            1-D tensor of shape `(B,)` with FDE per sample.
        """
        preds, target = _prepare_displacement_inputs(preds, target, self.pos_dim)
        return torch.norm(preds[:, -1] - target[:, -1], dim=-1)  # (B,)
