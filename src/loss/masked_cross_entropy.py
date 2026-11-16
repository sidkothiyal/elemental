"""Masked cross-entropy for masked-language-modelling / masked-diffusion training.

Computes cross-entropy only at positions selected by a boolean ``mask`` —
typically the positions corrupted by the LLaDA-style masked-diffusion forward
process. Positions where ``mask`` is ``False`` are mapped to ``ignore_index``
and dropped by ``F.cross_entropy``.
"""

import torch
import torch.nn.functional as F
from torch import nn


class MaskedCrossEntropy(nn.Module):  # pylint: disable=too-few-public-methods
    """Cross-entropy averaged over masked positions only.

    Args:
        ignore_index: Target value treated as padding. The default ``-100``
            matches PyTorch's ``F.cross_entropy``. Positions where
            ``mask == 0`` are rewritten to this value internally so that they
            contribute nothing to the loss.
    """

    def __init__(self, ignore_index: int = -100) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            pred: Logits ``(B, T, V)``.
            target: Target token ids ``(B, T)``.
            mask: Boolean / 0-1 mask ``(B, T)`` selecting the positions that
                contribute to the loss.

        Returns:
            Scalar loss averaged over masked positions. If no positions are
            masked the loss is ``0`` (with gradient zero) — the caller is
            expected to ensure at least one masked position per global batch.
        """
        if pred.ndim != 3:
            raise ValueError(f"pred must be 3D (B, T, V), got shape {tuple(pred.shape)}")
        if target.shape != pred.shape[:2]:
            raise ValueError(
                f"target shape {tuple(target.shape)} must match pred[:, :, :] "
                f"prefix {tuple(pred.shape[:2])}"
            )
        if mask.shape != target.shape:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} must match target shape " f"{tuple(target.shape)}"
            )

        mask_bool = mask.bool()
        if not mask_bool.any():
            return pred.sum() * 0.0  # connect to graph; return zero loss

        target_masked = torch.where(mask_bool, target, torch.full_like(target, self.ignore_index))
        # F.cross_entropy expects (B, V, T) for tensor inputs.
        return F.cross_entropy(
            pred.transpose(1, 2),
            target_masked,
            ignore_index=self.ignore_index,
        )
