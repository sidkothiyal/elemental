"""Argmax post-processor for classification predictions."""

import torch

from postprocessor.base import BasePostProcessor


class ArgmaxPostProcessor(BasePostProcessor):
    """Apply argmax to a prediction tensor.

    Useful for classification tasks where the model outputs logits and
    callbacks / metrics expect class indices.

    Args:
        dim: Dimension along which to compute argmax (default: 1, i.e.
            the class dimension for typical ``[B, C, ...]`` tensors).
    """

    def __init__(self, dim: int = 1):
        self.dim = dim

    def __call__(self, pred: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if pred.dim() > 1:
            return pred.argmax(dim=self.dim)
        return pred
