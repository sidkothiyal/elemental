"""Tensor-slice post-processor for extracting a contiguous slice from tensors."""

import torch

from postprocessor.base import BasePostProcessor


class TensorSlicePostProcessor(BasePostProcessor):
    """Extract a contiguous slice of dimensions from a tensor.

    Commonly used to get world_vector (indices 0:3) or rotation_delta (indices 3:6)
    from 7-DOF action tensors: 3 position + 3 rotation + 1 gripper.
    Input shape: (batch_size, num_chunks, num_actions) or similar.

    Returns tensor of shape (batch_size, num_chunks, slice_end - slice_start).
    """

    def __init__(self, slice_start: int = 0, slice_end: int = 3):
        """Initialize the slice indices.

        Args:
            slice_start: Start index (inclusive). E.g. 0 for world_vector, 3 for rotation_delta.
            slice_end: End index (exclusive). E.g. 3 for world_vector, 6 for rotation_delta.
        """
        if slice_start < 0 or slice_end <= slice_start:
            raise ValueError(
                f"slice_start ({slice_start}) must be non-negative and slice_end ({slice_end}) "
                "must be greater than slice_start"
            )
        self._slice_start = slice_start
        self._slice_end = slice_end

    def __call__(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Return x[..., slice_start:slice_end].

        Uses slicing (a view), so gradients backpropagate to x. .contiguous()
        preserves the computation graph when a copy is needed.

        Args:
            x: Tensor with shape (..., N) where N >= slice_end.

        Returns:
            Tensor of shape (..., slice_end - slice_start).
        """
        return x[..., self._slice_start : self._slice_end].contiguous()
