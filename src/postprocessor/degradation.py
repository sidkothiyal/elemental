"""Degradation post-processor: iteratively re-encodes predictions through the model."""

from typing import Any

import torch

from postprocessor.base import BasePostProcessor


class DegradationPostProcessor(BasePostProcessor):
    """Run the model N times, feeding each output back as input.

    Measures reconstruction stability under repeated encoding/decoding.
    The trainer's initial forward pass counts as iteration 1, so this
    post-processor runs ``num_iterations - 1`` additional passes.

    Args:
        num_iterations: Total number of forward passes including the
            trainer's initial pass.  Must be >= 1.
    """

    def __init__(self, num_iterations: int = 100):
        if num_iterations < 1:
            raise ValueError(f"num_iterations must be >= 1, got {num_iterations}")
        self.num_iterations = num_iterations

    def __call__(  # type: ignore[override]
        self,
        model: Any,
        pred: torch.Tensor,
    ) -> torch.Tensor:
        for _ in range(self.num_iterations - 1):
            pred = model(pred)
        return pred
