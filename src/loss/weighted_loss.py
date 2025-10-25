"""Weighted loss."""

from typing import List, Union

import torch
import torch.nn as nn


class WeightedLoss(nn.Module):
    """Weighted loss."""

    def __init__(self, loss_fns: List[nn.Module], weights: Union[List[float], torch.Tensor]):
        """Initialize the WeightedLoss.

        Args:
            loss_fns: The list of loss functions.
            weights: The list of weights.
        """
        super().__init__()
        self.loss_fns = loss_fns
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights)
        self.weights: torch.Tensor = weights
        if len(self.loss_fns) != len(self.weights):
            raise ValueError("The number of loss functions and weights must be the same")

    def forward(self, pred: torch.Tensor, target: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass.

        Args:
            pred: The predicted tensor.
            target: The target tensor.
            *args: The additional arguments.
            **kwargs: The additional keyword arguments.
        """
        self.weights = self.weights.to(pred.device)
        return sum(
            weight * loss_fn(pred, target, *args, **kwargs)
            for weight, loss_fn in zip(self.weights, self.loss_fns)
        )


if __name__ == "__main__":
    loss_fns = [nn.MSELoss(), nn.L1Loss()]
    weights = [0.5, 0.5]
    loss = WeightedLoss(loss_fns, weights)
    pred = torch.randn(2, 3, 480, 640)
    target = torch.randn(2, 3, 480, 640)
    loss = loss(pred, target)
    print(loss)
