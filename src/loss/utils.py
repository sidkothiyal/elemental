"""Utility functions for losses."""

import inspect

import torch
from torch import nn


def call_loss_with_epoch(
    loss_fn: nn.Module, pred: torch.Tensor, target: torch.Tensor, epoch: int, **kwargs
) -> torch.Tensor:
    """Call a loss function with optional epoch parameter support.

    Args:
        loss_fn: The loss function to call
        pred: Predicted tensor
        target: Target tensor
        epoch: Current epoch
        **kwargs: Additional keyword arguments for the loss

    Returns:
        Loss value
    """
    # Check if loss function expects epoch parameter
    loss_signature = inspect.signature(loss_fn.forward)
    if "epoch" in loss_signature.parameters:
        return loss_fn(pred, target, epoch=epoch, **kwargs)
    return loss_fn(pred, target, **kwargs)
