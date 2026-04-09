"""Shared helpers for model block tests."""

import torch
from torch import nn


def assert_gradients_flow(block: nn.Module, x: torch.Tensor) -> None:
    """Assert that gradients reach every trainable parameter of *block*."""
    block(x).sum().backward()
    for name, p in block.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"
