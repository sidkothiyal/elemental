"""Unit tests for model.combiner.Combiner and DiffusionCombiner."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

import pytest
import torch
from torch import nn

from model.combiner import Combiner


@pytest.mark.unit
class TestCombiner:
    def test_forward_basic(self):
        """Combiner with a simple cat+linear combiner."""

        # Use an Add combiner (simple: just add x + y)
        class AddCombiner(nn.Module):
            def forward(self, x, y):
                return x + y

        model = Combiner(combiner=AddCombiner())
        x = torch.randn(2, 8)
        y = torch.randn(2, 8)
        out = model(x, y)
        assert out.shape == (2, 8)
        assert torch.allclose(out, x + y)

    def test_with_x_and_y_models(self):
        """Pre-processes x and y before combining."""

        class AddCombiner(nn.Module):
            def forward(self, x, y):
                return x + y

        model = Combiner(
            combiner=AddCombiner(),
            x_model=nn.Linear(4, 8),
            y_model=nn.Linear(4, 8),
        )
        x = torch.randn(2, 4)
        y = torch.randn(2, 4)
        out = model(x, y)
        assert out.shape == (2, 8)

    def test_gradient_flows(self):
        class AddCombiner(nn.Module):
            def forward(self, x, y):
                return x + y

        model = Combiner(
            combiner=AddCombiner(),
            x_model=nn.Linear(4, 8),
            y_model=nn.Linear(4, 8),
            model=nn.Linear(8, 4),
        )
        x = torch.randn(2, 4)
        y = torch.randn(2, 4)
        model(x, y).sum().backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"
