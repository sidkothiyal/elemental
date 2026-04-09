"""Unit tests for Router module."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

import pytest
import torch
from torch import nn

from model.router import Router


class SimpleBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 16)

    def forward(self, x):
        return self.linear(x)


class SimpleHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 4)

    def forward(self, x):
        return self.linear(x)


@pytest.mark.unit
class TestRouter:
    def test_backbone_only(self):
        router = Router(backbone=SimpleBackbone())
        out = router(x=torch.randn(2, 8))
        assert "backbone" in out
        assert out["backbone"].shape == (2, 16)

    def test_backbone_and_head(self):
        router = Router(
            backbone=SimpleBackbone(),
            heads={"head1": SimpleHead()},
        )
        out = router(x=torch.randn(2, 8))
        assert "backbone" in out
        assert "head1" in out
        assert out["head1"].shape == (2, 4)

    def test_heads_to_run_subset(self):
        router = Router(
            backbone=SimpleBackbone(),
            heads={"h1": SimpleHead(), "h2": SimpleHead()},
        )
        out = router(x=torch.randn(2, 8), heads_to_run=["h1"])
        assert "h1" in out
        assert "h2" not in out

    def test_positional_arg(self):
        router = Router(backbone=SimpleBackbone())
        out = router(torch.randn(2, 8))
        assert "backbone" in out

    def test_rename_backbone_input(self):
        router = Router(
            backbone=SimpleBackbone(),
            rename_backbone_input={"x": "features"},
        )
        out = router(features=torch.randn(2, 8))
        assert "backbone" in out

    def test_rename_backbone_output(self):
        router = Router(
            backbone=SimpleBackbone(),
            rename_backbone_output="encoded",
        )
        out = router(x=torch.randn(2, 8))
        assert "encoded" in out
        assert "backbone" not in out

    def test_missing_context_key_raises(self):
        router = Router(
            backbone=SimpleBackbone(),
            rename_backbone_input={"x": "missing_key"},
        )
        with pytest.raises(KeyError, match="missing_key"):
            router(features=torch.randn(2, 8))

    def test_both_args_and_kwargs_raises(self):
        router = Router(backbone=SimpleBackbone())
        with pytest.raises(ValueError, match="both positional and keyword"):
            router(torch.randn(2, 8), x=torch.randn(2, 8))

    def test_gradient_flows(self):
        router = Router(
            backbone=SimpleBackbone(),
            heads={"head1": SimpleHead()},
        )
        x = torch.randn(2, 8, requires_grad=True)
        out = router(x=x)
        out["head1"].sum().backward()
        assert x.grad is not None
