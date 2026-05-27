"""Unit tests for model.quantizer.Quantizer (VQ bottleneck)."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch

from model.quantizer import Quantizer


@pytest.mark.unit
class TestQuantizer:
    def _make(self, codebook_size=16, codebook_dim=8):
        return Quantizer(
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            smart_init_enabled=False,  # disable smart init for deterministic tests
            codebook_revival_enabled=False,
            diversity_loss_enabled=False,
        )

    def test_output_shape_4d(self):
        q = self._make()
        x = torch.randn(2, 8, 4, 4)
        out = q(x)
        assert out.shape == x.shape

    def test_output_shape_3d(self):
        q = self._make()
        x = torch.randn(2, 6, 8)
        out = q(x)
        assert out.shape == x.shape

    def test_straight_through_gradient(self):
        """Gradient should flow back through the quantizer via straight-through estimator."""
        q = self._make(codebook_size=8, codebook_dim=4)
        q.eval()  # disable smart init updates
        x = torch.randn(2, 4, 2, 2, requires_grad=True)
        out = q(x)
        out.sum().backward()
        assert x.grad is not None
