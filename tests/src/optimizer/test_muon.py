"""Unit tests for Muon and Newton-Schulz orthogonalization."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

import pytest
import torch
from torch import nn

from optimizer.muon import Muon, zeropower_via_newtonschulz5


@pytest.mark.unit
class TestNewtonSchulz:
    def test_returns_near_orthogonal(self):
        # For a square matrix, the orthogonalized step should have singular
        # values close to 1.
        torch.manual_seed(0)
        g = torch.randn(32, 32)
        x = zeropower_via_newtonschulz5(g, steps=5)
        sigmas = torch.linalg.svdvals(x.float())
        # All singular values should be roughly in [0.5, 1.5] after 5 NS iterations.
        assert sigmas.min() > 0.5
        assert sigmas.max() < 1.5

    def test_handles_tall_matrix(self):
        # Tall matrices are transposed internally; output shape must match input.
        g = torch.randn(64, 16)
        x = zeropower_via_newtonschulz5(g, steps=5)
        assert x.shape == g.shape

    def test_handles_wide_matrix(self):
        g = torch.randn(16, 64)
        x = zeropower_via_newtonschulz5(g, steps=5)
        assert x.shape == g.shape

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError, match="2-D"):
            zeropower_via_newtonschulz5(torch.randn(8), steps=3)


@pytest.mark.unit
class TestMuonStep:
    def test_step_decreases_quadratic_loss(self):
        # On a simple quadratic loss with a 2-D weight, a few Muon steps should
        # drive the loss down monotonically.
        torch.manual_seed(0)
        w = nn.Parameter(torch.randn(8, 8))
        target = torch.zeros(8, 8)
        opt = Muon([w], lr=0.1, momentum=0.9, ns_steps=5)
        losses = []
        for _ in range(20):
            loss = ((w - target) ** 2).sum()
            losses.append(float(loss))
            opt.zero_grad()
            loss.backward()
            opt.step()
        assert losses[-1] < losses[0] * 0.5

    def test_rejects_non_2d_param(self):
        v = nn.Parameter(torch.randn(8))
        opt = Muon([v], lr=0.01)
        v.grad = torch.zeros_like(v)
        with pytest.raises(ValueError, match="2-D"):
            opt.step()

    def test_skips_param_with_no_grad(self):
        w = nn.Parameter(torch.randn(4, 4))
        opt = Muon([w], lr=0.1)
        before = w.detach().clone()
        opt.step()  # grad is None — no-op.
        torch.testing.assert_close(w.detach(), before)

    def test_weight_decay_shrinks_param(self):
        w = nn.Parameter(torch.full((4, 4), 1.0))
        opt = Muon([w], lr=0.01, momentum=0.0, weight_decay=0.5, ns_steps=1)
        w.grad = torch.zeros_like(w)
        opt.step()
        # With zero gradient + decoupled WD, p ← p * (1 - lr * wd) = 0.995.
        torch.testing.assert_close(w.detach(), torch.full((4, 4), 0.995), atol=1e-6, rtol=1e-6)
