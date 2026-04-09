"""Convergence tests: verify that training actually reduces loss.

Uses simple manual training loops for speed and reliability.
All tests are marked @pytest.mark.e2e.
"""

import pytest
import torch
from conftest import make_small_autoregressive_lm, make_small_unet
from torch import nn


@pytest.mark.e2e
class TestConvergence:
    """Verify that model training actually reduces loss on synthetic data."""

    def test_unet_loss_decreases(self) -> None:
        """UNet autoencoder loss should decrease over 30 steps on synthetic data."""
        model = make_small_unet()
        model.train()
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        losses = []
        for _ in range(30):
            x = torch.randn(2, 3, 32, 32)
            pred = model(x)
            loss = loss_fn(pred, x)  # autoencoder: reconstruct input
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        avg_first_5 = sum(losses[:5]) / 5
        avg_last_5 = sum(losses[-5:]) / 5
        assert (
            avg_last_5 < avg_first_5 * 0.9
        ), f"UNet loss didn't decrease enough: {avg_first_5:.4f} -> {avg_last_5:.4f}"

    def test_transformer_lm_loss_decreases(self) -> None:
        """Autoregressive LM loss should decrease over 30 steps."""
        model = make_small_autoregressive_lm(max_length=32)
        model.train()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

        # Use fixed data so model can overfit (proves convergence)
        fixed_tokens = torch.randint(0, 65, (2, 32))
        losses = []
        for _ in range(50):
            tokens = fixed_tokens
            logits = model(tokens[:, :-1], training=True)
            loss = loss_fn(logits.reshape(-1, 65), tokens[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        avg_first_5 = sum(losses[:5]) / 5
        avg_last_5 = sum(losses[-5:]) / 5
        assert (
            avg_last_5 < avg_first_5 * 0.95
        ), f"LM loss didn't decrease enough: {avg_first_5:.4f} -> {avg_last_5:.4f}"
