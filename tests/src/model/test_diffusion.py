"""Unit tests for model.diffusion.DenoisingDiffusionModel.

Key note: since commit 1289bc7, forward() returns a dict:
  training mode → {"pred_noise": Tensor, "pred_x0": Tensor}
  eval mode     → {"pred_x0": Tensor}  (after iterative denoising)
"""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

import pytest
import torch
from torch import nn

from model.diffusion import DenoisingDiffusionModel


class _TinyDenoiser(nn.Module):
    """Minimal denoiser: same input/output shape, ignores t."""

    def __init__(self, channels=3):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x, t=None, **kwargs):  # pylint: disable=unused-argument
        return self.conv(x)


def _make_model(channels=3, num_timesteps=10, **kwargs):
    return DenoisingDiffusionModel(
        model=_TinyDenoiser(channels),
        num_timesteps=num_timesteps,
        **kwargs,
    )


@pytest.mark.unit
class TestDenoisingDiffusionModelForward:
    def test_train_forward_returns_dict(self):
        model = _make_model()
        model.train()
        x = torch.randn(2, 3, 16, 16)
        t = torch.randint(0, 10, (2,))
        out = model(x, t)
        assert isinstance(out, dict), "forward() must return a dict"
        assert "pred_noise" in out
        assert "pred_x0" in out

    def test_train_output_shapes(self):
        model = _make_model()
        model.train()
        x = torch.randn(2, 3, 16, 16)
        t = torch.randint(0, 10, (2,))
        out = model(x, t)
        assert out["pred_noise"].shape == x.shape
        assert out["pred_x0"].shape == x.shape

    def test_eval_forward_returns_dict_with_pred_x0(self):
        model = _make_model(num_timesteps=10, num_denoising_steps=3)
        model.eval()
        x = torch.randn(2, 3, 16, 16)
        with torch.no_grad():
            out = model(x)
        assert isinstance(out, dict)
        assert "pred_x0" in out
        assert out["pred_x0"].shape == x.shape

    def test_gradient_flows_in_training(self):
        model = _make_model()
        model.train()
        x = torch.randn(2, 3, 16, 16)
        t = torch.randint(0, 10, (2,))
        out = model(x, t)
        (out["pred_noise"].sum() + out["pred_x0"].sum()).backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"


@pytest.mark.unit
class TestNoiseSchedules:
    def test_linear_schedule_creates_model(self):
        """Linear schedule should construct without error."""
        model = _make_model(noise_schedule="linear", beta_start=1e-4, beta_end=0.02)
        # sqrt_alphas_cumprod is stored — verify it's monotonically decreasing
        assert model.sqrt_alphas_cumprod[0] >= model.sqrt_alphas_cumprod[-1]

    def test_cosine_schedule_creates_model(self):
        """Cosine schedule should construct without error."""
        model = _make_model(noise_schedule="cosine")
        assert model.sqrt_alphas_cumprod is not None

    def test_unknown_schedule_raises(self):
        with pytest.raises(ValueError):
            _make_model(noise_schedule="unknown_schedule")


@pytest.mark.unit
class TestDirectPrediction:
    def test_direct_prediction_mode(self):
        model = _make_model(direct_prediction=True)
        model.eval()
        x = torch.randn(2, 3, 16, 16)
        t = torch.randint(0, 10, (2,))
        with torch.no_grad():
            out = model(x, t)
        assert isinstance(out, dict)
        assert "pred_x0" in out
