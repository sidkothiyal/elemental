"""Golden value tests for loss functions.

Pin exact numeric outputs for known inputs to catch silent regressions.
All tests use torch.manual_seed(42) for deterministic inputs.
"""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch
from torch import nn

from loss.charbonnier_loss import CharbonnierLoss
from loss.gradient_loss import GradientLoss
from loss.loss_container import LossComponent, LossContainer
from loss.masked_loss import MaskedLoss
from loss.shifted_teacher_forcing import ShiftedTeacherForcing
from loss.ssim import SSIMLoss
from loss.weighted_loss import WeightedLoss


def _seeded_image_pair(seed=42, batch=2, channels=3, height=32, width=32):
    """Return deterministic (pred, target) image tensors."""
    torch.manual_seed(seed)
    pred = torch.randn(batch, channels, height, width)
    target = torch.randn(batch, channels, height, width)
    return pred, target


def _seeded_sequence_pair(seed=42, batch=2, seq_len=16, vocab=65):
    """Return deterministic (pred_logits, target_ids) for sequence losses."""
    torch.manual_seed(seed)
    pred = torch.randn(batch, seq_len, vocab)
    target = torch.randint(0, vocab, (batch, seq_len))
    return pred, target


@pytest.mark.unit
@pytest.mark.golden
class TestLossGoldenValues:
    """Each test pins the loss value for a fixed seed. If a value changes,
    it means the loss computation changed — review before updating."""

    def test_charbonnier_golden(self):
        pred, target = _seeded_image_pair()
        loss_fn = CharbonnierLoss(eps=1e-3)
        result = loss_fn(pred, target).item()
        assert result == pytest.approx(
            1.1254781484603882, abs=1e-5
        ), f"CharbonnierLoss golden value changed: {result}"

    def test_gradient_loss_golden(self):
        pred, target = _seeded_image_pair()
        loss_fn = GradientLoss()
        result = loss_fn(pred, target).item()
        assert result == pytest.approx(
            3.179487705230713, abs=1e-5
        ), f"GradientLoss golden value changed: {result}"

    def test_ssim_loss_golden(self):
        pred, target = _seeded_image_pair()
        pred_clamped = pred.clamp(0, 1)
        target_clamped = target.clamp(0, 1)
        loss_fn = SSIMLoss(multi_scale=False)
        result = loss_fn(pred_clamped, target_clamped).item()
        assert result == pytest.approx(
            1.0062003135681152, abs=1e-4
        ), f"SSIMLoss golden value changed: {result}"

    def test_masked_loss_golden(self):
        pred, target = _seeded_image_pair()
        torch.manual_seed(99)
        mask = (torch.rand(2, 3, 32, 32) > 0.5).float()
        loss_fn = MaskedLoss(loss_function=nn.MSELoss(reduction="none"), mask_weight=1.0)
        result = loss_fn(pred, target, mask=mask).item()
        assert result == pytest.approx(
            1.9268214702606201, abs=1e-4
        ), f"MaskedLoss golden value changed: {result}"

    def test_shifted_teacher_forcing_golden(self):
        pred, target = _seeded_sequence_pair()
        loss_fn = ShiftedTeacherForcing(shift=1, loss_fn=nn.CrossEntropyLoss())
        result = loss_fn(pred, target).item()
        assert result == pytest.approx(
            4.845429420471191, abs=1e-4
        ), f"ShiftedTeacherForcing golden value changed: {result}"

    def test_weighted_loss_golden(self):
        pred, target = _seeded_image_pair()
        loss_fn = WeightedLoss(
            loss_fns=[nn.MSELoss(), nn.L1Loss()],
            weights=[0.7, 0.3],
        )
        result = loss_fn(pred, target).item()
        assert result == pytest.approx(
            1.7135831117630005, abs=1e-4
        ), f"WeightedLoss golden value changed: {result}"

    def test_loss_container_golden(self):
        pred, target = _seeded_image_pair()
        container = LossContainer(
            losses={
                "mse": LossComponent(loss_fn=nn.MSELoss(), pred_key="pred", target_key="target"),
            }
        )
        data = {"pred": pred, "target": target}
        total, _breakdown = container(data)
        assert total.item() == pytest.approx(
            1.9656291007995605, abs=1e-4
        ), f"LossContainer golden value changed: {total.item()}"
