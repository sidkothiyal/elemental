"""Unit tests for loss.loss_container.LossComponent and LossContainer."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

import pytest
import torch
from torch import nn

from loss.loss_container import LossComponent, LossContainer


@pytest.mark.unit
class TestLossComponent:
    def test_forward_resolves_keys(self):
        comp = LossComponent(loss_fn=nn.MSELoss(), pred_key="pred", target_key="target")
        data = {
            "pred": torch.ones(2, 3),
            "target": torch.zeros(2, 3),
        }
        loss = comp(data)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar
        assert loss.item() == pytest.approx(1.0)

    def test_weight_scaling(self):
        comp = LossComponent(loss_fn=nn.MSELoss(), weight=2.0)
        data = {"pred": torch.ones(2, 3), "target": torch.zeros(2, 3)}
        unweighted = LossComponent(loss_fn=nn.MSELoss(), weight=1.0)
        assert comp(data).item() == pytest.approx(unweighted(data).item() * 2.0, rel=1e-5)

    def test_context_mapping_kwargs(self):
        class _LossWithMask(nn.Module):
            def forward(self, pred, target, mask=None):
                if mask is not None:
                    return ((pred - target) ** 2 * mask).mean()
                return ((pred - target) ** 2).mean()

        comp = LossComponent(
            loss_fn=_LossWithMask(),
            pred_key="pred",
            target_key="target",
            context_mapping={"target/mask": "mask"},
        )
        data = {
            "pred": torch.ones(2, 3),
            "target": torch.zeros(2, 3),
            "target/mask": torch.ones(2, 3),
        }
        loss = comp(data)
        assert loss.item() == pytest.approx(1.0)

    def test_missing_key_raises(self):
        comp = LossComponent(loss_fn=nn.MSELoss(), pred_key="missing_key")
        with pytest.raises(KeyError):
            comp({"target": torch.zeros(2)})

    def test_gradient_flows(self):
        comp = LossComponent(loss_fn=nn.MSELoss())
        pred = torch.ones(2, 3, requires_grad=True)
        data = {"pred": pred, "target": torch.zeros(2, 3)}
        comp(data).backward()
        assert pred.grad is not None


@pytest.mark.unit
class TestLossContainer:
    def _make_container(self):
        return LossContainer(
            losses={
                "mse": LossComponent(nn.MSELoss(), weight=1.0),
                "l1": LossComponent(nn.L1Loss(), weight=0.5),
            }
        )

    def test_aggregates_multiple_components(self):
        container = self._make_container()
        data = {"pred": torch.ones(2, 3), "target": torch.zeros(2, 3)}
        total, _ = container(data)
        assert isinstance(total, torch.Tensor)
        assert total.ndim == 0

    def test_total_is_weighted_sum(self):
        container = self._make_container()
        data = {"pred": torch.ones(2, 3), "target": torch.zeros(2, 3)}
        total, _ = container(data)
        # MSE(ones, zeros) = 1.0, L1(ones, zeros) = 1.0
        # total = 1.0*1.0 + 0.5*1.0 = 1.5
        assert total.item() == pytest.approx(1.5, rel=1e-5)

    def test_gradient_flows(self):
        container = self._make_container()
        pred = torch.ones(2, 3, requires_grad=True)
        data = {"pred": pred, "target": torch.zeros(2, 3)}
        total, _ = container(data)
        total.backward()
        assert pred.grad is not None
