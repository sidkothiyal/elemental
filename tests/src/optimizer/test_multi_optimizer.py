"""Unit tests for MultiOptimizer."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

import pytest
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.optim.optimizer import Optimizer

from optimizer.multi_optimizer import MultiOptimizer


def _make_two_optimizer_setup():
    torch.manual_seed(0)
    a = nn.Parameter(torch.randn(4, 4))
    b = nn.Parameter(torch.randn(4))
    opt_a = AdamW([a], lr=1e-2)
    opt_b = AdamW([b], lr=5e-3)
    multi = MultiOptimizer({"a": opt_a, "b": opt_b})
    return multi, a, b, opt_a, opt_b


@pytest.mark.unit
class TestMultiOptimizerBasics:
    def test_is_torch_optimizer_subclass(self):
        # PyTorch LR schedulers gate on ``isinstance(_, Optimizer)``.
        multi, *_ = _make_two_optimizer_setup()
        assert isinstance(multi, Optimizer)

    def test_param_groups_aggregates_in_order(self):
        multi, a, b, *_ = _make_two_optimizer_setup()
        assert len(multi.param_groups) == 2
        assert multi.param_groups[0]["params"][0] is a
        assert multi.param_groups[1]["params"][0] is b

    def test_step_updates_both_groups(self):
        multi, a, b, *_ = _make_two_optimizer_setup()
        a_before = a.detach().clone()
        b_before = b.detach().clone()
        a.grad = torch.ones_like(a)
        b.grad = torch.ones_like(b)
        multi.step()
        assert not torch.equal(a.detach(), a_before)
        assert not torch.equal(b.detach(), b_before)

    def test_zero_grad_clears_all(self):
        multi, a, b, *_ = _make_two_optimizer_setup()
        a.grad = torch.ones_like(a)
        b.grad = torch.ones_like(b)
        multi.zero_grad(set_to_none=True)
        assert a.grad is None
        assert b.grad is None

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="at least one"):
            MultiOptimizer({})

    def test_rejects_non_optimizer(self):
        with pytest.raises(TypeError, match="not a torch Optimizer"):
            MultiOptimizer({"bad": object()})  # type: ignore[arg-type]


@pytest.mark.unit
class TestStateDictRoundTrip:
    def test_round_trip_preserves_state(self):
        multi, a, b, opt_a, opt_b = _make_two_optimizer_setup()
        # Take a step so AdamW's first/second moment buffers are populated.
        a.grad = torch.ones_like(a)
        b.grad = torch.ones_like(b)
        multi.step()
        sd = multi.state_dict()

        # Build a fresh setup and load.
        torch.manual_seed(0)
        a2 = nn.Parameter(torch.randn(4, 4))
        b2 = nn.Parameter(torch.randn(4))
        multi2 = MultiOptimizer({"a": AdamW([a2], lr=1e-2), "b": AdamW([b2], lr=5e-3)})
        multi2.load_state_dict(sd)

        # Verify state is non-empty for both sub-optimizers.
        for sub in multi2.optimizers.values():
            assert len(sub.state) > 0


@pytest.mark.unit
class TestSchedulerCompatibility:
    def test_step_lr_scales_each_sub_optimizer(self):
        multi, *_ = _make_two_optimizer_setup()
        sched = StepLR(multi, step_size=1, gamma=0.5)
        lrs_before = [g["lr"] for g in multi.param_groups]
        sched.step()
        lrs_after = [g["lr"] for g in multi.param_groups]
        for before, after in zip(lrs_before, lrs_after):
            assert after == pytest.approx(before * 0.5)


@pytest.mark.unit
class TestAddParamGroup:
    def test_routes_to_default_group(self):
        multi, *_ = _make_two_optimizer_setup()
        new_param = nn.Parameter(torch.randn(2, 2))
        multi.add_param_group({"params": [new_param], "lr": 1e-3})
        # Default is the last sub-optimizer ("b" here).
        assert any(
            any(p is new_param for p in g["params"]) for g in multi.optimizers["b"].param_groups
        )

    def test_routes_by_optimizer_name(self):
        multi, *_ = _make_two_optimizer_setup()
        new_param = nn.Parameter(torch.randn(2, 2))
        multi.add_param_group({"params": [new_param], "lr": 1e-3, "optimizer_name": "a"})
        assert any(
            any(p is new_param for p in g["params"]) for g in multi.optimizers["a"].param_groups
        )

    def test_unknown_optimizer_name_raises(self):
        multi, *_ = _make_two_optimizer_setup()
        new_param = nn.Parameter(torch.randn(2, 2))
        with pytest.raises(ValueError, match="unknown optimizer_name"):
            multi.add_param_group({"params": [new_param], "lr": 1e-3, "optimizer_name": "nope"})
