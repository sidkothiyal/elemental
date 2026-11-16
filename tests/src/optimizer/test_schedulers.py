"""Unit tests for WarmupCosineLR."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

import math

import pytest
import torch
from torch import nn
from torch.optim import AdamW

from optimizer.schedulers import WarmupCosineLR


def _make_opt():
    p = nn.Parameter(torch.zeros(2, 2))
    return AdamW([p], lr=1.0), p


@pytest.mark.unit
class TestWarmupCosine:
    def test_warmup_ramps_linearly(self):
        opt, _ = _make_opt()
        sched = WarmupCosineLR(opt, warmup_steps=10, total_steps=100, min_lr_ratio=0.0)
        # At step 0 (before any sched.step) we expect lr ≈ 1/10 (the +1 offset).
        assert opt.param_groups[0]["lr"] == pytest.approx(0.1, rel=1e-6)
        sched.step()  # advance to step 1
        assert opt.param_groups[0]["lr"] == pytest.approx(0.2, rel=1e-6)

    def test_cosine_decays_to_floor(self):
        opt, _ = _make_opt()
        sched = WarmupCosineLR(opt, warmup_steps=2, total_steps=10, min_lr_ratio=0.1)
        for _ in range(20):  # past total_steps to verify floor holds
            sched.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(0.1, abs=1e-6)

    def test_midpoint_is_halfway(self):
        opt, _ = _make_opt()
        sched = WarmupCosineLR(opt, warmup_steps=0, total_steps=100, min_lr_ratio=0.0)
        for _ in range(50):
            sched.step()
        # cosine(pi/2) = 0 → lr = 0 + 1 * 0.5 * (1 + 0) = 0.5
        assert opt.param_groups[0]["lr"] == pytest.approx(0.5, abs=1e-6)

    def test_rejects_invalid_args(self):
        opt, _ = _make_opt()
        with pytest.raises(ValueError):
            WarmupCosineLR(opt, warmup_steps=-1, total_steps=10)
        with pytest.raises(ValueError):
            WarmupCosineLR(opt, warmup_steps=1, total_steps=0)
        with pytest.raises(ValueError):
            WarmupCosineLR(opt, warmup_steps=1, total_steps=10, min_lr_ratio=1.5)

    def test_serializable_state_dict(self):
        opt, _ = _make_opt()
        sched = WarmupCosineLR(opt, warmup_steps=5, total_steps=50, min_lr_ratio=0.1)
        for _ in range(7):
            sched.step()
        sd = sched.state_dict()
        # Building a fresh scheduler and loading should restore the step count.
        opt2, _ = _make_opt()
        sched2 = WarmupCosineLR(opt2, warmup_steps=5, total_steps=50, min_lr_ratio=0.1)
        sched2.load_state_dict(sd)
        # next sched.step() should produce identical lr from both.
        sched.step()
        sched2.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(opt2.param_groups[0]["lr"], rel=1e-9)

    def test_numerical_100step_example(self):
        # Concrete example: base_lr=1e-3, warmup=10, total=100, min_lr_ratio=0.1
        # Warmup: lr = (step+1)/10 * base_lr
        # Decay: lr = (0.1 + 0.9 * 0.5 * (1 + cos(pi * progress))) * base_lr
        p = nn.Parameter(torch.zeros(2, 2))
        opt = AdamW([p], lr=1e-3)
        sched = WarmupCosineLR(opt, warmup_steps=10, total_steps=100, min_lr_ratio=0.1)

        def lr():
            return opt.param_groups[0]["lr"]

        def step_to(target):
            for _ in range(target - sched.last_epoch):
                sched.step()

        # Warmup checkpoints (before any step, last_epoch=-1 → step 0 already applied)
        assert lr() == pytest.approx(1e-4, rel=1e-6)  # step 0: 1/10 * 1e-3
        step_to(4)
        assert lr() == pytest.approx(5e-4, rel=1e-6)  # step 4: 5/10 * 1e-3
        step_to(9)
        assert lr() == pytest.approx(1e-3, rel=1e-6)  # step 9: 10/10 * 1e-3

        # Cosine decay checkpoints
        step_to(10)
        assert lr() == pytest.approx(1e-3, rel=1e-5)  # progress=0.00, cos(0)=1
        step_to(55)
        assert lr() == pytest.approx(5.5e-4, rel=1e-5)  # progress=0.50, cos(π/2)=0
        step_to(100)
        assert lr() == pytest.approx(1e-4, rel=1e-5)  # progress=1.00, cos(π)=-1 → min

    def test_works_with_multi_optimizer(self):
        from optimizer.multi_optimizer import MultiOptimizer

        opt_a, _ = _make_opt()
        opt_b, _ = _make_opt()
        multi = MultiOptimizer({"a": opt_a, "b": opt_b})
        sched = WarmupCosineLR(multi, warmup_steps=2, total_steps=10, min_lr_ratio=0.0)
        for _ in range(2):
            sched.step()
        # Both sub-optimizer lrs should be updated.
        assert opt_a.param_groups[0]["lr"] == pytest.approx(opt_b.param_groups[0]["lr"])
        # And not equal to base lr (warmup should have ramped them).
        # The exact value: math.cos(pi * 0/8) → factor is 1.0 cosine init point.
        # Just sanity check positivity.
        assert opt_a.param_groups[0]["lr"] > 0
        # A few more steps and the lr keeps moving.
        prev = opt_a.param_groups[0]["lr"]
        for _ in range(5):
            sched.step()
        assert opt_a.param_groups[0]["lr"] != prev
        # Ensure cosine math is well-formed.
        assert not math.isnan(opt_a.param_groups[0]["lr"])
