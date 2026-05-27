"""Unit tests for GRPOLoss."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

import pytest
import torch

from loss.grpo_loss import GRPOLoss


@pytest.mark.unit
class TestGRPOLoss:
    def test_zero_advantage_zero_loss(self):
        # When all advantages are zero, the policy-gradient term vanishes.
        loss = GRPOLoss()
        log_prob_new = torch.randn(4, 6, requires_grad=True)
        log_prob_old = torch.randn(4, 6)
        adv = torch.zeros(4)
        mask = torch.ones(4, 6)
        out = loss(log_prob_new, log_prob_old, adv, mask, clip_range=0.2)
        assert out.item() == pytest.approx(0.0, abs=1e-6)

    def test_loss_is_masked(self):
        # Tokens with mask=0 must not contribute. Compare against a smaller
        # batch of only the masked-in tokens.
        loss = GRPOLoss()
        log_prob_new = torch.tensor([[0.5, 0.0, -0.1]])
        log_prob_old = torch.tensor([[0.0, 0.0, 0.0]])
        adv = torch.tensor([1.0])
        mask_full = torch.tensor([[1.0, 0.0, 0.0]])
        # The mask-zeroed tokens should NOT influence the loss.
        out_full = loss(log_prob_new, log_prob_old, adv, mask_full, clip_range=0.5)
        # Compare to manual: ratio = exp(0.5) ≈ 1.649, advantage = 1, no clipping
        # since 1.0 - 0.5 = 0.5 < 1.649 < 1.5 — clip applies, so clipped =
        # min(1.5, ratio) * adv = 1.5. Unclipped = 1.649. PG = -min(1.649, 1.5) = -1.5.
        assert out_full.item() == pytest.approx(-1.5, abs=1e-3)

    def test_kl_term_added(self):
        # When kl_coef > 0 and ref provided, k3 KL term should appear.
        loss_no_kl = GRPOLoss(kl_coef=0.0)
        loss_with_kl = GRPOLoss(kl_coef=1.0)
        log_prob_new = torch.tensor([[-1.0, -2.0]])
        log_prob_old = torch.tensor([[-1.0, -2.0]])
        log_prob_ref = torch.tensor([[-3.0, -3.0]])
        adv = torch.tensor([0.0])  # zeros out the PG term
        mask = torch.ones(1, 2)

        a = loss_no_kl(log_prob_new, log_prob_old, adv, mask, log_prob_ref=log_prob_ref)
        b = loss_with_kl(log_prob_new, log_prob_old, adv, mask, log_prob_ref=log_prob_ref)
        assert a.item() == pytest.approx(0.0, abs=1e-6)
        # k3 KL is non-negative; with a non-trivial ref it must be > 0.
        assert b.item() > 0.0

    def test_gradient_flows_through_log_prob_new(self):
        loss = GRPOLoss()
        log_prob_new = torch.randn(2, 3, requires_grad=True)
        log_prob_old = torch.randn(2, 3)
        adv = torch.tensor([1.0, -1.0])
        mask = torch.ones(2, 3)
        out = loss(log_prob_new, log_prob_old, adv, mask)
        out.backward()
        assert log_prob_new.grad is not None
        assert torch.isfinite(log_prob_new.grad).all()

    def test_clip_range_as_tensor_accepted(self):
        loss = GRPOLoss()
        out = loss(
            torch.zeros(1, 1),
            torch.zeros(1, 1),
            torch.tensor([1.0]),
            torch.ones(1, 1),
            clip_range=torch.tensor(0.2),
        )
        # Loss should be a finite scalar.
        assert torch.isfinite(out)
