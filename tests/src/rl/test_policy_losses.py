"""Unit tests for PPO / value / entropy losses."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch

from loss.policy_losses import EntropyBonus, PPOClipLoss, ValueLoss


@pytest.mark.unit
def test_ppo_clip_loss_zero_when_ratio_one():
    """log_prob_new == log_prob_old ⇒ ratio=1 ⇒ loss = -advantage.mean()."""
    log_prob = torch.zeros(4)
    adv = torch.tensor([1.0, 2.0, 3.0, 4.0])
    loss = PPOClipLoss()(log_prob, log_prob, adv, clip_range=0.2)
    assert loss.item() == pytest.approx(-adv.mean().item(), abs=1e-5)


@pytest.mark.unit
def test_ppo_clip_loss_clamps_ratio_above_1_plus_eps():
    """With positive advantage and ratio > 1+eps, the clipped branch gets chosen."""
    clip = 0.2
    log_prob_old = torch.zeros(1)
    log_prob_new = torch.tensor([1.0])  # ratio = e >> 1.2
    adv = torch.tensor([1.0])
    loss = PPOClipLoss()(log_prob_new, log_prob_old, adv, clip_range=clip)
    assert loss.item() == pytest.approx(-(1.0 + clip), abs=1e-5)


@pytest.mark.unit
def test_ppo_clip_loss_clamps_ratio_below_1_minus_eps_for_negative_advantage():
    clip = 0.2
    log_prob_old = torch.zeros(1)
    log_prob_new = torch.tensor([-1.0])  # ratio = e^-1 << 1-0.2
    adv = torch.tensor([-1.0])
    loss = PPOClipLoss()(log_prob_new, log_prob_old, adv, clip_range=clip)
    # min(ratio * adv, clipped * adv) picks the smaller (more negative)
    # With adv<0 the clipped lower-bound term dominates.
    assert loss.item() == pytest.approx(-((1.0 - clip) * -1.0), abs=1e-5)


@pytest.mark.unit
def test_value_loss_without_clip():
    v_new = torch.tensor([1.0, 2.0, 3.0])
    returns = torch.tensor([1.5, 2.5, 3.5])
    out = ValueLoss()(v_new, returns, None, None)
    assert out.item() == pytest.approx(0.5 * ((v_new - returns) ** 2).mean().item())


@pytest.mark.unit
def test_value_loss_with_clip_uses_max_of_clipped_and_unclipped():
    v_new = torch.tensor([2.0])
    returns = torch.tensor([0.0])
    v_old = torch.tensor([0.0])
    # unclipped = 4.0; clipped to ±0.2 → (0.2)^2 = 0.04; max = 4.0
    out = ValueLoss()(v_new, returns, v_old, 0.2)
    assert out.item() == pytest.approx(0.5 * 4.0, abs=1e-5)


@pytest.mark.unit
def test_entropy_bonus_returns_mean():
    ent = torch.tensor([1.0, 2.0, 3.0])
    out = EntropyBonus()(ent, None)
    assert out.item() == pytest.approx(2.0)
