"""Unit tests for GRPO group-relative advantage computation."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

import pytest
import torch


@pytest.mark.unit
class TestGroupRelativeAdvantage:
    """Validate the GRPO advantage formula directly.

    The full GRPOAlgorithm needs a tokenizer and a verifier, so we test the
    advantage normalization in isolation: it should yield zero-mean within each
    group and unit standard deviation when ``normalize_advantages=True``.
    """

    def test_zero_mean_within_group(self):
        rewards = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 1.0]).view(2, 3)
        means = rewards.mean(dim=1, keepdim=True)
        adv = rewards - means
        # Each group's advantages must sum to zero.
        torch.testing.assert_close(adv.sum(dim=1), torch.zeros(2), atol=1e-6, rtol=1e-6)

    def test_unit_std_after_normalization(self):
        rewards = torch.tensor([0.0, 1.0, 0.0, 1.0]).view(1, 4)
        means = rewards.mean(dim=1, keepdim=True)
        stds = rewards.std(dim=1, keepdim=True).clamp(min=1e-8)
        adv = (rewards - means) / stds
        # std of a tensor with 4 entries equally split between 0 and 1 is 0.5774
        # (sample std with ddof=1). The normalized advantages should have unit
        # sample std.
        assert adv.std().item() == pytest.approx(1.0, abs=1e-5)

    def test_zero_advantage_when_all_rewards_equal(self):
        # When every completion in a group earns the same reward, advantages
        # collapse to zero — the group contributes no learning signal.
        rewards = torch.tensor([0.5] * 4).view(1, 4)
        means = rewards.mean(dim=1, keepdim=True)
        stds = rewards.std(dim=1, keepdim=True).clamp(min=1e-8)
        adv = (rewards - means) / stds
        torch.testing.assert_close(adv, torch.zeros_like(adv), atol=1e-6, rtol=1e-6)
