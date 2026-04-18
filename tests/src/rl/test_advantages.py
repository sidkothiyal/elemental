"""Unit tests for GAE advantage computation."""
# pylint: disable=missing-class-docstring,missing-function-docstring


import pytest
import torch

from rl.advantages import GAEConfig, compute_gae


@pytest.mark.unit
def test_gae_single_step_matches_td_residual():
    """With gamma=lam=0 and a single step, GAE reduces to the TD residual r - V(s)."""
    rewards = torch.tensor([[1.0, 2.0]])  # [T=1, num_envs=2]
    values = torch.tensor([[0.5, 0.5]])
    dones = torch.tensor([[1.0, 1.0]])  # terminal -> no bootstrap
    last_values = torch.tensor([10.0, 10.0])  # ignored because done=1
    adv, ret = compute_gae(rewards, values, dones, last_values, GAEConfig(gamma=0.0, lam=0.0))
    # delta = r - V(s). (next_values * non_terminal = 0 for both reasons.)
    assert torch.allclose(adv, torch.tensor([[0.5, 1.5]]))
    assert torch.allclose(ret, adv + values)


@pytest.mark.unit
def test_gae_zero_reward_zero_value_is_zero():
    rewards = torch.zeros(4, 2)
    values = torch.zeros(4, 2)
    dones = torch.zeros(4, 2)
    last_values = torch.zeros(2)
    adv, ret = compute_gae(rewards, values, dones, last_values, GAEConfig())
    assert torch.allclose(adv, torch.zeros_like(adv))
    assert torch.allclose(ret, torch.zeros_like(ret))


@pytest.mark.unit
def test_gae_backward_recursion_matches_hand_computed_three_steps():
    """3-step trajectory; compute the GAE by hand and compare."""
    gamma, lam = 0.9, 0.8
    rewards = torch.tensor([[1.0], [0.0], [2.0]])
    values = torch.tensor([[0.5], [0.7], [0.3]])
    dones = torch.tensor([[0.0], [0.0], [0.0]])
    last_values = torch.tensor([0.4])

    # Expected backwards:
    # t=2: delta = 2.0 + 0.9 * 0.4 * 1 - 0.3 = 2.06;     gae_2 = 2.06
    # t=1: delta = 0.0 + 0.9 * 0.3 * 1 - 0.7 = -0.43;    gae_1 = -0.43 + 0.9*0.8*2.06 = 1.0532
    # t=0: delta = 1.0 + 0.9 * 0.7 * 1 - 0.5 = 1.13;     gae_0 = 1.13 + 0.9*0.8*1.0532 = 1.888304
    expected = torch.tensor([[1.888304], [1.0532], [2.06]])

    adv, _ = compute_gae(rewards, values, dones, last_values, GAEConfig(gamma=gamma, lam=lam))
    assert torch.allclose(adv, expected, atol=1e-4)


@pytest.mark.unit
def test_gae_terminal_mask_prevents_bootstrap():
    """If an episode ends at t=0, the future value should not leak into gae[0]."""
    rewards = torch.tensor([[1.0], [5.0]])
    values = torch.tensor([[0.0], [0.0]])
    dones = torch.tensor([[1.0], [0.0]])
    last_values = torch.tensor([0.0])
    adv, _ = compute_gae(rewards, values, dones, last_values, GAEConfig(gamma=0.99, lam=0.95))
    # gae[0] should only see reward[0] since done[0] masks out the t=1 value.
    assert adv[0].item() == pytest.approx(1.0)
