"""Unit tests for OnPolicyRolloutBuffer."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch

from rl.advantages import GAEConfig
from rl.rollout_buffer import OnPolicyRolloutBuffer


@pytest.mark.unit
def test_buffer_add_and_flatten_roundtrip():
    num_steps, num_envs = 4, 2
    buf = OnPolicyRolloutBuffer(
        num_steps=num_steps,
        num_envs=num_envs,
        action_shape=(5,),
        obs_keys=("image",),
    )
    for _ in range(num_steps):
        buf.add(
            obs={"image": torch.randn(num_envs, 3, 8, 8)},
            action=torch.randn(num_envs, 5),
            log_prob=torch.randn(num_envs),
            value=torch.randn(num_envs),
            reward=torch.randn(num_envs),
            done=torch.zeros(num_envs),
        )
    buf.finalize(last_values=torch.zeros(num_envs), gae=GAEConfig())

    batches = list(buf.iter_minibatches(minibatch_size=3, shuffle=False))
    total_samples = sum(b["rl/action"].shape[0] for b in batches)
    assert total_samples == num_steps * num_envs

    first = batches[0]
    assert first["rl/action"].shape[-1] == 5
    assert first["input/image"].shape[1:] == (3, 8, 8)
    assert "rl/advantage" in first and "rl/return" in first


@pytest.mark.unit
def test_buffer_raises_before_finalize():
    buf = OnPolicyRolloutBuffer(num_steps=1, num_envs=1, action_shape=(2,), obs_keys=("image",))
    buf.add(
        obs={"image": torch.randn(1, 1, 2, 2)},
        action=torch.randn(1, 2),
        log_prob=torch.randn(1),
        value=torch.randn(1),
        reward=torch.randn(1),
        done=torch.zeros(1),
    )
    with pytest.raises(RuntimeError):
        list(buf.iter_minibatches(1))


@pytest.mark.unit
def test_buffer_overflow_raises():
    buf = OnPolicyRolloutBuffer(num_steps=1, num_envs=1, action_shape=(2,), obs_keys=("image",))
    for _ in range(1):
        buf.add(
            obs={"image": torch.randn(1, 1, 2, 2)},
            action=torch.randn(1, 2),
            log_prob=torch.randn(1),
            value=torch.randn(1),
            reward=torch.randn(1),
            done=torch.zeros(1),
        )
    with pytest.raises(IndexError):
        buf.add(
            obs={"image": torch.randn(1, 1, 2, 2)},
            action=torch.randn(1, 2),
            log_prob=torch.randn(1),
            value=torch.randn(1),
            reward=torch.randn(1),
            done=torch.zeros(1),
        )
