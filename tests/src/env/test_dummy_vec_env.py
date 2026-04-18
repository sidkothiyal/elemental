"""Smoke tests for DummyVecEnv."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch

from env.dummy_vec_env import DummyVecEnv


@pytest.mark.unit
def test_dummy_env_reset_shape():
    env = DummyVecEnv(num_envs=3, action_dim=4, num_action_chunks=5, image_shape=(3, 16, 16))
    obs = env.reset()
    assert obs["image"].shape == (3, 3, 16, 16)
    assert len(obs["text"]) == 3


@pytest.mark.unit
def test_dummy_env_step_shapes():
    env = DummyVecEnv(num_envs=2, action_dim=4, num_action_chunks=5, image_shape=(3, 8, 8))
    env.reset()
    action = torch.zeros(2, 4)
    next_obs, reward, done, info = env.step(action)
    assert next_obs["image"].shape == (2, 3, 8, 8)
    assert reward.shape == (2,)
    assert done.shape == (2,)
    assert info["success"].shape == (2,)


@pytest.mark.unit
def test_dummy_env_terminates_after_max_steps():
    env = DummyVecEnv(num_envs=1, max_episode_steps=3, image_shape=(3, 4, 4))
    env.reset()
    dones = []
    for _ in range(5):
        _, _, done, _ = env.step(torch.zeros(1, env.action_dim))
        dones.append(bool(done.item()))
    assert any(dones), "expected at least one episode termination"
