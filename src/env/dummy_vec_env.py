"""Deterministic dummy vectorized environment for unit tests and smoke runs."""

from typing import Any

import torch

from env.base_vec_env import BaseVecEnv


class DummyVecEnv(BaseVecEnv):
    """Random-observation / dense-reward env that exercises the full RL pipeline.

    Rewards are ``-||action||^2`` plus noise, and episodes terminate after
    ``max_episode_steps``. This is enough to verify rollout collection, advantage
    computation, and the PPO update loop end-to-end without any simulator deps.
    """

    def __init__(
        self,
        num_envs: int = 4,
        action_dim: int = 7,
        num_action_chunks: int = 20,
        image_shape: tuple[int, int, int] = (3, 224, 224),
        text_prompt: str = "Predict the next robot action.",
        max_episode_steps: int = 20,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ) -> None:
        self.num_envs = num_envs
        self.action_dim = action_dim
        self.num_action_chunks = num_action_chunks
        self.image_shape = image_shape
        self.text_prompt = text_prompt
        self.max_episode_steps = max_episode_steps
        self.device = torch.device(device)
        self._rng = torch.Generator(device="cpu").manual_seed(seed)
        self._step_count = torch.zeros(num_envs, dtype=torch.long)

        self.observation_spec = {
            "image": {"shape": image_shape, "dtype": torch.float32},
            "text": {"dtype": "str"},
        }
        self.action_spec = {
            "shape": (num_action_chunks, action_dim),
            "dtype": torch.float32,
        }

    def _sample_obs(self) -> dict[str, Any]:
        image = torch.rand(
            (self.num_envs, *self.image_shape), generator=self._rng, device="cpu"
        ).to(self.device)
        return {"image": image, "text": [self.text_prompt] * self.num_envs}

    def reset(self) -> dict[str, Any]:
        self._step_count.zero_()
        return self._sample_obs()

    def step(
        self, actions: torch.Tensor
    ) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, dict[str, Any]]:
        # actions: [num_envs, action_dim] — the first step of the emitted chunk.
        squared = (actions.to("cpu") ** 2).sum(dim=-1)
        reward = (-squared + 0.05 * torch.randn(self.num_envs, generator=self._rng)).to(self.device)
        self._step_count += 1
        done = (self._step_count >= self.max_episode_steps).to(self.device)
        info: dict[str, Any] = {
            "success": (reward > 0).to(self.device),
            "episode_step": self._step_count.clone().to(self.device),
        }
        if done.any():
            mask = done.to("cpu")
            self._step_count[mask] = 0
        return self._sample_obs(), reward, done, info

    def render(self) -> torch.Tensor:
        return torch.rand((self.num_envs, *self.image_shape), generator=self._rng, device="cpu")

    def close(self) -> None:
        return None
