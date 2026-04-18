"""Abstract vectorized-environment interface for RL training."""

from abc import ABC, abstractmethod
from typing import Any

import torch


class BaseVecEnv(ABC):
    """Vectorized environment returning observation dicts in elemental's container shape.

    Subclasses wrap concrete simulators (SIMPLER, Isaac Lab, gym VecEnv, text-generation
    envs for RLHF). Observation dicts use the same keys the model's conditioner expects
    (e.g. ``image``, ``text``) so the policy wrapper can forward them straight into the
    model without adapter code.
    """

    num_envs: int
    action_dim: int
    observation_spec: dict[str, Any]
    action_spec: dict[str, Any]

    @abstractmethod
    def reset(self) -> dict[str, Any]:
        """Reset all sub-environments. Returns the initial observation dict."""

    @abstractmethod
    def step(
        self, actions: torch.Tensor
    ) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Step all sub-environments with the given actions.

        Returns:
            ``(next_obs, reward, done, info)`` where ``reward`` and ``done`` are
            tensors of shape ``[num_envs]`` and ``info`` is a dict that may hold
            per-env fields such as ``"success"`` or ``"frames"``.
        """

    def render(self) -> torch.Tensor | None:
        """Return a ``[num_envs, C, H, W]`` frame tensor for video logging, or None."""
        return None

    def close(self) -> None:
        """Release simulator resources. Default: no-op."""
