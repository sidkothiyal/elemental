"""Stateful episode-level RL metrics.

Track success rate, mean return, and mean episode length across rollouts. Lives
alongside elemental's existing TorchMetrics-based metrics (``src/metrics/``) so
``MetricsCallback`` can consume them through the same interface.
"""

from __future__ import annotations

import torch
from torchmetrics import Metric


class EpisodeReturnTracker(Metric):
    """Running mean of episode returns.

    Expects per-step calls to ``update(reward, done)`` where ``reward`` is
    ``[num_envs]`` and ``done`` is a boolean mask of episode terminations.
    """

    full_state_update = False

    total_return: torch.Tensor
    num_episodes: torch.Tensor
    running_return: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.add_state("total_return", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_episodes", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("running_return", default=torch.tensor([]), dist_reduce_fx="cat")

    def update(self, reward: torch.Tensor, done: torch.Tensor) -> None:  # type: ignore[override]
        """Accumulate per-step rewards and finalize return when an episode ends."""
        if self.running_return.numel() == 0:
            self.running_return = torch.zeros_like(reward)
        self.running_return = self.running_return + reward
        done_bool = done.bool()
        if done_bool.any():
            self.total_return = self.total_return + self.running_return[done_bool].sum()
            self.num_episodes = self.num_episodes + done_bool.sum().float()
            self.running_return = self.running_return * (~done_bool).float()

    def compute(self) -> torch.Tensor:
        """Return mean episode return, or 0 if no episodes have completed."""
        if self.num_episodes == 0:
            return torch.tensor(0.0)
        return self.total_return / self.num_episodes


class SuccessRateTracker(Metric):
    """Running success rate: fraction of terminated episodes whose info['success'] was True."""

    full_state_update = False

    num_successes: torch.Tensor
    num_episodes: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.add_state("num_successes", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_episodes", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, done: torch.Tensor, success: torch.Tensor) -> None:  # type: ignore[override]
        """Count successes and terminations from the current env step."""
        done_bool = done.bool()
        success_bool = success.bool()
        self.num_successes = self.num_successes + (done_bool & success_bool).sum().float()
        self.num_episodes = self.num_episodes + done_bool.sum().float()

    def compute(self) -> torch.Tensor:
        """Return fraction of completed episodes that succeeded, or 0 if none."""
        if self.num_episodes == 0:
            return torch.tensor(0.0)
        return self.num_successes / self.num_episodes


class EpisodeLengthTracker(Metric):
    """Mean length (in env steps) of completed episodes."""

    full_state_update = False

    total_steps: torch.Tensor
    num_episodes: torch.Tensor
    running_length: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.add_state("total_steps", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_episodes", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("running_length", default=torch.tensor([]), dist_reduce_fx="cat")

    def update(self, done: torch.Tensor) -> None:  # type: ignore[override]
        """Increment step counters and record length when an episode ends."""
        done_bool = done.bool()
        if self.running_length.numel() == 0:
            self.running_length = torch.zeros_like(done_bool, dtype=torch.float32)
        self.running_length = self.running_length + 1.0
        if done_bool.any():
            self.total_steps = self.total_steps + self.running_length[done_bool].sum()
            self.num_episodes = self.num_episodes + done_bool.sum().float()
            self.running_length = self.running_length * (~done_bool).float()

    def compute(self) -> torch.Tensor:
        """Return mean episode length in env steps, or 0 if no episodes have completed."""
        if self.num_episodes == 0:
            return torch.tensor(0.0)
        return self.total_steps / self.num_episodes
