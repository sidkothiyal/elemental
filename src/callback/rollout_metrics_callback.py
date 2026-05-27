"""Rollout-metrics callback.

Aggregates per-step rewards / success signals seen during the ``rollout`` phase
and logs summary scalars to W&B. Kept separate from ``MetricsCallback`` because
it reads from the ``rl/*`` namespace instead of ``pred``/``target``.
"""

from __future__ import annotations

import logging

import torch

import wandb
from callback import callback

logger = logging.getLogger(__name__)


class RolloutMetricsCallback(callback.Callback):
    """Track per-step rewards / successes during rollout phase."""

    def __init__(
        self,
        *,
        reward_key: str = "rl/reward",
        done_key: str = "rl/done",
        success_key: str | None = None,
    ) -> None:
        super().__init__()
        self.reward_key = reward_key
        self.done_key = done_key
        self.success_key = success_key
        self._reward_sum: float = 0.0
        self._num_steps: int = 0

    def on_phase_start(self, phase: str) -> None:
        if phase == "rollout":
            self._reward_sum = 0.0
            self._num_steps = 0

    def on_batch_end(self, step: int, phase: str, data: dict) -> None:
        if phase != "rollout":
            return
        reward = data.get(self.reward_key)
        if isinstance(reward, torch.Tensor):
            self._reward_sum += float(reward.mean().item())
            self._num_steps += 1

    def on_phase_end(self, phase: str) -> None:
        if phase != "rollout" or self._num_steps == 0:
            return

        if wandb.run is None:
            return
        wandb.log({"rollout/mean_step_reward": self._reward_sum / self._num_steps})
