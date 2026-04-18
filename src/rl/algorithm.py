"""Abstract RL algorithm interface.

An ``BaseRLAlgorithm`` owns the two things that differ across RL methods:

1. ``collect_rollouts`` — how to drive the env and fill a buffer with transitions
2. ``update`` — how to consume that buffer to produce gradient updates

Everything else (epoch loop, callbacks, checkpointing, AMP) lives in ``RLTrainer``.
Swapping PPO for GRPO, A2C, or an off-policy method (SAC/DQN with a replay buffer)
is a config change + a new ``BaseRLAlgorithm`` subclass.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from env.base_vec_env import BaseVecEnv
    from rl.policy_wrapper import BasePolicyWrapper
    from rl.rollout_buffer import BaseRolloutBuffer


class BaseRLAlgorithm(ABC):
    """Strategy object that encapsulates RL-algorithm-specific logic."""

    @abstractmethod
    def collect_rollouts(
        self,
        *,
        env: "BaseVecEnv",
        policy_wrapper: "BasePolicyWrapper",
        buffer: "BaseRolloutBuffer",
        callbacks: Any,
        epoch: int,
        step_offset: int,
        device: torch.device,
    ) -> dict[str, float]:
        """Drive ``env.step`` and populate ``buffer``.

        Returns a dict of per-rollout episode metrics (mean return, success rate,
        episode length, ...) for logging.
        """

    @abstractmethod
    def update(
        self,
        *,
        buffer: "BaseRolloutBuffer",
        trainer: Any,  # RLTrainer — forward-declared to avoid circular import
    ) -> dict[str, float]:
        """Run policy updates over the collected rollouts. Returns averaged losses."""
