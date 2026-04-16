"""RL trainer skeleton.

Stubs only — bodies will be filled in when the SIMPLER env and rollout buffer
land (see ``whimsical-herding-aurora.md``). The shape of this class proves the
extension point: ``train_batch`` orchestrates ``_collect_rollouts``,
``_compute_advantages``, and ``_update_policy``, each of which currently
raises ``NotImplementedError``.
"""

import torch

from training.trainer import Trainer


class RLTrainer(Trainer):
    """Reinforcement-learning trainer skeleton."""

    def _collect_rollouts(self, batch: dict):  # pylint: disable=unused-argument
        raise NotImplementedError(
            "RLTrainer._collect_rollouts: SIMPLER env integration not wired yet."
        )

    def _compute_advantages(self, rollouts):  # pylint: disable=unused-argument
        raise NotImplementedError("RLTrainer._compute_advantages: not implemented yet.")

    def _update_policy(
        self,
        rollouts,  # pylint: disable=unused-argument
        advantages,  # pylint: disable=unused-argument
        *,
        training: bool = True,  # pylint: disable=unused-argument
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError("RLTrainer._update_policy: not implemented yet.")

    def train_batch(  # pylint: disable=arguments-differ
        self,
        batch: dict,
        batch_idx: int,  # pylint: disable=unused-argument
        epoch: int,  # pylint: disable=unused-argument
        step: int,  # pylint: disable=unused-argument
        data: dict,  # pylint: disable=unused-argument
        *,
        training: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Orchestrate the RL update: rollout → advantages → policy update."""
        rollouts = self._collect_rollouts(batch)
        advantages = self._compute_advantages(rollouts)
        return self._update_policy(rollouts, advantages, training=training)
