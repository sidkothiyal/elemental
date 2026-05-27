"""Abstract base class for policy wrappers.

A policy wrapper adapts an elemental ``nn.Module`` (VLA diffusion, autoregressive
LM, ...) to the act / evaluate_actions API that RL algorithms expect. It owns
any auxiliary heads needed for RL (e.g. a value head) so the base model file
stays unchanged.
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn


class BasePolicyWrapper(nn.Module, ABC):
    """Wraps a base model to expose an RL-friendly policy interface.

    Subclasses implement:

    - ``act(obs)``: sample an action given an observation; return log-prob, value,
      and any per-action extras (e.g. diffusion noise trajectory) needed to later
      recompute log-probs under an updated policy.
    - ``evaluate_actions(obs, action, action_info)``: recompute log-prob, entropy,
      and value for stored transitions. Called during the PPO update.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @abstractmethod
    @torch.no_grad()
    def act(
        self,
        obs: dict[str, Any],
        *,
        deterministic: bool = False,
    ) -> dict[str, Any]:
        """Sample an action.

        Returns a dict with at minimum:

            ``action``      — the executed action tensor
            ``log_prob``    — log-prob of ``action`` under the current policy (``[num_envs]``)
            ``value``       — value-head estimate of the observation (``[num_envs]``)
            ``action_info`` — dict of extras forwarded to ``evaluate_actions``
        """

    @abstractmethod
    def evaluate_actions(
        self,
        obs: dict[str, Any],
        action: torch.Tensor,
        action_info: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Recompute RL quantities for stored transitions.

        Must return a dict with keys ``log_prob``, ``entropy``, ``value`` — each
        a ``[batch]`` tensor — with gradients attached to parameters that should
        be updated by the RL loss.
        """
