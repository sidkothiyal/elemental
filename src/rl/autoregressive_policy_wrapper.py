"""Autoregressive LLM/VLM policy wrapper (stub).

Placeholder for RLHF / GRPO runs where the policy is a token-sampling LM. The
``act`` method generates token sequences and returns summed per-token log-probs;
``evaluate_actions`` recomputes log-probs + entropy over the stored tokens under
the current policy.

Left as a skeleton — concrete implementation lands with the first RLHF experiment.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from rl.policy_wrapper import BasePolicyWrapper


class AutoregressivePolicyWrapper(BasePolicyWrapper):
    """Token-sampling RL wrapper for LLM/VLM RLHF-style training."""

    def __init__(
        self,
        model: nn.Module,
        value_head: nn.Module,
        *,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float | None = None,
    ) -> None:
        super().__init__(model=model)
        self.value_head = value_head
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    @torch.no_grad()
    def act(self, obs: dict[str, Any], *, deterministic: bool = False) -> dict[str, Any]:
        raise NotImplementedError(
            "AutoregressivePolicyWrapper.act is a stub; wire up when the first RLHF "
            "experiment lands. Should return {action=tokens, log_prob, value, action_info}."
        )

    def evaluate_actions(
        self,
        obs: dict[str, Any],
        action: torch.Tensor,
        action_info: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError(
            "AutoregressivePolicyWrapper.evaluate_actions is a stub; recompute per-token "
            "log-probs + entropy under the current policy when implementing."
        )
