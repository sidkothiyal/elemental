"""Per-token clipped surrogate loss for GRPO with optional KL-to-reference."""

from __future__ import annotations

import torch
from torch import nn


class GRPOLoss(nn.Module):
    """GRPO clipped surrogate, masked over completion tokens, optional KL term.

    Expects (resolved by ``LossComponent`` from the trainer data container):

    - ``log_prob_new``     ``(B, T)`` per-token log-probs under the current policy
    - ``log_prob_old``     ``(B, T)`` per-token log-probs under the behavior policy
    - ``advantage``        ``(B,)``  per-prompt group-relative advantage
    - ``completion_mask``  ``(B, T)`` 1 on real completion tokens, 0 elsewhere
    - ``clip_range``       scalar epsilon
    - ``log_prob_ref``     ``(B, T)`` (optional) reference policy log-probs

    Loss = mean over masked tokens of:
        - min(ratio * advantage, clip(ratio, 1-e, 1+e) * advantage)
        + kl_coef * k3_kl(log_prob_new, log_prob_ref)

    where ``k3_kl(p, q) = exp(q - p) - (q - p) - 1`` is Schulman's positive,
    low-variance KL estimator (matches upstream nanochat / DeepSeek-Math GRPO).
    """

    def __init__(self, *, kl_coef: float = 0.0):
        super().__init__()
        self.kl_coef = kl_coef

    def forward(
        self,
        log_prob_new: torch.Tensor,
        log_prob_old: torch.Tensor,
        advantage: torch.Tensor,
        completion_mask: torch.Tensor,
        clip_range: float | torch.Tensor = 0.2,
        log_prob_ref: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the masked GRPO clipped surrogate loss (plus optional KL term)."""
        if isinstance(clip_range, torch.Tensor):
            clip_range = float(clip_range.item())

        adv = advantage.unsqueeze(-1).expand_as(log_prob_new)
        ratio = (log_prob_new - log_prob_old).exp()
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv
        per_token_pg = -torch.min(unclipped, clipped)

        loss_per_token = per_token_pg
        if self.kl_coef > 0 and log_prob_ref is not None:
            log_ratio = log_prob_ref - log_prob_new
            k3_kl = log_ratio.exp() - log_ratio - 1.0
            loss_per_token = loss_per_token + self.kl_coef * k3_kl

        denom = completion_mask.sum().clamp(min=1)
        return (loss_per_token * completion_mask).sum() / denom
