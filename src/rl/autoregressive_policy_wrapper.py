"""Autoregressive LLM policy wrapper for GRPO/RLHF training.

Implements the ``BasePolicyWrapper`` API for token-sampling policies:

- ``act``: vectorized batched generation. Returns the sampled tokens, the
  per-token log-probabilities under the (filtered) sampling distribution,
  a zero-valued "value" placeholder (GRPO has no value head), and a
  ``completion_mask`` that's 1 from the first generated token through the
  first ``<|eot|>`` (inclusive) and 0 thereafter.
- ``evaluate_actions``: a single forward pass over ``concat(prompt, completion)``
  to recompute per-token log-probabilities and entropy under the current policy.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from rl.policy_wrapper import BasePolicyWrapper
from utils.sampling import sample_next_token


class AutoregressivePolicyWrapper(BasePolicyWrapper):
    """Token-sampling policy wrapper for LLM RLHF/GRPO.

    Args:
        model: an ``nn.Module`` whose ``forward(input_ids, training=False)``
            returns logits of shape ``(B, T, V)``.
        max_new_tokens: max generation length per prompt.
        temperature: sampling temperature. ``0`` selects greedy.
        top_p: nucleus sampling cutoff (None disables).
        eos_token_id: stops generation early; locked-in for finished rows so the
            ``completion_mask`` is well-defined.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float | None = None,
        eos_token_id: int | None = None,
    ) -> None:
        super().__init__(model=model)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.eos_token_id = eos_token_id

    def _model_max_length(self) -> int | None:
        # Look through DDP / torch.compile wrappers if present.
        m = self.model
        for _ in range(3):
            inner = getattr(m, "module", None) or getattr(m, "_orig_mod", None)
            if inner is None:
                break
            m = inner
        return getattr(m, "max_length", None)

    @torch.no_grad()
    def act(self, obs: dict[str, Any], *, deterministic: bool = False) -> dict[str, Any]:
        input_ids: torch.Tensor = obs["input_ids"]
        device = input_ids.device
        batch_size = input_ids.size(0)
        prompt_len = input_ids.size(1)

        max_seq = self._model_max_length() or (prompt_len + self.max_new_tokens)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        tokens_collected: list[torch.Tensor] = []
        log_probs_collected: list[torch.Tensor] = []
        ctx = input_ids
        eos = self.eos_token_id

        temp = 0.0 if deterministic else self.temperature

        for _ in range(self.max_new_tokens):
            ctx_window = ctx[:, -max_seq:]
            logits = self.model(ctx_window, training=False)
            next_token, log_prob = sample_next_token(
                logits[:, -1, :], temperature=temp, top_p=self.top_p
            )
            if eos is not None:
                # Finished rows emit EOS forever; their log-prob contribution is zero.
                next_token = torch.where(finished, torch.full_like(next_token, eos), next_token)
                log_prob = torch.where(finished, torch.zeros_like(log_prob), log_prob)
                finished = finished | (next_token == eos)
            tokens_collected.append(next_token)
            log_probs_collected.append(log_prob)
            ctx = torch.cat([ctx, next_token.unsqueeze(-1)], dim=1)
            if eos is not None and bool(finished.all()):
                # Pad remaining timesteps with EOS / zero log-prob so the output
                # tensor has a fixed (max_new_tokens) time dim.
                pad_token = torch.full_like(next_token, eos)
                pad_lp = torch.zeros_like(log_prob)
                while len(tokens_collected) < self.max_new_tokens:
                    tokens_collected.append(pad_token.clone())
                    log_probs_collected.append(pad_lp.clone())
                break

        actions = torch.stack(tokens_collected, dim=1)  # (B, T)
        log_prob_per_token = torch.stack(log_probs_collected, dim=1)  # (B, T)

        if eos is not None:
            # 1 from start through first EOS (inclusive), 0 after.
            is_eos = actions == eos
            cum_eos = is_eos.long().cumsum(dim=1)
            completion_mask = (cum_eos - is_eos.long() == 0).float()
        else:
            completion_mask = torch.ones_like(actions, dtype=torch.float32)

        return {
            "action": actions,
            "log_prob": log_prob_per_token,
            "value": torch.zeros(batch_size, device=device),
            "action_info": {
                "completion_mask": completion_mask,
                "prompt_len": torch.full(
                    (batch_size,), prompt_len, dtype=torch.long, device=device
                ),
            },
        }

    def evaluate_actions(
        self,
        obs: dict[str, Any],
        action: torch.Tensor,
        action_info: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Recompute per-token log-prob + entropy over the stored ``action`` tokens.

        Returns ``log_prob`` and ``entropy`` of shape ``(B, T)`` (not scalar) and
        a zero ``value`` placeholder. The GRPO loss masks by ``completion_mask``;
        we keep the per-token shape so the loss can do that masking itself.
        """
        del action_info  # mask is read by the loss directly from the data container
        prompt_ids = obs["input_ids"]
        # Concatenate prompt + completion. We feed all but the last token to the
        # model so logits at position i predict position i+1.
        full = torch.cat([prompt_ids, action], dim=1)
        max_seq = self._model_max_length() or full.size(1)
        if full.size(1) > max_seq:
            # Truncate from the left; keep the last `max_seq` tokens. This means
            # we may drop early prompt tokens for very long contexts.
            full = full[:, -max_seq:]
        logits = self.model(full[:, :-1], training=True)  # (B, P + T - 1, V)
        log_probs_all = F.log_softmax(logits, dim=-1)

        prompt_len = prompt_ids.size(1)
        completion_logits = log_probs_all[:, prompt_len - 1 :, :]  # (B, T, V)
        # Truncate action to match completion_logits time dim if we truncated above.
        action_aligned = action[:, : completion_logits.size(1)]
        log_prob_per_token = completion_logits.gather(-1, action_aligned.unsqueeze(-1)).squeeze(-1)
        entropy_per_token = -(completion_logits.exp() * completion_logits).sum(dim=-1)

        return {
            "log_prob": log_prob_per_token,
            "entropy": entropy_per_token,
            "value": torch.zeros(action.size(0), device=action.device),
        }
