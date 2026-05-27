"""Token sampling utilities shared by autoregressive models and RL rollouts."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a single next token from per-row logits.

    Args:
        logits: shape ``(B, V)`` — last-step logits per batch row.
        temperature: divides logits before softmax. ``0`` selects argmax (greedy).
        top_k: keep only the top-k logits; the rest are masked to ``-inf``.
        top_p: nucleus sampling — keep the smallest set whose cumulative probability
            crosses ``top_p``. Disabled when ``None`` or ``>= 1``.
        generator: optional ``torch.Generator`` for reproducible sampling.

    Returns:
        ``(token, log_prob)`` both shape ``(B,)``. ``log_prob`` is the log-probability
        of the sampled token under the (filtered) categorical distribution actually
        sampled from — required for RL importance ratios.
    """
    if logits.dim() != 2:
        raise ValueError(f"expected logits of shape (B, V), got {tuple(logits.shape)}")

    if temperature == 0:
        token = logits.argmax(dim=-1)
        log_prob = F.log_softmax(logits, dim=-1).gather(-1, token.unsqueeze(-1)).squeeze(-1)
        return token, log_prob

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        topk_vals, _ = torch.topk(logits, k, dim=-1)
        threshold = topk_vals[..., -1:]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cumulative = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        # Drop tokens whose cumulative probability strictly exceeds top_p, but
        # always keep the top-1 — shift the mask right by one and clear position 0.
        remove_sorted = cumulative > top_p
        remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
        remove_sorted[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove_sorted, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)

    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    token = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
    log_prob = log_probs.gather(-1, token.unsqueeze(-1)).squeeze(-1)
    return token, log_prob
