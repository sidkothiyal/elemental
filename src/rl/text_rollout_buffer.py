"""On-policy rollout buffer for text/GRPO training.

Differs from ``OnPolicyRolloutBuffer`` in that completions are variable-length
sequences (padded to ``T_max``) rather than fixed-shape actions, advantages are
per-prompt scalars (group-relative, populated by the algorithm), and there is
no value head — so no GAE / returns are computed here.
"""

from __future__ import annotations

from typing import Any, Iterator

import torch


class TextRolloutBuffer:
    """Flat buffer over completion samples (B*G of them per rollout).

    Stores everything in pre-allocated ``(capacity, T_max)`` (or ``(capacity,)``)
    tensors. ``add()`` overwrites the buffer with a fresh rollout — no
    accumulation across rollouts.
    """

    def __init__(
        self,
        *,
        capacity: int,
        prompt_max_length: int,
        completion_max_length: int,
        device: str | torch.device = "cpu",
    ) -> None:
        self.capacity = capacity
        self.prompt_max_length = prompt_max_length
        self.completion_max_length = completion_max_length
        self.device = torch.device(device)
        self.reset()

    def reset(self) -> None:
        """(Re)allocate buffer tensors and clear the size counter."""
        cap = self.capacity
        p_len = self.prompt_max_length
        c_len = self.completion_max_length
        dev = self.device
        self.prompt_ids = torch.zeros(cap, p_len, dtype=torch.long, device=dev)
        self.actions = torch.zeros(cap, c_len, dtype=torch.long, device=dev)
        self.log_prob_old = torch.zeros(cap, c_len, device=dev)
        self.log_prob_ref = torch.zeros(cap, c_len, device=dev)
        self.completion_mask = torch.zeros(cap, c_len, device=dev)
        self.advantages = torch.zeros(cap, device=dev)
        self.rewards = torch.zeros(cap, device=dev)
        self.size = 0
        self._has_ref = False

    def add(
        self,
        *,
        prompt_ids: torch.Tensor,
        actions: torch.Tensor,
        log_prob_old: torch.Tensor,
        completion_mask: torch.Tensor,
        advantages: torch.Tensor,
        rewards: torch.Tensor,
        log_prob_ref: torch.Tensor | None = None,
    ) -> None:
        """Overwrite the buffer with a single rollout's samples (left-pads prompts)."""
        n = actions.size(0)
        if n > self.capacity:
            raise ValueError(f"rollout has {n} samples, buffer capacity is {self.capacity}")
        # Right-pad / truncate prompts to a fixed prompt_max_length so minibatches
        # are uniform-shaped.
        p = prompt_ids.size(1)
        if p < self.prompt_max_length:
            pad = torch.zeros(
                n, self.prompt_max_length - p, dtype=prompt_ids.dtype, device=prompt_ids.device
            )
            prompt_ids = torch.cat([pad, prompt_ids], dim=1)  # left-pad
        elif p > self.prompt_max_length:
            prompt_ids = prompt_ids[:, -self.prompt_max_length :]

        self.prompt_ids[:n] = prompt_ids.to(self.device)
        self.actions[:n] = actions.to(self.device)
        self.log_prob_old[:n] = log_prob_old.to(self.device)
        self.completion_mask[:n] = completion_mask.to(self.device)
        self.advantages[:n] = advantages.to(self.device)
        self.rewards[:n] = rewards.to(self.device)
        if log_prob_ref is not None:
            self.log_prob_ref[:n] = log_prob_ref.to(self.device)
            self._has_ref = True
        self.size = n

    def finalize(self) -> None:
        """Hook called after ``add``; no-op for GRPO since advantages are already set."""
        # No-op for GRPO: advantages were populated by ``add``.

    def iter_minibatches(
        self,
        minibatch_size: int,
        shuffle: bool = True,
    ) -> Iterator[dict[str, Any]]:
        """Yield trainer-shaped minibatch dicts over the current buffer contents."""
        if self.size == 0:
            return
        indices = torch.randperm(self.size) if shuffle else torch.arange(self.size)
        for start in range(0, self.size, minibatch_size):
            idx = indices[start : start + minibatch_size]
            batch: dict[str, Any] = {
                "input/input_ids": self.prompt_ids[idx],
                "rl/action": self.actions[idx],
                "rl/log_prob_old": self.log_prob_old[idx],
                "rl/advantage": self.advantages[idx],
                "rl/reward": self.rewards[idx],
                "action_info/completion_mask": self.completion_mask[idx],
            }
            if self._has_ref:
                batch["rl/log_prob_ref"] = self.log_prob_ref[idx]
            yield batch
