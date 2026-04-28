"""Group Relative Policy Optimization algorithm.

For each prompt, sample G completions under the current policy. Score each
completion with a verifier. Compute group-relative advantages — for each group::

    advantage_i = (reward_i - mean(group)) / (std(group) + eps)

Push (prompt, action_tokens, log_prob_old, completion_mask, advantage) into the
``TextRolloutBuffer``. The update loop calls ``trainer.train_batch`` over
minibatches just like PPO; the GRPO loss does the per-token clipping + masking.
"""

from __future__ import annotations

import logging
import math
import statistics
from typing import TYPE_CHECKING, Any

import torch

from data_processor.bpe_tokenizer import (
    BPETokenizer,
)
from rl.algorithm import BaseRLAlgorithm

if TYPE_CHECKING:  # pragma: no cover
    from env.prompt_iterator_env import PromptIteratorEnv
    from rl.policy_wrapper import BasePolicyWrapper
    from rl.text_rollout_buffer import TextRolloutBuffer
    from rl.verifiers.base import Verifier

logger = logging.getLogger(__name__)


class GRPOAlgorithm(BaseRLAlgorithm):
    """GRPO: sample G completions per prompt, score, compute group-relative advantages."""

    def __init__(
        self,
        *,
        verifier: "Verifier",
        tokenizer_path: str,
        group_size: int = 8,
        prompts_per_batch: int = 8,
        num_policy_epochs: int = 1,
        minibatch_size: int = 16,
        clip_range: float = 0.2,
        normalize_advantages: bool = True,
        kl_coef: float = 0.0,
    ) -> None:
        self.verifier = verifier
        self.tokenizer_path = tokenizer_path
        self.group_size = group_size
        self.prompts_per_batch = prompts_per_batch
        self.num_policy_epochs = num_policy_epochs
        self.minibatch_size = minibatch_size
        self.clip_range = clip_range
        self.normalize_advantages = normalize_advantages
        self.kl_coef = kl_coef
        self._tokenizer = None  # lazy

    def _get_tokenizer(self):
        """Lazily construct the BPE tokenizer (avoids loading on dataloader workers)."""
        if self._tokenizer is None:
            self._tokenizer = BPETokenizer(self.tokenizer_path)
        return self._tokenizer

    @property
    def steps_per_rollout(self) -> int:
        """Number of trainer steps per rollout (one step = one prompt batch)."""
        # Used by RLTrainer for callback step indexing; one "step" = one prompt batch.
        return self.prompts_per_batch

    def collect_rollouts(
        self,
        *,
        env: "PromptIteratorEnv",
        policy_wrapper: "BasePolicyWrapper",
        buffer: "TextRolloutBuffer",
        callbacks: Any,
        epoch: int,
        step_offset: int,
        device: torch.device,
    ) -> dict[str, float]:
        del epoch  # rollout step is the salient metric
        buffer.reset()
        policy_wrapper.eval()

        obs = env.reset()
        prompt_ids = obs["input_ids"].to(device)
        # Repeat each prompt G times for group sampling.
        repeated_prompt_ids = prompt_ids.repeat_interleave(self.group_size, dim=0)
        repeated_prompt_text = [t for t in obs["prompt_text"] for _ in range(self.group_size)]
        repeated_answer_text = [t for t in obs["answer_text"] for _ in range(self.group_size)]

        callbacks.on_batch_start(step_offset, "rollout")
        out = policy_wrapper.act({"input_ids": repeated_prompt_ids}, deterministic=False)
        actions = out["action"]  # (B*G, T)
        log_prob_old = out["log_prob"]  # (B*G, T)
        completion_mask = out["action_info"]["completion_mask"]  # (B*G, T)

        tokenizer = self._get_tokenizer()
        completions: list[str] = []
        for i in range(actions.size(0)):
            mask = completion_mask[i].bool()
            tokens = actions[i][mask].tolist()
            completions.append(tokenizer.decode(tokens, skip_special_tokens=True))

        rewards = torch.tensor(
            [
                self.verifier.score(p, c, a)
                for p, c, a in zip(repeated_prompt_text, completions, repeated_answer_text)
            ],
            dtype=torch.float32,
            device=device,
        )

        rewards_grouped = rewards.view(-1, self.group_size)
        means = rewards_grouped.mean(dim=1, keepdim=True)
        if self.normalize_advantages:
            stds = rewards_grouped.std(dim=1, keepdim=True).clamp(min=1e-8)
            adv_grouped = (rewards_grouped - means) / stds
        else:
            adv_grouped = rewards_grouped - means
        advantages = adv_grouped.view(-1)

        log_prob_ref = None
        ref_model = getattr(policy_wrapper, "reference_model", None)
        if self.kl_coef > 0 and ref_model is not None:
            with torch.no_grad():
                full = torch.cat([repeated_prompt_ids, actions], dim=1)
                logits = ref_model(full[:, :-1], training=False)
                log_probs_all = torch.nn.functional.log_softmax(logits, dim=-1)
                p_len = repeated_prompt_ids.size(1)
                completion_log_probs = log_probs_all[:, p_len - 1 :, :]
                action_aligned = actions[:, : completion_log_probs.size(1)]
                log_prob_ref = completion_log_probs.gather(
                    -1, action_aligned.unsqueeze(-1)
                ).squeeze(-1)

        buffer.add(
            prompt_ids=repeated_prompt_ids,
            actions=actions,
            log_prob_old=log_prob_old,
            completion_mask=completion_mask,
            advantages=advantages,
            rewards=rewards,
            log_prob_ref=log_prob_ref,
        )
        buffer.finalize()

        callbacks.on_batch_end(
            step_offset,
            "rollout",
            data={
                "rl/reward": rewards,
                "rl/advantage": advantages,
            },
        )

        return {
            "rollout/reward_mean": float(rewards.mean()),
            "rollout/reward_max": float(rewards.max()),
            "rollout/correct_fraction": float((rewards > 0).float().mean()),
            "rollout/completion_length_mean": float(completion_mask.sum(dim=1).mean()),
            "rollout/num_samples": float(rewards.numel()),
        }

    def update(
        self,
        *,
        buffer: "TextRolloutBuffer",
        trainer: Any,
    ) -> dict[str, float]:
        policy_wrapper = trainer.policy_wrapper
        policy_wrapper.train()

        running: dict[str, float] = {}
        total_minibatches_per_epoch = max(1, math.ceil(buffer.size / self.minibatch_size))
        trainer._total_batches = total_minibatches_per_epoch * self.num_policy_epochs
        trainer._current_loader = None  # type: ignore[attr-defined]

        global_batch_idx = 0
        approx_kls: list[float] = []
        for _policy_epoch in range(self.num_policy_epochs):
            for minibatch in buffer.iter_minibatches(self.minibatch_size, shuffle=True):
                minibatch["rl/clip_range"] = torch.tensor(
                    self.clip_range, device=buffer.device, dtype=torch.float32
                )

                data: dict[str, Any] = {}
                losses = trainer.train_batch(
                    minibatch,
                    global_batch_idx,
                    epoch=trainer._rl_epoch,
                    step=global_batch_idx,
                    data=data,
                    training=True,
                )
                for key, value in losses.items():
                    running[key] = running.get(key, 0.0) + float(value.item())

                with torch.no_grad():
                    log_ratio = data["rl/log_prob_old"] - data["rl/log_prob_new"]
                    mask = data.get("action_info/completion_mask")
                    if mask is None:
                        approx_kl = float(log_ratio.mean())
                    else:
                        approx_kl = float((log_ratio * mask).sum() / mask.sum().clamp(min=1))
                    approx_kls.append(approx_kl)
                global_batch_idx += 1

        metrics = {f"grpo/{k}": v / max(1, global_batch_idx) for k, v in running.items()}
        metrics["grpo/approx_kl"] = statistics.mean(approx_kls) if approx_kls else 0.0
        metrics["grpo/num_updates"] = float(global_batch_idx)
        return metrics
