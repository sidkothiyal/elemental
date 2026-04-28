"""A no-step "environment" that streams prompt batches to GRPO rollout collection.

GRPO has no transition dynamics — each rollout is a single batch of prompt →
completion → reward. We still surface the data through ``BaseVecEnv`` so the
existing ``RLTrainer`` plumbing (env build hook, env.close on test) is reused
unchanged.

``reset`` yields the next batch of prompts. ``step`` is a no-op that returns
zero reward and ``done = True`` for every env — GRPO's ``collect_rollouts``
ignores the return value entirely and reads its rewards from the verifier.
"""

from __future__ import annotations

import threading
from typing import Any, Iterator

import torch

from data_processor.bpe_tokenizer import BPETokenizer
from env.base_vec_env import BaseVecEnv


class PromptIteratorEnv(BaseVecEnv):
    """Yield batches of chat-formatted prompts from a GSM8K-style dataset.

    Args:
        dataset: a map-style dataset returning ``{"prompt_text", "answer_text"}`` per item.
        tokenizer_path: path to the saved nanochat ``BPETokenizer`` artifact.
        batch_size: number of prompts per ``reset()``.
        max_prompt_length: truncate prompt token sequences to this length;
            also the tensor width returned in ``input_ids``.
    """

    action_dim = 0
    observation_spec: dict[str, Any] = {"input_ids": "int64"}
    action_spec: dict[str, Any] = {}

    def __init__(
        self,
        *,
        dataset: Any,
        tokenizer_path: str,
        batch_size: int,
        max_prompt_length: int,
    ) -> None:
        self.dataset = dataset
        self.tokenizer = BPETokenizer(tokenizer_path)
        self.num_envs = batch_size
        self.max_prompt_length = max_prompt_length
        self._lock = threading.Lock()
        self._index = 0

    def _next_batch(self) -> list[dict[str, str]]:
        with self._lock:
            n = len(self.dataset)
            batch = []
            for _ in range(self.num_envs):
                ex = self.dataset[self._index % n]
                self._index += 1
                batch.append(ex)
            return batch

    def reset(self) -> dict[str, Any]:
        batch = self._next_batch()
        encoded = []
        for ex in batch:
            out = self.tokenizer.encode_chat(
                [{"role": "user", "content": ex["prompt_text"]}],
                add_generation_prompt=True,
            )
            ids = out["input_ids"][-self.max_prompt_length :]
            encoded.append(ids)

        max_len = min(self.max_prompt_length, max(len(ids) for ids in encoded))
        pad_id = self.tokenizer.eot_token_id
        padded: list[list[int]] = []
        for ids in encoded:
            pad_len = max_len - len(ids)
            padded.append([pad_id] * pad_len + ids[-max_len:])
        prompt_ids = torch.tensor(padded, dtype=torch.long)

        return {
            "input_ids": prompt_ids,
            "prompt_text": [ex["prompt_text"] for ex in batch],
            "answer_text": [ex["answer_text"] for ex in batch],
        }

    def step(
        self, actions: torch.Tensor
    ) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, dict[str, Any]]:
        del actions  # unused: GRPO consumes prompts via reset() only
        b = self.num_envs
        empty_obs = {"input_ids": torch.zeros(b, 1, dtype=torch.long)}
        return empty_obs, torch.zeros(b), torch.ones(b), {}

    def close(self) -> None:
        pass

    def __iter__(self) -> Iterator[dict[str, Any]]:
        while True:
            yield self.reset()
