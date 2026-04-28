"""GSM8K prompts dataset for GRPO rollouts.

Loads the standard GSM8K split and exposes ``(prompt_text, answer_text)`` pairs.
The dataset itself does not tokenize — the GRPO policy wrapper takes raw text,
encodes the chat-formatted prompt at sample-time, and returns the canonical
reference answer to the verifier.
"""

from __future__ import annotations

import logging

from datasets import load_dataset

from dataset.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class GSM8KDataset(BaseDataset):
    """Map-style GSM8K problems for GRPO rollouts.

    Args:
        dataset: HF dataset id; defaults to ``openai/gsm8k``.
        split: which split to load (``train`` or ``test``).
        name: HF config; GSM8K uses ``"main"``.
        question_key: field on each example with the problem text.
        answer_key: field on each example with the reference solution.
    """

    def __init__(
        self,
        *,
        dataset: str = "openai/gsm8k",
        split: str = "train",
        name: str = "main",
        question_key: str = "question",
        answer_key: str = "answer",
    ) -> None:
        super().__init__()
        self._data = load_dataset(dataset, name=name, split=split)
        self.question_key = question_key
        self.answer_key = answer_key

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict[str, str]:
        ex = self._data[idx]
        return {
            "prompt_text": ex[self.question_key],
            "answer_text": ex[self.answer_key],
        }
