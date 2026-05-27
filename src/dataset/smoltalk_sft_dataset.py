"""SmolTalk-style SFT dataset with assistant-only loss masking.

Each conversation is encoded with the project's BPE tokenizer using the
nanochat chat template, then converted into an already-shifted training
example::

    {"input": ids[:-1], "target": targets}

where ``targets[i]`` equals ``ids[i + 1]`` for assistant-content positions
(including the terminating ``<|eot|>``) and ``ignore_index`` (default ``-100``)
elsewhere — letting us reuse the existing ``ShiftedTeacherForcing(shift=0)``
loss without a separate masked variant.
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch
from datasets import load_dataset

from data_processor.bpe_tokenizer import BPETokenizer, ChatMessage
from dataset.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


def encode_sft_example(
    messages: Sequence[ChatMessage | dict],
    tokenizer: BPETokenizer,
    *,
    max_length: int,
    ignore_index: int = -100,
    pad_token_id: int | None = None,
) -> dict[str, torch.Tensor]:
    """Encode one conversation into a fixed-length, already-shifted training example.

    Args:
        messages: list of ``ChatMessage`` (or role/content dicts) for one conversation.
        tokenizer: project ``BPETokenizer``.
        max_length: number of *input* tokens; the underlying chat encoding is
            taken to ``max_length + 1`` and shifted by one.
        ignore_index: value for non-assistant target positions.
        pad_token_id: token used to right-pad short examples; defaults to the
            tokenizer's ``<|eot|>``.

    Returns:
        ``{"input": (L,), "target": (L,)}`` with ``L == max_length``.
    """
    if pad_token_id is None:
        pad_token_id = tokenizer.eot_token_id

    out = tokenizer.encode_chat(messages, add_generation_prompt=False)
    ids = out["input_ids"][: max_length + 1]
    mask = out["loss_mask"][: max_length + 1]

    pad_needed = (max_length + 1) - len(ids)
    if pad_needed > 0:
        ids = ids + [pad_token_id] * pad_needed
        mask = mask + [0] * pad_needed

    input_ids = ids[:-1]
    next_ids = ids[1:]
    next_mask = mask[1:]
    targets = [next_id if keep else ignore_index for next_id, keep in zip(next_ids, next_mask)]

    return {
        "input": torch.tensor(input_ids, dtype=torch.long),
        "target": torch.tensor(targets, dtype=torch.long),
    }


class SmolTalkSFTDataset(BaseDataset):
    """HuggingFace ``smoltalk``-style chat SFT dataset.

    Args:
        tokenizer_path: path to the saved nanochat ``BPETokenizer`` artifact.
        max_length: input sequence length per example (target has the same length).
        dataset: HF dataset id; defaults to ``HuggingFaceTB/smoltalk``.
        split: which split to load.
        name: optional HF dataset configuration.
        messages_key: field on each example holding the conversation; SmolTalk
            uses ``"messages"``.
        ignore_index: target value for non-assistant positions.
    """

    def __init__(
        self,
        *,
        tokenizer_path: str,
        max_length: int,
        dataset: str = "HuggingFaceTB/smoltalk",
        split: str = "train",
        name: str | None = None,
        messages_key: str = "messages",
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.tokenizer = BPETokenizer(tokenizer_path)
        self.max_length = max_length
        self.messages_key = messages_key
        self.ignore_index = ignore_index
        self._data = load_dataset(dataset, name=name, split=split)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self._data[idx]
        messages = ex[self.messages_key]
        return encode_sft_example(
            messages,
            self.tokenizer,
            max_length=self.max_length,
            ignore_index=self.ignore_index,
            pad_token_id=self.tokenizer.eot_token_id,
        )
