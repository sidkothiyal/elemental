"""Byte-level BPE tokenizer used by the nanochat pipeline.

Wraps HuggingFace's ``tokenizers`` library with three additions tailored for
chat training:

1. Four nanochat-style special tokens (``<|bos|>``, ``<|user|>``, ``<|assistant|>``,
   ``<|eot|>``) registered as added tokens so they're never split by BPE.
2. ``encode_chat`` produces both ``input_ids`` and a per-token ``loss_mask`` —
   the SFT trainer multiplies the next-token loss by this mask, training only
   on assistant content + its terminating ``<|eot|>``.
3. ``train_bpe`` is a thin driver that materializes a tokenizer artifact from a
   corpus iterator; used by the standalone tokenizer-training entry point.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Sequence

from tokenizers import Tokenizer as HFTokenizer  # type: ignore[attr-defined]
from tokenizers.decoders import ByteLevel as ByteLevelDecoder  # type: ignore[attr-defined]
from tokenizers.models import BPE  # type: ignore[attr-defined]
from tokenizers.pre_tokenizers import (  # type: ignore[attr-defined]
    ByteLevel as ByteLevelPreTokenizer,
)
from tokenizers.trainers import BpeTrainer  # type: ignore[attr-defined]

BOS = "<|bos|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
EOT = "<|eot|>"
MASK = "<|mask|>"
DEFAULT_SPECIAL_TOKENS: tuple[str, ...] = (BOS, USER, ASSISTANT, EOT)


@dataclass
class ChatMessage:
    """One turn in a chat conversation."""

    role: str  # "user" or "assistant"
    content: str


class BPETokenizer:
    """Byte-level BPE tokenizer + chat-aware encoding."""

    def __init__(self, tokenizer_path: str, *, register_mask: bool = False):
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"tokenizer artifact not found at {tokenizer_path!r}")
        self._tokenizer = HFTokenizer.from_file(tokenizer_path)
        self.special_token_ids: dict[str, int] = {}
        for tok in DEFAULT_SPECIAL_TOKENS:
            tok_id = self._tokenizer.token_to_id(tok)
            if tok_id is None:
                raise ValueError(
                    f"required special token {tok!r} missing from tokenizer at {tokenizer_path!r}"
                )
            self.special_token_ids[tok] = tok_id
        if register_mask:
            # Reserve a virtual ``<|mask|>`` sentinel id one past the real
            # vocabulary. The HF tokenizer is not modified — this id is never
            # produced by ``encode()`` nor consumed by ``decode()``; it is
            # only inserted into already-tokenized streams by masked-diffusion
            # model code (LLaDA-style training).
            self.special_token_ids[MASK] = self._tokenizer.get_vocab_size()

    @property
    def vocab_size(self) -> int:
        """Total vocab size, including the virtual ``<|mask|>`` sentinel if registered."""
        base = self._tokenizer.get_vocab_size()
        return base + (1 if MASK in self.special_token_ids else 0)

    @property
    def bos_token_id(self) -> int:
        """ID of the ``<|bos|>`` special token."""
        return self.special_token_ids[BOS]

    @property
    def eot_token_id(self) -> int:
        """ID of the ``<|eot|>`` end-of-turn special token."""
        return self.special_token_ids[EOT]

    @property
    def mask_token_id(self) -> int:
        """ID of the virtual ``<|mask|>`` sentinel (only valid when ``register_mask=True``)."""
        if MASK not in self.special_token_ids:
            raise RuntimeError(
                "mask token not registered — pass register_mask=True to BPETokenizer."
            )
        return self.special_token_ids[MASK]

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        """Encode raw text. Special tokens are NOT inserted unless explicitly requested."""
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = False) -> str:
        """Decode token IDs back to text."""
        return self._tokenizer.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def encode_chat(
        self,
        messages: Sequence[ChatMessage | dict],
        *,
        add_generation_prompt: bool = False,
    ) -> dict[str, list[int]]:
        """Encode a chat conversation into ``input_ids`` + ``loss_mask``.

        Format produced::

            <|bos|><|user|>...<|eot|><|assistant|>...<|eot|>...

        ``loss_mask[i] == 1`` exactly on assistant content tokens **and** the
        ``<|eot|>`` that terminates each assistant turn — so the model learns
        both what to say and when to stop. All other positions (the BOS,
        role markers, user content, padding) get mask ``0``.

        If ``add_generation_prompt`` is True, append ``<|assistant|>`` at the
        end (mask 0) to set up inference-time generation.
        """
        bos = self.special_token_ids[BOS]
        user = self.special_token_ids[USER]
        asst = self.special_token_ids[ASSISTANT]
        eot = self.special_token_ids[EOT]

        ids: list[int] = [bos]
        mask: list[int] = [0]

        for raw in messages:
            msg = raw if isinstance(raw, ChatMessage) else ChatMessage(**raw)
            content_ids = self.encode(msg.content)
            if msg.role == "user":
                ids.append(user)
                mask.append(0)
                ids.extend(content_ids)
                mask.extend([0] * len(content_ids))
                ids.append(eot)
                mask.append(0)
            elif msg.role == "assistant":
                ids.append(asst)
                mask.append(0)  # the role marker itself is not a training target
                ids.extend(content_ids)
                mask.extend([1] * len(content_ids))
                ids.append(eot)
                mask.append(1)
            else:
                raise ValueError(f"unknown role {msg.role!r}; expected 'user' or 'assistant'")

        if add_generation_prompt:
            ids.append(asst)
            mask.append(0)

        return {"input_ids": ids, "loss_mask": mask}


def train_bpe(
    text_iterator: Iterable[str],
    *,
    vocab_size: int,
    save_path: str,
    special_tokens: Sequence[str] = DEFAULT_SPECIAL_TOKENS,
    min_frequency: int = 2,
) -> str:
    """Train a byte-level BPE tokenizer and write it to ``save_path``.

    Returns the path written. The file is a JSON artifact loadable via
    ``BPETokenizer(save_path)`` (which relies on ``tokenizers.Tokenizer.from_file``).
    """
    tokenizer = HFTokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=list(special_tokens),
        initial_alphabet=ByteLevelPreTokenizer.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(text_iterator, trainer)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
    tokenizer.save(save_path)
    return save_path
