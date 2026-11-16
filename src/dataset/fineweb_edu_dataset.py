"""Streaming token-packed dataset for nanochat-style pretraining.

Streams documents from a HuggingFace dataset (FineWeb-EDU by default), tokenizes
each one with the project's BPE tokenizer, separates docs with ``<|bos|>``, and
packs the byte-level token stream into fixed ``(seq_len + 1)``-token windows.

The dataset emits already-shifted batches so the loss can run with ``shift=0``::

    {"input": tokens[:-1], "target": tokens[1:]}

Sharding splits across two dimensions: the *node* dimension (DDP rank) is
handled inside ``__iter__`` via ``split_dataset_by_node``, while the *worker*
dimension is left to HF ``datasets``, which auto-shards an ``IterableDataset``
across DataLoader workers on its own. Doing both manually double-shards the
worker axis and silently drops most of the corpus. That's why this is an
``IterableDataset`` and the trainer skips ``DistributedSampler`` wrapping for it.
"""

from __future__ import annotations

import logging
import zlib
from typing import Any, Iterable, Iterator

import torch
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import IterableDataset

from data_processor.bpe_tokenizer import (
    BPETokenizer,
)

logger = logging.getLogger(__name__)


def holdout_keep(doc_id: str, mod: int, residue: int, mode: str) -> bool:
    """Decide whether a document belongs to the current split.

    Hashes ``doc_id`` into ``mod`` deterministic buckets (``zlib.crc32`` — unlike
    Python's per-process-salted ``hash``, it is stable across processes and
    workers, so the same doc lands in the same bucket on every rank). The
    document is "held out" iff its bucket equals ``residue``.

    - ``mode == "keep"``    → return True only for held-out docs (the eval split).
    - ``mode == "exclude"`` → return True only for non-held-out docs (training).

    With the same ``(mod, residue)`` the two modes are exact complements, so a
    train loader (``exclude``) and a val loader (``keep``) are guaranteed disjoint
    regardless of how the stream is sharded across nodes/workers.
    """
    is_holdout = (zlib.crc32(doc_id.encode("utf-8")) % mod) == residue
    if mode == "keep":
        return is_holdout
    if mode == "exclude":
        return not is_holdout
    raise ValueError(f"holdout_mode must be 'keep' or 'exclude', got {mode!r}")


def pack_token_stream(
    text_iter: Iterable[str],
    tokenizer,
    *,
    seq_len: int,
    max_docs: int | None = None,
) -> Iterator[dict[str, torch.Tensor]]:
    """Tokenize a stream of texts and emit fixed-length training windows.

    Pure function: easy to unit-test without HuggingFace or DDP. Each yielded
    dict has shape-``(seq_len,)`` ``input`` and ``target`` tensors.
    """
    bos_id = tokenizer.bos_token_id
    buffer: list[int] = []
    n_docs = 0
    for text in text_iter:
        ids = tokenizer.encode(text)
        if not ids:
            continue
        buffer.append(bos_id)
        buffer.extend(ids)
        # Drain whole windows of size seq_len + 1 from the head of the buffer.
        while len(buffer) >= seq_len + 1:
            chunk = buffer[: seq_len + 1]
            buffer = buffer[seq_len + 1 :]
            tokens = torch.tensor(chunk, dtype=torch.long)
            yield {"input": tokens[:-1], "target": tokens[1:]}
        n_docs += 1
        if max_docs is not None and n_docs >= max_docs:
            return


def _ddp_rank_and_world() -> tuple[int, int]:
    """Return the DDP ``(rank, world_size)`` — the *node* dimension only.

    Do **not** fold the DataLoader worker dimension in here. HF ``datasets``
    already auto-shards an ``IterableDataset`` across DataLoader workers inside
    its own ``__iter__`` (keyed off ``get_worker_info()``). Sharding the worker
    dimension a second time here divides it twice and silently drops
    ``1 - 1/num_workers`` of the corpus (e.g. 75% gone at ``num_workers=4``) —
    verified empirically. So we shard only by DDP rank and let HF handle workers.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


class FineWebEduDataset(IterableDataset):  # pylint: disable=too-many-instance-attributes
    """Streaming HF dataset → BPE tokenize → packed token windows.

    Args:
        tokenizer_path: path to the saved nanochat ``BPETokenizer`` artifact.
        seq_len: number of input tokens per training window (target has the same
            length, shifted by one).
        dataset: HF dataset id (default ``HuggingFaceFW/fineweb-edu``).
        split: which split to stream.
        text_key: field name on each example holding the document text.
        name: optional HF dataset configuration (e.g. ``"sample-10BT"`` for FineWeb-EDU).
        max_docs: optional cap on the number of source documents per shard.
        holdout_mod: when set, hash each doc into this many buckets to carve out a
            deterministic held-out eval split (see :func:`holdout_keep`). ``None``
            (default) disables holdout and streams every document.
        holdout_residue: the held-out bucket index (``0 <= residue < holdout_mod``).
        holdout_mode: ``"exclude"`` drops held-out docs (training stream);
            ``"keep"`` keeps only them (eval stream). Same ``(mod, residue)`` on
            both ⇒ guaranteed-disjoint train/val.
        holdout_key: example field used as the stable hash key (FineWeb-EDU has a
            unique ``"id"`` per document).
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        tokenizer_path: str,
        seq_len: int,
        dataset: str = "HuggingFaceFW/fineweb-edu",
        split: str = "train",
        text_key: str = "text",
        name: str | None = None,
        max_docs: int | None = None,
        holdout_mod: int | None = None,
        holdout_residue: int = 0,
        holdout_mode: str = "exclude",
        holdout_key: str = "id",
    ) -> None:
        super().__init__()
        self.tokenizer_path = tokenizer_path
        self.seq_len = seq_len
        self.dataset_name = dataset
        self.split = split
        self.text_key = text_key
        self.name = name
        self.max_docs = max_docs
        self.holdout_mod = holdout_mod
        self.holdout_residue = holdout_residue
        self.holdout_mode = holdout_mode
        self.holdout_key = holdout_key
        # Stateful-resume bookkeeping: the live HF stream (for state_dict) and a
        # pending state to apply when the stream is (re)built in __iter__.
        self._hf_ds: Any | None = None
        self._pending_state: dict[str, Any] | None = None

    def _build_text_iter(self, rank: int, world_size: int) -> Iterable[str]:
        ds = load_dataset(
            self.dataset_name,
            name=self.name,
            split=self.split,
            streaming=True,
        )
        # Shard the *node* dimension only. Streaming IterableDataset has no
        # .shard(); split_dataset_by_node divides shards evenly when
        # n_shards % world_size == 0, else falls back to per-example modulo
        # skipping. Worker-level sharding is handled automatically by HF inside
        # the DataLoader workers — see _ddp_rank_and_world.
        if world_size > 1:
            ds = split_dataset_by_node(ds, rank=rank, world_size=world_size)
        # Replay a saved stream position (resume). HF IterableDataset gained
        # load_state_dict in datasets>=2.20; guard so older versions degrade to a
        # fresh stream rather than crashing.
        if self._pending_state is not None and hasattr(ds, "load_state_dict"):
            ds.load_state_dict(self._pending_state)
            self._pending_state = None
        self._hf_ds = ds
        text_key = self.text_key
        for ex in ds:
            if not self._in_split(ex):
                continue
            text = ex.get(text_key)
            if text:
                yield text

    def _in_split(self, ex: dict) -> bool:
        """Apply the deterministic holdout filter (no-op when disabled)."""
        if self.holdout_mod is None:
            return True
        doc_id = ex.get(self.holdout_key)
        if doc_id is None:
            # No stable id → fall back to the text so behaviour stays deterministic.
            doc_id = ex.get(self.text_key, "")
        return holdout_keep(str(doc_id), self.holdout_mod, self.holdout_residue, self.holdout_mode)

    def state_dict(self) -> dict[str, Any]:
        """Return the underlying HF stream position for stateful resume.

        Empty dict when no stream is active or the installed ``datasets`` lacks
        ``state_dict`` (resume then restarts from the top). Granularity is the HF
        document stream; the in-flight packing buffer (< ``seq_len + 1`` tokens)
        is intentionally not persisted.
        """
        if self._hf_ds is not None and hasattr(self._hf_ds, "state_dict"):
            return {"hf": self._hf_ds.state_dict()}
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Stash a saved stream position to replay on the next ``__iter__``."""
        self._pending_state = state.get("hf") if state else None

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        # Defer tokenizer construction so DataLoader worker forks each get a
        # fresh handle (HF tokenizers and HF datasets dislike forks otherwise).
        tokenizer = BPETokenizer(self.tokenizer_path)
        rank, world_size = _ddp_rank_and_world()
        text_iter = self._build_text_iter(rank, world_size)
        yield from pack_token_stream(
            text_iter, tokenizer, seq_len=self.seq_len, max_docs=self.max_docs
        )
