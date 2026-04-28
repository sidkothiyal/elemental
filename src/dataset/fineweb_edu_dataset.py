"""Streaming token-packed dataset for nanochat-style pretraining.

Streams documents from a HuggingFace dataset (FineWeb-EDU by default), tokenizes
each one with the project's BPE tokenizer, separates docs with ``<|bos|>``, and
packs the byte-level token stream into fixed ``(seq_len + 1)``-token windows.

The dataset emits already-shifted batches so the loss can run with ``shift=0``::

    {"input": tokens[:-1], "target": tokens[1:]}

Sharding is performed inside ``__iter__`` using rank/world_size and DataLoader
worker info — that's why this is an ``IterableDataset`` and the trainer skips
``DistributedSampler`` wrapping for it.
"""

from __future__ import annotations

import logging
from typing import Iterable, Iterator

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset, get_worker_info

from data_processor.bpe_tokenizer import (
    BPETokenizer,
)

logger = logging.getLogger(__name__)


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


def _shard_index_and_count() -> tuple[int, int]:
    """Compose (rank * num_workers + worker_id, world_size * num_workers).

    Each (rank, worker) pair gets a unique stream slot.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank, world_size = 0, 1

    info = get_worker_info()
    if info is None:
        worker_id, num_workers = 0, 1
    else:
        worker_id, num_workers = info.id, info.num_workers

    shard_idx = rank * num_workers + worker_id
    total_shards = world_size * num_workers
    return shard_idx, total_shards


class FineWebEduDataset(IterableDataset):
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
    """

    def __init__(
        self,
        *,
        tokenizer_path: str,
        seq_len: int,
        dataset: str = "HuggingFaceFW/fineweb-edu",
        split: str = "train",
        text_key: str = "text",
        name: str | None = None,
        max_docs: int | None = None,
    ) -> None:
        super().__init__()
        self.tokenizer_path = tokenizer_path
        self.seq_len = seq_len
        self.dataset_name = dataset
        self.split = split
        self.text_key = text_key
        self.name = name
        self.max_docs = max_docs

    def _build_text_iter(self, shard_idx: int, total_shards: int) -> Iterable[str]:
        ds = load_dataset(
            self.dataset_name,
            name=self.name,
            split=self.split,
            streaming=True,
        )
        ds = ds.shard(num_shards=total_shards, index=shard_idx, contiguous=False)
        text_key = self.text_key
        for ex in ds:
            text = ex.get(text_key)
            if text:
                yield text

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        # Defer tokenizer construction so DataLoader worker forks each get a
        # fresh handle (HF tokenizers and HF datasets dislike forks otherwise).
        tokenizer = BPETokenizer(self.tokenizer_path)
        shard_idx, total_shards = _shard_index_and_count()
        text_iter = self._build_text_iter(shard_idx, total_shards)
        yield from pack_token_stream(
            text_iter, tokenizer, seq_len=self.seq_len, max_docs=self.max_docs
        )
