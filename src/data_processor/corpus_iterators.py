"""Lightweight text iterators consumed by the BPE trainer and dataset packers.

Kept as plain callables/classes so they're trivially Hydra-instantiable. Each
yields ``str`` documents — the BPE trainer reads them once, dataset packers
loop over them to feed token streams.
"""

from __future__ import annotations

from typing import Iterator

from datasets import load_dataset


class FileLineIterator:
    """Yield non-empty lines from a UTF-8 text file. Useful for tokenizer dev/tests."""

    def __init__(self, path: str, *, max_lines: int | None = None):
        self.path = path
        self.max_lines = max_lines

    def __iter__(self) -> Iterator[str]:
        count = 0
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield line
                count += 1
                if self.max_lines is not None and count >= self.max_lines:
                    return


class HFStreamIterator:
    """Yield ``text_key`` field from a HuggingFace dataset opened in streaming mode.

    Importing ``datasets`` lazily so the tokenizer/file-iterator paths don't
    require it.
    """

    def __init__(
        self,
        dataset: str,
        *,
        split: str = "train",
        text_key: str = "text",
        name: str | None = None,
        max_docs: int | None = None,
    ):
        self.dataset = dataset
        self.split = split
        self.text_key = text_key
        self.name = name
        self.max_docs = max_docs

    def __iter__(self) -> Iterator[str]:
        ds = load_dataset(
            self.dataset,
            name=self.name,
            split=self.split,
            streaming=True,
        )
        count = 0
        for ex in ds:
            text = ex.get(self.text_key)
            if not text:
                continue
            yield text
            count += 1
            if self.max_docs is not None and count >= self.max_docs:
                return
