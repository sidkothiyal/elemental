"""Unit tests for data_processor.tokenizer.Tokenizer.

The Tokenizer loads from pre-processed lz4 pkl files, so we create a minimal
fixture in the same format rather than testing on real data.
"""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pathlib
import tempfile

import dill
import lz4.frame as lz4
import pytest
import torch

from data_processor.tokenizer import Tokenizer


def _write_tokenizer_data(directory: pathlib.Path, text: str = "hello world"):
    """Create the lz4/dill data structure the Tokenizer expects."""
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = dict(enumerate(chars))
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    for split in ("train", "val"):
        path = directory / f"{split}_data.pkl.lz4"
        with lz4.open(str(path), "wb") as f:
            dill.dump((chars, stoi, itos, data), f)


@pytest.mark.unit
class TestTokenizer:
    def _make_tokenizer(self, tmp_dir, text="hello world"):
        d = pathlib.Path(tmp_dir)
        _write_tokenizer_data(d, text=text)
        return Tokenizer(data_path=str(d))

    def test_vocab_size_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            tok = self._make_tokenizer(tmp)
            assert len(tok.stoi) > 0

    def test_encode_returns_list_of_ints(self):
        with tempfile.TemporaryDirectory() as tmp:
            tok = self._make_tokenizer(tmp, text="abc")
            encoded = tok.encode("abc")
            assert isinstance(encoded, list)
            assert all(isinstance(i, int) for i in encoded)

    def test_encode_decode_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            tok = self._make_tokenizer(tmp, text="abcde")
            text = "abc"
            encoded = tok.encode(text)
            decoded = tok.decode([torch.tensor(i) for i in encoded])
            assert decoded == text

    def test_vocab_covers_all_chars(self):
        with tempfile.TemporaryDirectory() as tmp:
            tok = self._make_tokenizer(tmp, text="hello world")
            unique_chars = set("hello world")
            assert len(tok.chars) == len(unique_chars)
