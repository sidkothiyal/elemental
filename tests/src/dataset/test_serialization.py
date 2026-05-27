"""Unit tests for dataset.serialization (serialize/deserialize sample dicts)."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch

from dataset.serialization import deserialize_sample, serialize_sample


def _make_sample():
    return {
        "image": torch.randn(3, 16, 16),
        "label": torch.tensor(42),
    }


@pytest.mark.unit
class TestSerialization:
    @pytest.mark.parametrize("compression", ["lz4", "zstd", "none"])
    def test_roundtrip(self, compression):
        original = _make_sample()
        data = serialize_sample(original, compression=compression)
        recovered = deserialize_sample(data, compression=compression)
        for key, val in original.items():
            assert torch.allclose(val.float(), recovered[key].float())

    def test_unsupported_compression_raises(self):
        with pytest.raises(ValueError):
            serialize_sample(_make_sample(), compression="gzip")

    def test_serialized_is_bytes(self):
        data = serialize_sample(_make_sample())
        assert isinstance(data, bytes)

    def test_empty_sample_roundtrip(self):
        empty = {}
        data = serialize_sample(empty)
        assert deserialize_sample(data) == {}
