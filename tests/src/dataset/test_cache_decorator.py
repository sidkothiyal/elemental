"""Unit tests for cache_getitem decorator."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

import tempfile
from pathlib import Path

import pytest
import torch

from dataset.cache_decorator import cache_getitem


def _make_dataset_class(cache_dir):
    """Create a dataset class with cached __getitem__."""

    class CachedDataset:
        split = "train"
        call_count = 0

        @cache_getitem(cache_dir=cache_dir)
        def __getitem__(self, idx):
            self.call_count += 1
            return {"data": torch.randn(3, 8, 8), "label": idx}

    return CachedDataset()


@pytest.mark.unit
class TestCacheGetitem:
    def test_cache_miss_then_hit(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = _make_dataset_class(tmp)
            result1 = ds[0]
            assert ds.call_count == 1

            result2 = ds[0]
            assert ds.call_count == 1  # cached, no recompute

            assert torch.equal(result1["data"], result2["data"])
            assert result1["label"] == result2["label"]

    def test_different_indices_different_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = _make_dataset_class(tmp)
            _ = ds[0]
            _ = ds[1]
            assert ds.call_count == 2

    def test_cache_files_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = _make_dataset_class(tmp)
            _ = ds[0]
            files = list(Path(tmp).glob("getitem_*.dill.lz4"))
            assert len(files) == 1

    def test_corrupted_cache_recomputes(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = _make_dataset_class(tmp)
            _ = ds[0]
            # Corrupt the cache file
            cache_files = list(Path(tmp).glob("getitem_*.dill.lz4"))
            cache_files[0].write_bytes(b"corrupted")

            _ = ds[0]
            assert ds.call_count == 2  # recomputed after corruption
