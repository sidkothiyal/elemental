"""Tests for ``model.cached_module.CachedModule``."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn

from model.cached_module import CachedModule, _compute_sample_key, _hash_value

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _CountingLinear(nn.Module):
    """Linear layer that counts how many times forward() is called."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.call_count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.call_count += 1
        return self.linear(x)


class _KwargsModule(nn.Module):
    """Module that accepts keyword arguments (mimics HFVLM interface)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.call_count = 0

    def forward(self, texts: list[str], images: list[list[torch.Tensor]]) -> torch.Tensor:
        self.call_count += 1
        # Stack first image from each sample, project it (texts ignored)
        _ = texts
        batch = torch.stack([imgs[0] for imgs in images])
        return self.proj(batch)


# ---------------------------------------------------------------------------
# Hashing tests
# ---------------------------------------------------------------------------


class TestHashing:
    def test_same_tensor_same_hash(self):
        t = torch.randn(3, 4)
        assert _hash_value(t) == _hash_value(t.clone())

    def test_different_tensor_different_hash(self):
        assert _hash_value(torch.zeros(3)) != _hash_value(torch.ones(3))

    def test_shape_matters(self):
        a = torch.zeros(6)
        b = torch.zeros(2, 3)
        assert _hash_value(a) != _hash_value(b)

    def test_string_hashing(self):
        assert _hash_value("hello") == _hash_value("hello")
        assert _hash_value("hello") != _hash_value("world")

    def test_nested_list_hashing(self):
        a = [[torch.ones(2)], [torch.zeros(2)]]
        b = [[torch.ones(2)], [torch.zeros(2)]]
        assert _hash_value(a) == _hash_value(b)

    def test_sample_key_deterministic(self):
        args = (torch.randn(4),)
        kwargs = {"text": "hi"}
        assert _compute_sample_key(args, kwargs) == _compute_sample_key(args, kwargs)


# ---------------------------------------------------------------------------
# CachedModule tests — positional args (tensor batches)
# ---------------------------------------------------------------------------


class TestCachedModuleTensorArgs:
    def test_cache_hit_skips_forward(self):
        inner = _CountingLinear(4, 8)
        cached = CachedModule(inner, max_memory_items=-1, enabled=True)

        x = torch.randn(2, 4)
        out1 = cached(x)
        assert inner.call_count == 1  # full batch miss

        out2 = cached(x)
        assert inner.call_count == 1  # all hits, no new forward
        assert torch.allclose(out1, out2)

    def test_partial_cache_hit(self):
        inner = _CountingLinear(4, 8)
        cached = CachedModule(inner, max_memory_items=-1, enabled=True)

        x1 = torch.randn(1, 4)
        x2 = torch.randn(1, 4)

        cached(x1)
        assert inner.call_count == 1

        # Batch with one known and one new sample
        x_mixed = torch.cat([x1, x2], dim=0)
        cached(x_mixed)
        assert inner.call_count == 2  # only x2 computed

    def test_disabled_always_forwards(self):
        inner = _CountingLinear(4, 8)
        cached = CachedModule(inner, enabled=False)

        x = torch.randn(2, 4)
        cached(x)
        cached(x)
        assert inner.call_count == 2

    def test_lru_eviction(self):
        inner = _CountingLinear(4, 8)
        cached = CachedModule(inner, max_memory_items=2)

        a, b, c = torch.randn(1, 4), torch.randn(1, 4), torch.randn(1, 4)
        cached(a)
        cached(b)
        assert cached.cache_size == 2

        cached(c)  # evicts a
        assert cached.cache_size == 2

        # a is evicted → recompute
        cached(a)
        assert inner.call_count == 4  # a(1) + b(1) + c(1) + a_again(1)

    def test_clear_cache(self):
        inner = _CountingLinear(4, 8)
        cached = CachedModule(inner, max_memory_items=-1)

        cached(torch.randn(2, 4))
        assert cached.cache_size == 2
        cached.clear_cache()
        assert cached.cache_size == 0


# ---------------------------------------------------------------------------
# CachedModule tests — keyword args (mimics VLAConditioner → HFVLM call)
# ---------------------------------------------------------------------------


class TestCachedModuleKwargs:
    def test_kwargs_caching(self):
        inner = _KwargsModule(dim=4)
        cached = CachedModule(inner, max_memory_items=-1, enabled=True)

        texts = ["hello", "world"]
        images = [[torch.randn(4)], [torch.randn(4)]]

        out1 = cached(texts=texts, images=images)
        assert inner.call_count == 1

        out2 = cached(texts=texts, images=images)
        assert inner.call_count == 1
        assert torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# Disk cache tests
# ---------------------------------------------------------------------------


class TestDiskCache:
    def test_disk_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            inner = _CountingLinear(4, 8)
            cached = CachedModule(inner, cache_dir=tmpdir)

            x = torch.randn(2, 4)
            out1 = cached(x)
            assert inner.call_count == 1

            # New wrapper, same dir → should load from disk
            inner2 = _CountingLinear(4, 8)
            inner2.linear = inner.linear  # same weights
            cached2 = CachedModule(inner2, cache_dir=tmpdir)

            out2 = cached2(x)
            assert inner2.call_count == 0
            assert torch.allclose(out1, out2)

    def test_disk_lz4_compression(self):
        pytest.importorskip("lz4.frame")

        with tempfile.TemporaryDirectory() as tmpdir:
            inner = _CountingLinear(4, 8)
            cached = CachedModule(inner, cache_dir=tmpdir, compress=True)

            x = torch.randn(2, 4)
            out1 = cached(x)

            # Verify files have .lz4 suffix
            files = list(Path(tmpdir).glob("*.pt.lz4"))
            assert len(files) == 2

            # New wrapper loads compressed files
            inner2 = _CountingLinear(4, 8)
            inner2.linear = inner.linear
            cached2 = CachedModule(inner2, cache_dir=tmpdir, compress=True)
            out2 = cached2(x)
            assert inner2.call_count == 0
            assert torch.allclose(out1, out2)
