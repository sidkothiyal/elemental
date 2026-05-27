"""Tests for the S3 remote cache backends for torch.compile inductor and Triton caches."""

# pylint: disable=missing-function-docstring,protected-access,unused-argument,invalid-name

import logging
import os
from unittest.mock import MagicMock, patch

import pytest
from torch._inductor.remote_cache import (
    RemoteAOTAutogradCache,
    RemoteAutotuneCache,
    RemoteBundledAutotuneCache,
    RemoteDynamoPGOCache,
    RemoteFxGraphCache,
)

import utils.s3_cache_backend as _module
from utils.s3_cache_backend import (
    S3RemoteCacheBackend,
    TritonS3RemoteCacheBackend,
    install_s3_cache_backend,
)


@pytest.fixture(autouse=True)
def reset_s3_singletons():
    """Reset module-level singletons before each test.

    S3RemoteCacheBackend uses a process-level boto3 client and memory cache so
    that per-kernel instantiation is cheap. Tests must reset these between runs
    to stay independent.
    """
    _module._s3_client = None
    _module._s3_bucket = ""
    _module._mem_cache.clear()
    yield
    _module._s3_client = None
    _module._s3_bucket = ""
    _module._mem_cache.clear()


class TestS3RemoteCacheBackend:
    """Unit tests for S3RemoteCacheBackend (inductor caches)."""

    @patch("boto3.client")
    def test_init_with_defaults(self, mock_client):
        env = {"TORCHINDUCTOR_S3_CACHE_BUCKET": "test-bucket"}
        for key in ("AWS_ENDPOINT_URL", "TORCHINDUCTOR_S3_CACHE_REGION"):
            os.environ.pop(key, None)
        with patch.dict(os.environ, env):
            os.environ.pop("AWS_ENDPOINT_URL", None)
            os.environ.pop("TORCHINDUCTOR_S3_CACHE_REGION", None)
            backend = S3RemoteCacheBackend()
        mock_client.assert_called_once_with("s3", endpoint_url=None, region_name="auto")
        assert backend._bucket == "test-bucket"

    @patch.dict(
        os.environ,
        {
            "TORCHINDUCTOR_S3_CACHE_BUCKET": "my-bucket",
            "AWS_ENDPOINT_URL": "https://abc123.r2.cloudflarestorage.com",
            "TORCHINDUCTOR_S3_CACHE_REGION": "us-east-1",
        },
    )
    @patch("boto3.client")
    def test_init_with_custom_endpoint_and_region(self, mock_client):
        backend = S3RemoteCacheBackend()
        mock_client.assert_called_once_with(
            "s3",
            endpoint_url="https://abc123.r2.cloudflarestorage.com",
            region_name="us-east-1",
        )
        assert backend._bucket == "my-bucket"

    @patch.dict(os.environ, {}, clear=False)
    @patch("boto3.client")
    def test_init_raises_without_bucket(self, mock_client):
        os.environ.pop("TORCHINDUCTOR_S3_CACHE_BUCKET", None)
        with pytest.raises(RuntimeError, match="TORCHINDUCTOR_S3_CACHE_BUCKET must be set"):
            S3RemoteCacheBackend()

    @patch.dict(os.environ, {"TORCHINDUCTOR_S3_CACHE_BUCKET": "test-bucket"})
    @patch("boto3.client")
    def test_client_is_singleton_across_instances(self, mock_client):
        """Multiple instantiations must reuse the same boto3 client."""
        b1 = S3RemoteCacheBackend()
        b2 = S3RemoteCacheBackend()
        b3 = S3RemoteCacheBackend()
        mock_client.assert_called_once()  # boto3.client() called only once
        assert b1._client is b2._client is b3._client

    @patch.dict(os.environ, {"TORCHINDUCTOR_S3_CACHE_BUCKET": "test-bucket"})
    @patch("boto3.client")
    def test_put(self, mock_client):
        backend = S3RemoteCacheBackend()
        backend._put("my-key", b"my-data")
        backend._client.put_object.assert_called_once_with(
            Bucket="test-bucket", Key="my-key", Body=b"my-data"
        )

    @patch.dict(os.environ, {"TORCHINDUCTOR_S3_CACHE_BUCKET": "test-bucket"})
    @patch("boto3.client")
    def test_get_hit(self, mock_client):
        backend = S3RemoteCacheBackend()
        body_mock = MagicMock()
        body_mock.read.return_value = b"cached-data"
        backend._client.get_object.return_value = {"Body": body_mock}

        result = backend._get("my-key")
        assert result == b"cached-data"
        backend._client.get_object.assert_called_once_with(Bucket="test-bucket", Key="my-key")

    @patch.dict(os.environ, {"TORCHINDUCTOR_S3_CACHE_BUCKET": "test-bucket"})
    @patch("boto3.client")
    def test_get_miss(self, mock_client):
        backend = S3RemoteCacheBackend()
        no_such_key = type("NoSuchKey", (Exception,), {})
        backend._client.exceptions.NoSuchKey = no_such_key
        backend._client.get_object.side_effect = no_such_key()

        result = backend._get("missing-key")
        assert result is None

    @patch.dict(os.environ, {"TORCHINDUCTOR_S3_CACHE_BUCKET": "test-bucket"})
    @patch("boto3.client")
    def test_mem_cache_avoids_repeated_s3_get(self, mock_client):
        """Second _get for the same key must not call S3 again."""
        backend = S3RemoteCacheBackend()
        body_mock = MagicMock()
        body_mock.read.return_value = b"cached-data"
        backend._client.get_object.return_value = {"Body": body_mock}

        first = backend._get("my-key")
        second = backend._get("my-key")

        assert first == b"cached-data"
        assert second == b"cached-data"
        backend._client.get_object.assert_called_once()  # S3 hit only once

    @patch.dict(os.environ, {"TORCHINDUCTOR_S3_CACHE_BUCKET": "test-bucket"})
    @patch("boto3.client")
    def test_mem_cache_remembers_miss(self, mock_client):
        """Known-miss keys must not trigger a second S3 request."""
        backend = S3RemoteCacheBackend()
        no_such_key = type("NoSuchKey", (Exception,), {})
        backend._client.exceptions.NoSuchKey = no_such_key
        backend._client.get_object.side_effect = no_such_key()

        first = backend._get("missing-key")
        second = backend._get("missing-key")

        assert first is None
        assert second is None
        backend._client.get_object.assert_called_once()  # S3 miss recorded only once

    @patch.dict(os.environ, {"TORCHINDUCTOR_S3_CACHE_BUCKET": "test-bucket"})
    @patch("boto3.client")
    def test_mem_cache_shared_across_instances(self, mock_client):
        """A second backend instance must see hits written by the first."""
        b1 = S3RemoteCacheBackend()
        b2 = S3RemoteCacheBackend()

        body_mock = MagicMock()
        body_mock.read.return_value = b"shared-data"
        b1._client.get_object.return_value = {"Body": body_mock}

        b1._get("shared-key")  # populates process-wide mem cache
        result = b2._get("shared-key")  # must hit mem cache, not S3

        assert result == b"shared-data"
        b1._client.get_object.assert_called_once()  # S3 called only once total

    @patch.dict(os.environ, {"TORCHINDUCTOR_S3_CACHE_BUCKET": "test-bucket"})
    @patch("boto3.client")
    def test_put_populates_mem_cache(self, mock_client):
        """After _put, _get must return from mem cache without touching S3."""
        backend = S3RemoteCacheBackend()

        backend._put("my-key", b"fresh-data")
        result = backend._get("my-key")

        assert result == b"fresh-data"
        backend._client.get_object.assert_not_called()


class TestTritonS3RemoteCacheBackend:
    """Unit tests for TritonS3RemoteCacheBackend."""

    @patch.dict(os.environ, {"TORCHINDUCTOR_S3_CACHE_BUCKET": "triton-bucket"})
    @patch("boto3.client")
    def test_init(self, mock_client):
        backend = TritonS3RemoteCacheBackend("abc123key")
        assert backend._bucket == "triton-bucket"
        assert backend._prefix == "triton/abc123key"

    @patch.dict(os.environ, {"TORCHINDUCTOR_S3_CACHE_BUCKET": "triton-bucket"})
    @patch("boto3.client")
    def test_put(self, mock_client):
        backend = TritonS3RemoteCacheBackend("key1")
        backend.put("kernel.cubin", b"\x7fELF")
        backend._client.put_object.assert_called_once_with(
            Bucket="triton-bucket", Key="triton/key1/kernel.cubin", Body=b"\x7fELF"
        )

    @patch.dict(os.environ, {"TORCHINDUCTOR_S3_CACHE_BUCKET": "triton-bucket"})
    @patch("boto3.client")
    def test_get_hit(self, mock_client):
        backend = TritonS3RemoteCacheBackend("key1")
        body_mock = MagicMock()
        body_mock.read.return_value = b"\x7fELF"
        backend._client.get_object.return_value = {"Body": body_mock}

        result = backend.get(["kernel.cubin"])
        assert result == {"kernel.cubin": b"\x7fELF"}

    @patch.dict(os.environ, {"TORCHINDUCTOR_S3_CACHE_BUCKET": "triton-bucket"})
    @patch("boto3.client")
    def test_get_miss(self, mock_client):
        backend = TritonS3RemoteCacheBackend("key1")
        no_such_key = type("NoSuchKey", (Exception,), {})
        backend._client.exceptions.NoSuchKey = no_such_key
        backend._client.get_object.side_effect = no_such_key()

        result = backend.get(["missing.cubin"])
        assert result == {}

    @patch.dict(os.environ, {"TORCHINDUCTOR_S3_CACHE_BUCKET": "triton-bucket"})
    @patch("boto3.client")
    def test_get_partial(self, mock_client):
        """Only returns files that exist; silently skips missing ones."""
        backend = TritonS3RemoteCacheBackend("key1")
        no_such_key = type("NoSuchKey", (Exception,), {})
        backend._client.exceptions.NoSuchKey = no_such_key

        def side_effect(Bucket, Key):
            if "missing" in Key:
                raise no_such_key()
            body = MagicMock()
            body.read.return_value = b"data"
            return {"Body": body}

        backend._client.get_object.side_effect = side_effect
        result = backend.get(["exists.cubin", "missing.cubin"])
        assert result == {"exists.cubin": b"data"}


class TestInstallS3CacheBackend:
    """Tests for the install_s3_cache_backend helper."""

    def _save_and_restore(self, classes):
        return {cls: cls.backend_override_cls for cls in classes}

    def _restore(self, originals):
        for cls, orig in originals.items():
            cls.backend_override_cls = orig

    def test_default_enables_fx_autotune_aot(self):
        all_cls = [
            RemoteFxGraphCache,
            RemoteAutotuneCache,
            RemoteBundledAutotuneCache,
            RemoteAOTAutogradCache,
            RemoteDynamoPGOCache,
        ]
        originals = self._save_and_restore(all_cls)
        try:
            install_s3_cache_backend()
            assert RemoteFxGraphCache.backend_override_cls is S3RemoteCacheBackend
            assert RemoteAutotuneCache.backend_override_cls is S3RemoteCacheBackend
            assert RemoteBundledAutotuneCache.backend_override_cls is S3RemoteCacheBackend
            assert RemoteAOTAutogradCache.backend_override_cls is S3RemoteCacheBackend
            # PGO disabled by default
            assert RemoteDynamoPGOCache.backend_override_cls is originals[RemoteDynamoPGOCache]
        finally:
            self._restore(originals)

    def test_pgo_disabled_by_default(self):
        original = RemoteDynamoPGOCache.backend_override_cls
        try:
            install_s3_cache_backend(fx_graph=False, autotune=False, aot_autograd=False)
            assert RemoteDynamoPGOCache.backend_override_cls is original
        finally:
            RemoteDynamoPGOCache.backend_override_cls = original

    def test_pgo_can_be_enabled(self):
        original = RemoteDynamoPGOCache.backend_override_cls
        try:
            install_s3_cache_backend(fx_graph=False, autotune=False, aot_autograd=False, pgo=True)
            assert RemoteDynamoPGOCache.backend_override_cls is S3RemoteCacheBackend
        finally:
            RemoteDynamoPGOCache.backend_override_cls = original

    def test_selective_enable(self):
        all_cls = [
            RemoteFxGraphCache,
            RemoteAutotuneCache,
            RemoteBundledAutotuneCache,
            RemoteAOTAutogradCache,
        ]
        originals = self._save_and_restore(all_cls)
        try:
            install_s3_cache_backend(fx_graph=True, autotune=False, aot_autograd=False)
            assert RemoteFxGraphCache.backend_override_cls is S3RemoteCacheBackend
            assert RemoteAutotuneCache.backend_override_cls is originals[RemoteAutotuneCache]
            assert RemoteAOTAutogradCache.backend_override_cls is originals[RemoteAOTAutogradCache]
        finally:
            self._restore(originals)

    def test_triton_flag_logs_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="utils.s3_cache_backend"):
            install_s3_cache_backend(
                fx_graph=False, autotune=False, aot_autograd=False, triton=True
            )
        assert "separate system" in caplog.text
