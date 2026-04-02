"""S3-compatible remote cache backends for torch.compile inductor and Triton caches.

PyTorch's OSS inductor remote cache only supports Redis. This module provides
S3-compatible alternatives that plug into PyTorch's existing ``backend_override_cls``
hook (for inductor caches) and Triton's ``TRITON_REMOTE_CACHE_BACKEND`` mechanism.

## The five torch.compile cache artifacts

| Cache              | Size       | Regen cost          | Remote cache ROI    |
|--------------------|------------|---------------------|---------------------|
| AutotuningCache    | KB-low MB  | 30-300s+ (GPU bench)| Best — tiny + slow  |
| FXGraphCache       | 1-50 MB    | 5-40s (Inductor)    | Good                |
| AOTAutogradCache   | 1-20 MB    | 2-30s (autograd)    | Good (training)     |
| PGO cache          | KB         | negligible          | Nice-to-have        |
| TritonCache/cubins | 50 MB-4 GB | 10-120s+ (Triton)   | Risky on slow nets  |

For slow/high-latency object stores (R2, S3 cross-region), prioritise the first
three by default. TritonCache cubins are already bundled into FXGraphCache entries
via ``bundle_triton_into_fx_graph_cache`` (enabled by default in OSS PyTorch), so
a remote hit on FXGraphCache implicitly restores the cubins as well. Only enable
the standalone TritonCache remote if you are on a fast network and have GBs of
free bucket bandwidth.

## Environment variables

TORCHINDUCTOR_S3_CACHE_BUCKET : str  **Required**
    S3 bucket name for all inductor caches.
TORCHINDUCTOR_S3_CACHE_REGION : str  optional (default "auto")
    AWS region — "auto" works for R2 and most providers.
AWS_ENDPOINT_URL : str  optional
    Custom endpoint for S3-compatible stores (e.g. R2, MinIO).
AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY : str
    Standard boto3 credentials, picked up automatically.

## Enabling torch.compile remote caches

Set the inductor feature-flag env vars **and** add to the experiment config:

    export TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE=1
    export TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE=1
    export TORCHINDUCTOR_BUNDLED_AUTOTUNE_REMOTE_CACHE=1

    torch_compile:
      remote_cache:
        enabled: true

## Enabling the Triton kernel remote cache (optional, slow-network warning)

Triton has its own separate cache system. To use S3 for it, set:

    export TRITON_CACHE_MANAGER=triton.runtime.cache.RemoteCacheManager
    export TRITON_REMOTE_CACHE_BACKEND=utils.s3_cache_backend.TritonS3RemoteCacheBackend

And enable it in config:

    torch_compile:
      remote_cache:
        triton: true
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional

import boto3
from torch._inductor.remote_cache import (
    RemoteAOTAutogradCache,
    RemoteAutotuneCache,
    RemoteBundledAutotuneCache,
    RemoteCacheBackend,
    RemoteDynamoPGOCache,
    RemoteFxGraphCache,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Process-level singletons shared across all S3RemoteCacheBackend instances
#
# PyTorch inductor creates a new RemoteCache (and thus a new backend) for
# every kernel/subgraph it compiles. A large model can have hundreds of
# kernels, so constructing a boto3 client per instance causes hundreds of
# DNS lookups and connection setups — the visible "startup delay" at the
# beginning of each epoch during torch.compile.
#
# Solution: one boto3 client + one memory cache per process, shared by all
# S3RemoteCacheBackend instances via module-level variables.
# ---------------------------------------------------------------------------

_client_lock: threading.Lock = threading.Lock()
_s3_client: Optional[Any] = None  # boto3 client singleton
_s3_bucket: str = ""  # resolved once alongside the client
_mem_cache: dict[str, Optional[bytes]] = {}  # process-wide hit/miss cache


def _get_s3_client() -> tuple[Any, str]:
    """Return the process-level (client, bucket) pair, creating it on first call."""
    global _s3_client, _s3_bucket
    if _s3_client is not None:
        return _s3_client, _s3_bucket
    with _client_lock:
        if _s3_client is not None:  # double-checked locking
            return _s3_client, _s3_bucket
        bucket = os.environ.get("TORCHINDUCTOR_S3_CACHE_BUCKET", "")
        if not bucket:
            raise RuntimeError(
                "TORCHINDUCTOR_S3_CACHE_BUCKET must be set to use the S3 remote cache backend"
            )
        endpoint_url = os.environ.get("AWS_ENDPOINT_URL")
        region = os.environ.get("TORCHINDUCTOR_S3_CACHE_REGION", "auto")
        _s3_client = boto3.client("s3", endpoint_url=endpoint_url, region_name=region)
        _s3_bucket = bucket
        logger.info(
            "S3RemoteCacheBackend: created boto3 client (bucket=%s, endpoint=%s)",
            bucket,
            endpoint_url or "default",
        )
    return _s3_client, _s3_bucket


# ---------------------------------------------------------------------------
# Inductor S3 backend  (used by FXGraph / Autotune / AOTAutograd / PGO)
# ---------------------------------------------------------------------------


class S3RemoteCacheBackend(RemoteCacheBackend[bytes]):
    """boto3/S3 implementation of PyTorch's ``RemoteCacheBackend``.

    Plugged into inductor cache classes via ``backend_override_cls`` by
    :func:`install_s3_cache_backend`.

    The boto3 client and the in-process memory cache are process-level
    singletons — all instances share them so that per-kernel instantiation
    (PyTorch creates one backend per compilation unit) does not re-create
    connections or forget previously fetched results.
    """

    def __init__(self) -> None:
        super().__init__()
        # Resolve (and lazily create) the shared client on first instantiation.
        # Subsequent instantiations return immediately with no I/O.
        self._client, self._bucket = _get_s3_client()

    def _get(self, key: str) -> Optional[bytes]:
        # Check the process-wide memory cache first — avoids S3 round-trips for
        # keys that were already fetched (or confirmed missing) in this process.
        if key in _mem_cache:
            return _mem_cache[key]
        try:
            resp = self._client.get_object(Bucket=self._bucket, Key=key)
            data = resp["Body"].read()
            _mem_cache[key] = data
            return data
        except self._client.exceptions.NoSuchKey:
            _mem_cache[key] = None
            return None

    def _put(self, key: str, data: bytes) -> None:
        self._client.put_object(Bucket=self._bucket, Key=key, Body=data)
        _mem_cache[key] = data


# ---------------------------------------------------------------------------
# Triton S3 backend  (separate interface used by triton.runtime.cache)
# ---------------------------------------------------------------------------


class TritonS3RemoteCacheBackend:
    """S3 implementation of Triton's ``RemoteCacheBackend`` interface.

    Triton's remote cache uses a different interface from PyTorch inductor's:
    ``get`` receives a list of filenames and returns a dict, ``put`` writes one
    file at a time.

    Activated by setting:
        TRITON_CACHE_MANAGER=triton.runtime.cache.RemoteCacheManager
        TRITON_REMOTE_CACHE_BACKEND=utils.s3_cache_backend.TritonS3RemoteCacheBackend

    Warning: Triton cubin files can be 50 MB - 4+ GB total. Only use this on
    fast networks. On slow stores the transfer latency defeats the purpose —
    recompiling from a cached FXGraph entry is faster.
    """

    def __init__(self, key: str) -> None:
        bucket = os.environ.get("TORCHINDUCTOR_S3_CACHE_BUCKET", "")
        if not bucket:
            raise RuntimeError(
                "TORCHINDUCTOR_S3_CACHE_BUCKET must be set to use the Triton S3 cache backend"
            )
        self._bucket = bucket
        self._prefix = f"triton/{key}"

        endpoint_url = os.environ.get("AWS_ENDPOINT_URL")
        region = os.environ.get("TORCHINDUCTOR_S3_CACHE_REGION", "auto")
        self._client = boto3.client("s3", endpoint_url=endpoint_url, region_name=region)

    def _s3_key(self, filename: str) -> str:
        return f"{self._prefix}/{filename}"

    def get(self, filenames: List[str]) -> Dict[str, bytes]:
        """Fetch cached Triton kernel binaries from S3 by filename."""
        results: Dict[str, bytes] = {}
        for filename in filenames:
            try:
                resp = self._client.get_object(Bucket=self._bucket, Key=self._s3_key(filename))
                results[filename] = resp["Body"].read()
            except self._client.exceptions.NoSuchKey:
                pass
        return results

    def put(self, filename: str, data: bytes) -> None:
        """Upload a Triton kernel binary to S3."""
        self._client.put_object(Bucket=self._bucket, Key=self._s3_key(filename), Body=data)


# ---------------------------------------------------------------------------
# Install helper
# ---------------------------------------------------------------------------


def install_s3_cache_backend(
    fx_graph: bool = True,
    autotune: bool = True,
    aot_autograd: bool = True,
    pgo: bool = False,
    triton: bool = False,
) -> None:
    """Patch inductor ``RemoteCache`` subclasses to use :class:`S3RemoteCacheBackend`.

    Must be called **before** ``torch.compile``.

    Args:
        fx_graph: Enable for FXGraphCache (1-50 MB; skips Inductor codegen).
        autotune: Enable for AutotuningCache + BundledAutotuneCache
                  (KB-low MB; best ROI — tiny but very expensive to regenerate).
        aot_autograd: Enable for AOTAutogradCache (1-20 MB; training only).
        pgo: Enable for PGO / DynamoPGO cache (KB; low priority).
        triton: Log a warning only — Triton's cache uses a separate mechanism.
                Set TRITON_CACHE_MANAGER and TRITON_REMOTE_CACHE_BACKEND env vars
                to activate it (see module docstring).
    """
    enabled: list[str] = []

    if fx_graph:
        RemoteFxGraphCache.backend_override_cls = S3RemoteCacheBackend
        enabled.append("FXGraphCache")

    if autotune:
        RemoteAutotuneCache.backend_override_cls = S3RemoteCacheBackend
        RemoteBundledAutotuneCache.backend_override_cls = S3RemoteCacheBackend
        enabled.append("AutotuningCache")

    if aot_autograd:
        RemoteAOTAutogradCache.backend_override_cls = S3RemoteCacheBackend
        enabled.append("AOTAutogradCache")

    if pgo:
        RemoteDynamoPGOCache.backend_override_cls = S3RemoteCacheBackend
        enabled.append("PGOCache")

    if triton:
        logger.warning(
            "triton=True has no effect here — Triton's cache is a separate system. "
            "To enable it set: TRITON_CACHE_MANAGER=triton.runtime.cache.RemoteCacheManager "
            "and TRITON_REMOTE_CACHE_BACKEND=utils.s3_cache_backend.TritonS3RemoteCacheBackend"
        )

    if enabled:
        logger.info("Installed S3RemoteCacheBackend for: %s", ", ".join(enabled))
    else:
        logger.warning("install_s3_cache_backend called but no cache types were enabled")
