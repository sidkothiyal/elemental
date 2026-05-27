"""Transparent caching wrapper for deterministic (frozen) `nn.Module` s.

`CachedModule` sits between a caller and any `nn.Module` whose output is
fully determined by its inputs (e.g. a frozen vision encoder).  On the first
call for a given input it runs the wrapped module normally and stores the
result; on subsequent calls with the same input it returns the cached tensor
directly, skipping the expensive forward pass.

Cache keys are derived by hashing every positional and keyword argument, so
**no changes are required to the wrapped module, the dataset, or the
trainer**.  The wrapper can be dropped into any Hydra config:

.. code-block:: yaml

   model:
     _target_: model.cached_module.CachedModule
     module:
       _target_: model.hf_llm.HFVLM
       ...
     max_memory_items: 0   # 0 = unlimited
     cache_dir: null        # set a path for disk persistence
     compress: false         # lz4-compress disk files
"""

import hashlib
import io
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

import lz4.frame as _lz4
import torch
from torch import nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------


def _hash_value(value: Any) -> bytes:
    """Return a deterministic SHA-256 digest for a single value.

    Handles :class:`torch.Tensor`, ``list``, ``tuple``, ``dict``, PIL images
    (via ``tobytes()``), and any ``repr()``-able scalar.

    Args:
        value: The value to hash. Nested containers are hashed recursively.

    Returns:
        A 32-byte SHA-256 digest uniquely representing *value*.

    Raises:
        ValueError: If *value* cannot be hashed.
    """
    h = hashlib.sha256()

    if isinstance(value, torch.Tensor):
        # Contiguous CPU bytes are the fastest reliable fingerprint.
        t = value.detach().cpu().contiguous()
        h.update(t.numpy().tobytes())
        # Include shape + dtype so tensors with identical bytes but
        # different interpretations don't collide.
        h.update(repr((tuple(t.shape), t.dtype)).encode())
    elif isinstance(value, (list, tuple)):
        for item in value:
            h.update(_hash_value(item))
    elif isinstance(value, dict):
        for k in sorted(value.keys()):
            h.update(repr(k).encode())
            h.update(_hash_value(value[k]))
    else:
        # str, int, float, PIL.Image, …
        try:
            # PIL images expose tobytes()
            h.update(value.tobytes())  # type: ignore[union-attr]
        except AttributeError:
            h.update(repr(value).encode())
        except Exception as exc:
            raise ValueError(f"Failed to hash value of type {type(value)}") from exc
    return h.digest()


def _compute_sample_key(args_i: tuple[Any, ...], kwargs_i: dict[str, Any]) -> str:
    """Compute a hex cache key for one sample's worth of arguments.

    Args:
        args_i: Positional arguments for a single sample (already sliced from
            the batch).
        kwargs_i: Keyword arguments for a single sample (already sliced).

    Returns:
        A 64-character lowercase hex string uniquely identifying the inputs.
    """
    h = hashlib.sha256()
    for a in args_i:
        h.update(_hash_value(a))
    for k in sorted(kwargs_i.keys()):
        h.update(repr(k).encode())
        h.update(_hash_value(kwargs_i[k]))
    return h.hexdigest()


def _slice_arg(value: Any, idx: int) -> Any:
    """Extract the *idx*-th sample from a batched argument.

    Args:
        value: A batched tensor, list, tuple, or shared scalar/string.
        idx: The sample index to extract.

    Returns:
        The *idx*-th element for tensors and sequences; *value* unchanged for
        scalars or strings that are shared across the batch.
    """
    if isinstance(value, torch.Tensor):
        return value[idx]
    if isinstance(value, (list, tuple)):
        return value[idx]
    # Scalars / strings that are shared across the batch (e.g. a fixed prompt
    # string) are returned as-is.
    return value


def _collate_arg(values: list[Any], original: Any) -> Any:
    """Re-batch a list of per-sample argument slices.

    Args:
        values: A list of per-sample slices produced by :func:`_slice_arg`.
        original: The full-batch argument before slicing.  Used to decide
            whether *values* should be stacked into a tensor, collected into a
            list, or returned as a shared scalar.

    Returns:
        A batched representation of *values* matching the type of *original*.
    """
    if isinstance(values[0], torch.Tensor):
        return torch.stack(values)
    if isinstance(values[0], (list, tuple)):
        return type(values[0])(values)
    # If the original arg was a list (e.g. list[str]), re-wrap slices into a list.
    if isinstance(original, (list, tuple)):
        return list(values)
    # Shared scalar — just return it once.
    return values[0]


# ---------------------------------------------------------------------------
# Disk I/O helpers (optional lz4 compression)
# ---------------------------------------------------------------------------


def _save_tensor(tensor: torch.Tensor, path: Path, compress: bool) -> None:
    """Serialize *tensor* to *path*, optionally lz4-compressing the bytes.

    Args:
        tensor: The tensor to persist.
        path: Destination file path.
        compress: When ``True``, apply lz4 frame compression before writing.
            Requires the ``lz4`` package.
    """
    buf = io.BytesIO()
    torch.save(tensor, buf)
    data = buf.getvalue()
    if compress:
        data = _lz4.compress(data)
    path.write_bytes(data)


def _load_tensor(path: Path, compress: bool) -> torch.Tensor:
    """Load a tensor previously saved by :func:`_save_tensor`.

    Args:
        path: Source file path.
        compress: When ``True``, decompress lz4 frame data before loading.

    Returns:
        The deserialized :class:`torch.Tensor`.
    """
    data = path.read_bytes()
    if compress:
        data = _lz4.decompress(data)
    buf = io.BytesIO(data)
    return torch.load(buf, weights_only=True)


# ---------------------------------------------------------------------------
# CachedModule
# ---------------------------------------------------------------------------


class CachedModule(nn.Module):
    """Transparent caching wrapper for a deterministic `nn.Module`.

    Args:
        module: The module whose outputs should be cached.
        max_memory_items: Maximum entries kept in the in-memory LRU cache.
            `0` means unlimited.
        cache_dir: If set, cached tensors are also persisted to disk under
            this directory so they survive across runs.
        compress: When `True`, disk files are lz4-compressed (requires
            the `lz4` package).
        enabled: Master switch — when `False` the wrapper is a no-op.
        disable_dynamo: When `True` (default), the wrapped module's forward
            call is excluded from `torch.compile` tracing.  Set to `False`
            if the wrapped module is compile-friendly.
    """

    def __init__(
        self,
        module: nn.Module,
        max_memory_items: int = 0,
        cache_dir: str | None = None,
        compress: bool = False,
        enabled: bool = True,
        disable_dynamo: bool = True,
    ) -> None:
        super().__init__()
        self.module = module
        self.max_memory_items = max_memory_items
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.compress = compress
        self.enabled = enabled
        self.disable_dynamo = disable_dynamo

        # key (hex str) → tensor (CPU)
        self._memory_cache: OrderedDict[str, torch.Tensor] = OrderedDict()

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- cache operations ---------------------------------------------------

    def _lookup(self, key: str) -> torch.Tensor | None:
        """Return the cached tensor for *key*, or ``None`` on a miss.

        Checks the in-memory LRU cache first, then the disk cache (if
        configured).  A disk hit is promoted to the in-memory cache.

        Args:
            key: Hex cache key produced by :func:`_compute_sample_key`.

        Returns:
            The cached CPU tensor, or ``None`` if not found.
        """
        if key in self._memory_cache:
            self._memory_cache.move_to_end(key)
            return self._memory_cache[key]
        if self.cache_dir:
            suffix = ".pt.lz4" if self.compress else ".pt"
            path = self.cache_dir / f"{key}{suffix}"
            if path.exists():
                tensor = _load_tensor(path, self.compress)
                self._memory_store(key, tensor)
                return tensor
        return None

    def _store(self, key: str, tensor: torch.Tensor) -> None:
        """Persist *tensor* under *key* in the in-memory and (optionally) disk cache.

        Args:
            key: Hex cache key produced by :func:`_compute_sample_key`.
            tensor: The tensor to store.  Moved to CPU before storing.
        """
        cpu_tensor = tensor.detach().cpu()
        self._memory_store(key, cpu_tensor)
        if self.cache_dir:
            suffix = ".pt.lz4" if self.compress else ".pt"
            _save_tensor(cpu_tensor, self.cache_dir / f"{key}{suffix}", self.compress)

    def _memory_store(self, key: str, tensor: torch.Tensor) -> None:
        """Insert *tensor* into the in-memory LRU cache, evicting if full.

        Args:
            key: Hex cache key.
            tensor: CPU tensor to store.
        """
        if self.max_memory_items == 0:
            return
        self._memory_cache[key] = tensor
        if self.max_memory_items > 0 and len(self._memory_cache) > self.max_memory_items:
            self._memory_cache.popitem(last=False)

    def clear_cache(self) -> None:
        """Remove all entries from the in-memory cache."""
        self._memory_cache.clear()

    @property
    def cache_size(self) -> int:
        """Number of entries currently in the in-memory cache."""
        return len(self._memory_cache)

    # -- forward ------------------------------------------------------------

    def _infer_batch_size(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> int:
        """Determine batch size from the first tensor or list argument.

        Args:
            args: Positional arguments passed to :meth:`forward`.
            kwargs: Keyword arguments passed to :meth:`forward`.

        Returns:
            The inferred batch size, or ``1`` if no batch dimension is found.
        """
        for a in args:
            if isinstance(a, torch.Tensor):
                return a.shape[0]
            if isinstance(a, (list, tuple)):
                return len(a)
        for a in kwargs.values():
            if isinstance(a, torch.Tensor):
                return a.shape[0]
            if isinstance(a, (list, tuple)):
                return len(a)
        return 1

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Run the wrapped module, returning a cached result when available.

        Args:
            *args: Positional arguments forwarded to the wrapped module.
            **kwargs: Keyword arguments forwarded to the wrapped module.

        Returns:
            Output tensor, assembled from cached per-sample results and any
            newly computed misses.
        """
        if not self.enabled:
            return self.module(*args, **kwargs)

        miss_args, miss_kwargs, cache_context = self._cache_lookup(args, kwargs)

        # All hits — no module call needed.
        if miss_args is None:
            return cache_context

        # Run the wrapped module — optionally outside dynamo for models with
        # compile-unfriendly ops (.item(), KV-cache guards, etc.).
        if self.disable_dynamo:
            miss_output = self._run_module_no_dynamo(miss_args, miss_kwargs)
        else:
            with torch.no_grad():
                miss_output = self.module(*miss_args, **miss_kwargs)

        return self._cache_store(miss_output, cache_context)

    @torch._dynamo.disable
    def _run_module_no_dynamo(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> torch.Tensor:
        """Run the wrapped module outside dynamo to avoid graph breaks.

        Args:
            args: Positional arguments for the miss sub-batch.
            kwargs: Keyword arguments for the miss sub-batch.

        Returns:
            Raw output tensor from the wrapped module.
        """
        with torch.no_grad():
            return self.module(*args, **kwargs)

    @torch._dynamo.disable
    def _cache_lookup(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...] | None, dict[str, Any] | None, object]:
        """Per-sample cache lookup (opaque to dynamo).

        Returns:
            (miss_args, miss_kwargs, cache_context)
            If all samples are cached, miss_args is `None` and cache_context
            is the assembled output tensor.  Otherwise cache_context is an
            opaque dict consumed by `_cache_store`.
        """
        batch_size = self._infer_batch_size(args, kwargs)

        keys: list[str] = []
        cached: dict[int, torch.Tensor] = {}
        miss_positions: list[int] = []

        for i in range(batch_size):
            args_i = tuple(_slice_arg(a, i) for a in args)
            kwargs_i = {k: _slice_arg(v, i) for k, v in kwargs.items()}
            key = _compute_sample_key(args_i, kwargs_i)
            keys.append(key)
            result = self._lookup(key)
            if result is not None:
                cached[i] = result
            else:
                miss_positions.append(i)

        if not miss_positions:
            device = self._module_device()
            return None, None, self._assemble(cached, batch_size, device)

        miss_args = tuple(_collate_arg([_slice_arg(a, i) for i in miss_positions], a) for a in args)
        miss_kwargs = {
            k: _collate_arg([_slice_arg(v, i) for i in miss_positions], v)
            for k, v in kwargs.items()
        }

        ctx = {
            "keys": keys,
            "cached": cached,
            "miss_positions": miss_positions,
            "batch_size": batch_size,
        }
        return miss_args, miss_kwargs, ctx

    @torch._dynamo.disable
    def _cache_store(self, miss_output: torch.Tensor, ctx: dict) -> torch.Tensor:
        """Store miss results and assemble the full batch (opaque to dynamo)."""
        keys = ctx["keys"]
        cached = ctx["cached"]
        miss_positions = ctx["miss_positions"]
        batch_size = ctx["batch_size"]

        for j, pos in enumerate(miss_positions):
            single = miss_output[j]
            self._store(keys[pos], single)
            cached[pos] = single.detach().cpu()

        return self._assemble(cached, batch_size, miss_output.device)

    # -- helpers ------------------------------------------------------------

    def _module_device(self) -> torch.device:
        """Return the device of the wrapped module's first parameter.

        Returns:
            The :class:`torch.device` of the first parameter, or ``cpu`` if
            the module has no parameters.
        """
        try:
            return next(self.module.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @staticmethod
    def _assemble(
        results: dict[int, torch.Tensor], batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Stack per-sample tensors into a batched output tensor.

        Args:
            results: Mapping from sample index to per-sample tensor (CPU).
            batch_size: Total number of samples in the batch.
            device: Target device for the assembled tensor.

        Returns:
            A tensor of shape ``(batch_size, *sample_shape)`` on *device*.
        """
        return torch.stack([results[i].to(device) for i in range(batch_size)])
