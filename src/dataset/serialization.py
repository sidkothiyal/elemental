"""Serialization utilities for tensor data over the network.

Provides efficient serialization and deserialization of sample dicts
(``dict[str, torch.Tensor]``) using ``torch.save``/``torch.load`` with
optional compression (lz4, zstd).  Used by both the dataset server
(``dataset_server.py``) and the remote dataset client
(``remote_dataset.py``).
"""

import io
import logging

import lz4.frame
import torch
import zstandard

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compression helpers
# ---------------------------------------------------------------------------

_COMPRESSORS: dict[str, tuple] = {}


def _get_compressor(compression: str):
    """Return (compress_fn, decompress_fn) for the given compression name.

    Lazily imports compression libraries so that unused backends don't need
    to be installed.
    """
    if compression in _COMPRESSORS:
        return _COMPRESSORS[compression]

    if compression == "lz4":
        _COMPRESSORS["lz4"] = (lz4.frame.compress, lz4.frame.decompress)
    elif compression == "zstd":
        ctx_c = zstandard.ZstdCompressor()
        ctx_d = zstandard.ZstdDecompressor()
        _COMPRESSORS["zstd"] = (ctx_c.compress, ctx_d.decompress)
    elif compression == "none":
        _COMPRESSORS["none"] = (lambda x: x, lambda x: x)
    else:
        raise ValueError(
            f"Unsupported compression: {compression!r}. Choose from 'lz4', 'zstd', or 'none'."
        )

    return _COMPRESSORS[compression]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def serialize_sample(
    sample: dict[str, torch.Tensor],
    compression: str = "lz4",
) -> bytes:
    """Serialize a sample dict to compressed bytes.

    Uses ``torch.save`` into an in-memory buffer (preserves tensor dtypes,
    shapes, and strides exactly) then applies the requested compression.

    Args:
        sample: Dictionary mapping string keys to tensors (or other
            picklable values).
        compression: Compression algorithm — ``"lz4"`` (default),
            ``"zstd"``, or ``"none"``.

    Returns:
        Compressed bytes suitable for network transmission.
    """
    buf = io.BytesIO()
    torch.save(sample, buf)
    raw = buf.getvalue()

    compress_fn, _ = _get_compressor(compression)
    return compress_fn(raw)


def deserialize_sample(
    data: bytes,
    compression: str = "lz4",
) -> dict[str, torch.Tensor]:
    """Decompress and deserialize bytes back to a sample dict.

    Inverse of :func:`serialize_sample`.

    Args:
        data: Compressed bytes produced by :func:`serialize_sample`.
        compression: Must match the compression used during serialization.

    Returns:
        Reconstructed sample dictionary.
    """
    _, decompress_fn = _get_compressor(compression)
    raw = decompress_fn(data)

    buf = io.BytesIO(raw)
    return torch.load(buf, weights_only=False)


def estimate_sample_size(sample: dict[str, torch.Tensor]) -> int:
    """Estimate the uncompressed size in bytes of a sample dict.

    Useful for bandwidth monitoring and compression ratio reporting.

    Args:
        sample: Sample dictionary.

    Returns:
        Approximate size in bytes.
    """
    total = 0
    for value in sample.values():
        if isinstance(value, torch.Tensor):
            total += value.nelement() * value.element_size()
        else:
            # Rough estimate for non-tensor values
            total += 64
    return total
