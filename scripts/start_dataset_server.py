#!/usr/bin/env python3
"""Start a dataset server using the same Hydra config as training.

The server instantiates the dataset from the training config's
``dataloader.<split>.dataset`` section, then serves processed samples
over ZeroMQ.

Usage
-----
::

    # Serve the training split of a compression autoencoder experiment
    python scripts/start_dataset_server.py \\
        experiment/autoencoder/compression_ae \\
        --host 0.0.0.0 --port 5555 --split train --num_workers 4

    # Health check an already-running server
    python scripts/start_dataset_server.py --check --host 192.168.1.100 --port 5555

    # Benchmark compression for a given config
    python scripts/start_dataset_server.py \\
        experiment/autoencoder/compression_ae \\
        --benchmark --split train

Environment variables
---------------------
``ELEMENTAL_DATASET_SERVER_TOKEN``
    Shared secret for HMAC authentication.  Set on both server and
    training client.
"""

import logging
import os
import sys
import time

# Ensure the src directory is on the path so Hydra-based imports work
_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# pylint: disable=wrong-import-position
import fire  # noqa: E402
import hydra  # noqa: E402
import hydra.utils as hy_utils  # noqa: E402

from dataset.dataset_server import DatasetServer  # noqa: E402
from dataset.remote_dataset import RemoteDataset  # noqa: E402
from dataset.serialization import estimate_sample_size, serialize_sample  # noqa: E402

# pylint: enable=wrong-import-position

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _check_server(host: str = "127.0.0.1", port: int = 5555) -> None:
    """Send a PING to a running server and print the response."""

    ds = RemoteDataset(server_host=host, server_port=port, timeout_ms=5000)
    resp = ds.ping()
    logger.info("Server response: %s", resp)


def _benchmark(
    config_file: str,
    split: str = "train",
    num_samples: int = 100,
) -> None:
    """Benchmark serialization speed for lz4 and zstd on real samples."""
    with hydra.initialize(version_base=None, config_path="../configs"):
        config = hydra.compose(config_name=config_file)

    dataset = hy_utils.instantiate(config.dataloader[split].dataset)
    num_samples = min(num_samples, len(dataset))
    logger.info("Benchmarking %d samples from %s split...", num_samples, split)

    for compression in ("lz4", "zstd", "none"):
        try:
            start = time.perf_counter()
            total_raw = 0
            total_compressed = 0
            for i in range(num_samples):
                sample = dataset[i]
                total_raw += estimate_sample_size(sample)
                data = serialize_sample(sample, compression)
                total_compressed += len(data)
            elapsed = time.perf_counter() - start

            logger.info(
                "  %4s: %.1f MB raw -> %.1f MB compressed (%.1f%% ratio), "
                "%.1f samples/s, %.1f MB/s throughput",
                compression,
                total_raw / 1e6,
                total_compressed / 1e6,
                total_compressed / max(total_raw, 1) * 100,
                num_samples / elapsed,
                total_compressed / 1e6 / elapsed,
            )
        except ImportError:
            logger.warning("  %s: not installed, skipping", compression)


def main(
    config_file: str | None = None,
    host: str = "0.0.0.0",
    port: int = 5555,
    split: str = "train",
    num_workers: int = 4,
    prefetch_size: int = 64,
    compression: str = "lz4",
    check: bool = False,
    benchmark: bool = False,
) -> None:
    """Start (or check/benchmark) a dataset server.

    Args:
        config_file: Hydra config name (e.g. ``experiment/autoencoder/compression_ae``).
            Required unless ``--check`` is used.
        host: Bind address for the server.
        port: Bind port for the server.
        split: Which dataloader split to serve (``train``, ``val``, ``test``).
        num_workers: Parallel sample-processing threads.
        prefetch_size: Samples to prepare ahead in the queue.
        compression: ``"lz4"``, ``"zstd"``, or ``"none"``.
        check: If True, just ping an existing server and exit.
        benchmark: If True, benchmark compression speeds and exit.
    """
    if check:
        _check_server(host, port)
        return

    if config_file is None:
        logger.error("config_file is required (unless --check is used)")
        sys.exit(1)

    if benchmark:
        _benchmark(config_file, split)
        return

    # Load training config and instantiate the dataset
    with hydra.initialize(version_base=None, config_path="../configs"):
        config = hydra.compose(config_name=config_file)

    logger.info("Instantiating dataset for split=%s from config=%s", split, config_file)
    dataset = hy_utils.instantiate(config.dataloader[split].dataset)
    logger.info("Dataset loaded: %d samples", len(dataset))

    auth_token = os.environ.get("ELEMENTAL_DATASET_SERVER_TOKEN")

    server = DatasetServer(
        dataset=dataset,
        host=host,
        port=port,
        num_workers=num_workers,
        prefetch_size=prefetch_size,
        compression=compression,
        auth_token=auth_token,
    )
    server.start()


if __name__ == "__main__":
    fire.Fire(main)
