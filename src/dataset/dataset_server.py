"""Dataset server — serves processed samples over ZeroMQ.

Run this on a machine that holds the dataset.  The server instantiates the
real ``torch.utils.data.Dataset`` (via Hydra config), calls ``__getitem__``
in worker threads, compresses the result, and sends it to remote training
clients over ZeroMQ.

Protocol
--------
Clients communicate via ZeroMQ REQ/REP (or DEALER/ROUTER for async).
Every message is a JSON-encoded dict with a ``"type"`` field.

Request types:

* ``HANDSHAKE`` — authenticate and learn dataset length.
* ``GETITEM``   — fetch a single sample by index.
* ``GETBATCH``  — fetch multiple samples by index list.
* ``PING``      — health check.

See :class:`DatasetServer` for full details.
"""

import hashlib
import hmac
import json
import logging
import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import zmq
from torch.utils.data import Dataset

from dataset.serialization import estimate_sample_size, serialize_sample

logger = logging.getLogger(__name__)

# Maximum clock skew allowed for handshake timestamps (seconds)
_HANDSHAKE_TIMESTAMP_TOLERANCE = 300

# Default environment variable for the shared auth token
_AUTH_TOKEN_ENV = "ELEMENTAL_DATASET_SERVER_TOKEN"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_hmac(token: str, timestamp: str) -> str:
    """Compute HMAC-SHA256 of *timestamp* using *token* as key."""
    return hmac.new(token.encode(), timestamp.encode(), hashlib.sha256).hexdigest()


def _verify_hmac(token: str, timestamp: str, client_hmac: str) -> bool:
    """Verify a client-provided HMAC and check timestamp freshness."""
    expected = _compute_hmac(token, timestamp)
    if not hmac.compare_digest(expected, client_hmac):
        return False
    # Reject stale timestamps to prevent replay attacks
    try:
        ts = float(timestamp)
    except (ValueError, TypeError):
        return False
    return abs(time.time() - ts) < _HANDSHAKE_TIMESTAMP_TOLERANCE


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class DatasetServer:
    """Serve processed dataset samples over ZeroMQ.

    Args:
        dataset: A ``torch.utils.data.Dataset`` instance.
        host: Bind address (default ``"0.0.0.0"``).
        port: Bind port (default ``5555``).
        num_workers: Number of threads for parallel ``__getitem__`` calls.
        prefetch_size: How many samples to prepare ahead in the queue.
        compression: ``"lz4"`` (default), ``"zstd"``, or ``"none"``.
        auth_token: Shared secret for HMAC handshake.  Falls back to
            ``$ELEMENTAL_DATASET_SERVER_TOKEN`` if *None*.  If still *None*,
            authentication is disabled (a warning is logged).
    """

    def __init__(
        self,
        dataset: Dataset,
        host: str = "0.0.0.0",
        port: int = 5555,
        num_workers: int = 4,
        prefetch_size: int = 64,
        compression: str = "lz4",
        auth_token: str | None = None,
    ):
        self.dataset = dataset
        self.host = host
        self.port = port
        self.num_workers = num_workers
        self.prefetch_size = prefetch_size
        self.compression = compression
        self.auth_token = auth_token or os.environ.get(_AUTH_TOKEN_ENV)

        self._shutdown_event = threading.Event()
        self._stats = {
            "samples_served": 0,
            "bytes_sent": 0,
            "start_time": 0.0,
        }

        if self.auth_token is None:
            logger.warning(
                "No auth token provided (set $%s or pass auth_token). "
                "Server is running WITHOUT authentication.",
                _AUTH_TOKEN_ENV,
            )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start serving requests.  Blocks until :meth:`shutdown` is called."""
        self._stats["start_time"] = time.time()
        dataset_len = len(self.dataset)
        logger.info(
            "Starting DatasetServer on %s:%d  (dataset_len=%d, workers=%d, "
            "prefetch=%d, compression=%s)",
            self.host,
            self.port,
            dataset_len,
            self.num_workers,
            self.prefetch_size,
            self.compression,
        )

        # Log sample size estimate from first sample
        try:
            sample0 = self.dataset[0]
            raw_size = estimate_sample_size(sample0)
            compressed = serialize_sample(sample0, self.compression)
            logger.info(
                "Sample 0: raw ~%d bytes, compressed %d bytes (%.1f%% ratio)",
                raw_size,
                len(compressed),
                len(compressed) / max(raw_size, 1) * 100,
            )
        except Exception as e:
            logger.warning("Could not estimate sample size: %s", e)

        # Install signal handlers for graceful shutdown
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: self.shutdown())

        ctx = zmq.Context()
        socket = ctx.socket(zmq.ROUTER)
        socket.bind(f"tcp://{self.host}:{self.port}")

        # Use a poller so we can check the shutdown event periodically
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)

        # Thread pool for parallel __getitem__ processing
        executor = ThreadPoolExecutor(
            max_workers=self.num_workers,
            thread_name_prefix="ds-worker",
        )

        logger.info("Server ready — listening on tcp://%s:%d", self.host, self.port)

        try:
            while not self._shutdown_event.is_set():
                # Poll with 500ms timeout so we can check shutdown flag
                events = dict(poller.poll(500))
                if socket not in events:
                    continue

                # Multipart: [identity, empty, payload]
                frames = socket.recv_multipart()
                if len(frames) < 3:
                    continue
                identity = frames[0]
                payload = frames[-1]

                try:
                    msg = json.loads(payload)
                except json.JSONDecodeError:
                    self._send_error(socket, identity, "Invalid JSON")
                    continue

                msg_type = msg.get("type", "").upper()

                if msg_type == "HANDSHAKE":
                    self._handle_handshake(socket, identity, msg, dataset_len)
                elif msg_type == "GETITEM":
                    # Process in thread pool for parallelism
                    executor.submit(self._handle_getitem, ctx, identity, msg, socket)
                elif msg_type == "GETBATCH":
                    executor.submit(self._handle_getbatch, ctx, identity, msg, socket)
                elif msg_type == "PING":
                    self._handle_ping(socket, identity)
                else:
                    self._send_error(socket, identity, f"Unknown type: {msg_type}")

        finally:
            executor.shutdown(wait=False)
            socket.close()
            ctx.term()
            self._log_stats()
            logger.info("Server shut down.")

    def shutdown(self) -> None:
        """Signal the server to stop."""
        logger.info("Shutdown requested.")
        self._shutdown_event.set()

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _handle_handshake(
        self,
        socket: zmq.Socket,
        identity: bytes,
        msg: dict,
        dataset_len: int,
    ) -> None:
        # Verify auth if token is set
        if self.auth_token is not None:
            ts = msg.get("timestamp", "")
            client_hmac = msg.get("token", "")
            if not _verify_hmac(self.auth_token, str(ts), client_hmac):
                self._send_reply(
                    socket,
                    identity,
                    {"status": "error", "message": "Authentication failed"},
                )
                logger.warning("Authentication failed from client %r", identity[:8])
                return

        self._send_reply(
            socket,
            identity,
            {"status": "ok", "length": dataset_len, "compression": self.compression},
        )
        logger.info("Client %r authenticated (dataset_len=%d)", identity[:8], dataset_len)

    def _handle_getitem(
        self,
        ctx: zmq.Context,  # pylint: disable=unused-argument
        identity: bytes,
        msg: dict,
        socket: zmq.Socket,
    ) -> None:
        idx = msg.get("idx")
        if idx is None:
            self._send_error(socket, identity, "Missing 'idx'")
            return

        try:
            sample = self.dataset[idx]
            data = serialize_sample(sample, self.compression)
            self._send_data(socket, identity, data)
            self._stats["samples_served"] += 1
            self._stats["bytes_sent"] += len(data)
        except Exception as e:
            logger.error("Error processing idx=%s: %s", idx, e, exc_info=True)
            self._send_error(socket, identity, f"getitem error: {e}")

    def _handle_getbatch(
        self,
        ctx: zmq.Context,  # pylint: disable=unused-argument
        identity: bytes,
        msg: dict,
        socket: zmq.Socket,
    ) -> None:
        indices = msg.get("indices", [])
        if not indices:
            self._send_error(socket, identity, "Missing 'indices'")
            return

        try:
            samples = [self.dataset[i] for i in indices]
            data = serialize_sample({"__batch__": samples}, self.compression)
            self._send_data(socket, identity, data)
            self._stats["samples_served"] += len(indices)
            self._stats["bytes_sent"] += len(data)
        except Exception as e:
            logger.error("Error processing batch %s: %s", indices, e, exc_info=True)
            self._send_error(socket, identity, f"getbatch error: {e}")

    def _handle_ping(self, socket: zmq.Socket, identity: bytes) -> None:
        elapsed = time.time() - self._stats["start_time"]
        self._send_reply(
            socket,
            identity,
            {
                "status": "pong",
                "uptime_s": round(elapsed, 1),
                "samples_served": self._stats["samples_served"],
            },
        )

    # ------------------------------------------------------------------
    # Transport helpers
    # ------------------------------------------------------------------

    def _send_reply(self, socket: zmq.Socket, identity: bytes, obj: dict) -> None:
        """Send a JSON reply to the identified client."""
        socket.send_multipart([identity, b"", json.dumps(obj).encode()])

    def _send_data(self, socket: zmq.Socket, identity: bytes, data: bytes) -> None:
        """Send binary data as a multipart message with JSON metadata."""
        meta = json.dumps({"status": "ok", "size": len(data)}).encode()
        socket.send_multipart([identity, b"", meta, data])

    def _send_error(self, socket: zmq.Socket, identity: bytes, message: str) -> None:
        self._send_reply(socket, identity, {"status": "error", "message": message})

    def _log_stats(self) -> None:
        elapsed = time.time() - self._stats["start_time"]
        mb_sent = self._stats["bytes_sent"] / (1024 * 1024)
        rate = mb_sent / max(elapsed, 0.001)
        logger.info(
            "Stats: %d samples served, %.1f MB sent, %.2f MB/s avg over %.0fs",
            self._stats["samples_served"],
            mb_sent,
            rate,
            elapsed,
        )
