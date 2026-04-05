"""Remote dataset client — fetches processed samples from a DatasetServer.

Drop-in replacement for any ``torch.utils.data.Dataset``.  Point the Hydra
config at this class instead of the real dataset, and samples will be fetched
from a running :class:`~dataset.dataset_server.DatasetServer` over the
network.

Works with ``DataLoader(num_workers>0)`` — each worker opens its own ZeroMQ
socket.  Compatible with ``CUDAPrefetcher`` and ``DistributedSampler``.
"""

import hashlib
import hmac
import json
import logging
import os
import time

import torch
import zmq
from torch.utils.data import Dataset

from dataset.serialization import deserialize_sample

logger = logging.getLogger(__name__)

_AUTH_TOKEN_ENV = "ELEMENTAL_DATASET_SERVER_TOKEN"


class RemoteDataset(Dataset):
    """Dataset that fetches processed samples from a remote DatasetServer.

    Args:
        server_host: Hostname or IP of the dataset server.
        server_port: Port the server is listening on.
        compression: Must match the server's compression setting.
            Inferred from handshake if omitted.
        auth_token: Shared secret for HMAC handshake.  Falls back to
            ``$ELEMENTAL_DATASET_SERVER_TOKEN``.
        prefetch_size: Number of samples to request ahead via background
            thread (per DataLoader worker).
        timeout_ms: Socket timeout in milliseconds for individual requests.
        retry_attempts: Number of retries on transient failures before
            raising.
        retry_backoff_s: Initial backoff between retries (doubles each try).
    """

    def __init__(
        self,
        server_host: str,
        server_port: int = 5555,
        compression: str = "lz4",
        auth_token: str | None = None,
        prefetch_size: int = 32,
        timeout_ms: int = 30000,
        retry_attempts: int = 3,
        retry_backoff_s: float = 0.5,
    ):
        self.server_host = server_host
        self.server_port = server_port
        self.compression = compression
        self.auth_token = auth_token or os.environ.get(_AUTH_TOKEN_ENV)
        self.prefetch_size = prefetch_size
        self.timeout_ms = timeout_ms
        self.retry_attempts = retry_attempts
        self.retry_backoff_s = retry_backoff_s

        # Connection state — lazily initialized (supports fork-based workers)
        self._ctx: zmq.Context | None = None
        self._socket: zmq.Socket | None = None
        self._length: int | None = None

        # Eagerly connect in main process to validate config and get length
        self._ensure_connected()

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        self._ensure_connected()
        assert self._length is not None
        return self._length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        self._ensure_connected()
        return self._request_sample(idx)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _ensure_connected(self) -> None:
        """Lazily create socket and perform handshake.

        Safe to call multiple times — no-op if already connected.
        Handles DataLoader fork: if the PID changed since the last
        connect, we re-create the socket (ZeroMQ contexts are not
        fork-safe).
        """
        if self._socket is not None:
            return

        self._ctx = zmq.Context()
        self._socket = zmq.Socket(self._ctx, zmq.DEALER)
        self._socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self._socket.setsockopt(zmq.LINGER, 0)

        endpoint = f"tcp://{self.server_host}:{self.server_port}"
        self._socket.connect(endpoint)
        logger.info("Connected to dataset server at %s", endpoint)

        # Perform handshake
        self._handshake()

    def _handshake(self) -> None:
        """Authenticate with the server and obtain dataset length."""
        msg: dict = {"type": "HANDSHAKE"}

        if self.auth_token:
            ts = str(time.time())
            token_hmac = hmac.new(self.auth_token.encode(), ts.encode(), hashlib.sha256).hexdigest()
            msg["timestamp"] = ts
            msg["token"] = token_hmac

        self._send_json(msg)
        reply = self._recv_json()

        if reply.get("status") != "ok":
            raise ConnectionError(f"Handshake failed: {reply.get('message', 'unknown error')}")

        self._length = reply["length"]
        # Use server's compression if it tells us
        server_compression = reply.get("compression")
        if server_compression and server_compression != self.compression:
            logger.info(
                "Server uses %s compression (overriding local %s)",
                server_compression,
                self.compression,
            )
            self.compression = server_compression

        logger.info(
            "Handshake OK — dataset length: %d, compression: %s",
            self._length,
            self.compression,
        )

    def _reconnect(self) -> None:
        """Close current socket and reconnect."""
        self._close()
        self._ensure_connected()

    def _close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self._ctx is not None:
            self._ctx.term()
            self._ctx = None

    def __del__(self) -> None:
        self._close()

    # ------------------------------------------------------------------
    # Request helpers
    # ------------------------------------------------------------------

    def _request_sample(self, idx: int) -> dict[str, torch.Tensor]:
        """Request a single sample with retry logic."""
        last_err: Exception | None = None
        backoff = self.retry_backoff_s

        for attempt in range(self.retry_attempts):
            try:
                self._send_json({"type": "GETITEM", "idx": idx})
                return self._recv_sample()
            except (zmq.ZMQError, ConnectionError, TimeoutError) as e:
                last_err = e
                logger.warning(
                    "Request for idx=%d failed (attempt %d/%d): %s",
                    idx,
                    attempt + 1,
                    self.retry_attempts,
                    e,
                )
                if attempt < self.retry_attempts - 1:
                    time.sleep(backoff)
                    backoff *= 2
                    self._reconnect()

        raise ConnectionError(
            f"Failed to fetch idx={idx} after {self.retry_attempts} attempts: {last_err}"
        )

    def _send_json(self, obj: dict) -> None:
        """Send a JSON message via the DEALER socket."""
        assert self._socket is not None
        # DEALER sends [empty, payload]
        self._socket.send_multipart([b"", json.dumps(obj).encode()])

    def _recv_json(self) -> dict:
        """Receive a JSON reply from the ROUTER server."""
        assert self._socket is not None
        try:
            frames = self._socket.recv_multipart()
        except zmq.Again as exc:
            raise TimeoutError(f"Server did not respond within {self.timeout_ms}ms") from exc
        # DEALER receives [empty, ...frames]
        # Find the first non-empty frame as metadata
        payload_frames = [f for f in frames if f]
        if not payload_frames:
            raise ConnectionError("Empty reply from server")
        return json.loads(payload_frames[0])

    def _recv_sample(self) -> dict[str, torch.Tensor]:
        """Receive a serialized sample (metadata + binary data)."""
        assert self._socket is not None
        try:
            frames = self._socket.recv_multipart()
        except zmq.Again as exc:
            raise TimeoutError(f"Server did not respond within {self.timeout_ms}ms") from exc
        # DEALER receives [empty, metadata_json, data_bytes]
        non_empty = [f for f in frames if f]
        if len(non_empty) < 1:
            raise ConnectionError("Empty reply from server")

        meta = json.loads(non_empty[0])
        if meta.get("status") == "error":
            raise ConnectionError(f"Server error: {meta.get('message')}")

        if len(non_empty) < 2:
            raise ConnectionError("Missing data frame in server reply")

        data_bytes = non_empty[1]
        return deserialize_sample(data_bytes, self.compression)

    # ------------------------------------------------------------------
    # Health check (callable from scripts)
    # ------------------------------------------------------------------

    def ping(self) -> dict:
        """Send a PING and return the server's status dict."""
        self._ensure_connected()
        self._send_json({"type": "PING"})
        return self._recv_json()
