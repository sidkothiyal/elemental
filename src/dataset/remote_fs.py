"""Remote filesystem abstraction with local caching and prefetching.

Wraps ``fsspec`` to provide transparent access to remote files (SSH, S3,
HTTP) with two operating modes:

* **stream** — per-file on-the-fly caching.  Files are downloaded lazily
  on first access and cached locally.  A background prefetcher can
  download files ahead of the training iteration.
* **sync** — bulk directory sync via ``rsync``/``scp`` or fsspec.
  Blocks until the entire remote directory is mirrored locally, then
  the dataset reads from the local copy.

Both modes are transparent to dataset code: ``resolve_path()`` always
returns a local ``pathlib.Path``.
"""

import logging
import os
import pathlib
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import fsspec
import tqdm

logger = logging.getLogger(__name__)


class RemoteFileSystem:
    """Unified remote file access with local caching.

    Args:
        protocol: Storage protocol — ``"ssh"``, ``"s3"``, ``"http"``,
            or ``"local"`` (passthrough).
        remote_root: Root path on the remote system.  Format depends on
            protocol:
            - ssh: ``"user@host:/path/to/dir"``
            - s3:  ``"s3://bucket/prefix"``
            - http: ``"https://example.com/data"``
        cache_dir: Local directory for cached files.
        mode: ``"stream"`` (default) or ``"sync"``.
        prefetch_workers: Thread pool size for background prefetching.
        ssh_key_path: Path to SSH private key (SSH protocol only).
        port: SSH port (SSH protocol only, default 22).
        **storage_options: Extra kwargs forwarded to the fsspec filesystem
            (e.g., ``anon=True`` for public S3 buckets).
    """

    def __init__(
        self,
        protocol: str = "local",
        remote_root: str = "",
        cache_dir: str = "/tmp/elemental_cache",
        mode: str = "stream",
        prefetch_workers: int = 4,
        ssh_key_path: str | None = None,
        port: int = 22,
        **storage_options,
    ):
        self.protocol = protocol.lower()
        self.remote_root = remote_root.rstrip("/")
        self.cache_dir = pathlib.Path(cache_dir)
        self.mode = mode.lower()
        self.prefetch_workers = prefetch_workers
        self.ssh_key_path = ssh_key_path
        self.port = port
        self.storage_options = storage_options

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Lazily initialized fsspec filesystem
        self._fs = None
        self._fs_lock = threading.Lock()

        # Track ongoing prefetches
        self._prefetch_executor: ThreadPoolExecutor | None = None
        self._prefetch_futures: list = []

        # Sync mode: do the sync immediately
        if self.mode == "sync" and self.protocol != "local":
            self.sync_directory()

    @property
    def is_remote(self) -> bool:
        """Whether this filesystem points to a remote location."""
        return self.protocol != "local"

    # ------------------------------------------------------------------
    # fsspec filesystem (lazy init)
    # ------------------------------------------------------------------

    def _get_fs(self):
        """Lazily create the fsspec filesystem instance."""
        if self._fs is not None:
            return self._fs

        with self._fs_lock:
            if self._fs is not None:
                return self._fs

            if self.protocol == "ssh":
                # Parse user@host:/path format
                host_info, _ = self._parse_ssh_remote(self.remote_root)
                user, host = host_info.split("@") if "@" in host_info else (None, host_info)

                ssh_opts = {}
                if user:
                    ssh_opts["username"] = user
                if self.ssh_key_path:
                    ssh_opts["key_filename"] = os.path.expanduser(self.ssh_key_path)
                if self.port != 22:
                    ssh_opts["port"] = self.port
                ssh_opts.update(self.storage_options)

                self._fs = fsspec.filesystem("ssh", host=host, **ssh_opts)

            elif self.protocol == "s3":
                self._fs = fsspec.filesystem("s3", **self.storage_options)

            elif self.protocol in ("http", "https"):
                self._fs = fsspec.filesystem("http", **self.storage_options)

            elif self.protocol == "local":
                self._fs = fsspec.filesystem("file")

            else:
                # Try fsspec generic — it may support the protocol
                self._fs = fsspec.filesystem(self.protocol, **self.storage_options)

            return self._fs

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def _remote_path(self, relative: str) -> str:
        """Convert a relative path to the full remote path for fsspec."""
        if self.protocol == "ssh":
            _, base_path = self._parse_ssh_remote(self.remote_root)
            return f"{base_path}/{relative}".replace("//", "/")
        if self.protocol in ("s3",):
            # s3://bucket/prefix/relative
            return f"{self.remote_root}/{relative}"
        return f"{self.remote_root}/{relative}"

    def _cache_path(self, relative: str) -> pathlib.Path:
        """Local cache path for a given relative path."""
        # Use a flat cache with hashed names to avoid nested directory issues
        safe_name = relative.replace("/", "_").replace("\\", "_")
        return self.cache_dir / safe_name

    def _cache_path_structured(self, relative: str) -> pathlib.Path:
        """Local cache path preserving directory structure."""
        path = self.cache_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _parse_ssh_remote(remote_root: str) -> tuple[str, str]:
        """Parse ``user@host:/path`` into ``(user@host, /path)``."""
        if ":" in remote_root:
            host_part, path_part = remote_root.split(":", 1)
            return host_part, path_part
        return remote_root, ""

    # ------------------------------------------------------------------
    # File access (streaming mode)
    # ------------------------------------------------------------------

    def get_local_path(self, path: str | pathlib.Path) -> pathlib.Path:
        """Download a remote file to the local cache if not already cached.

        Args:
            path: Path relative to ``remote_root``, or an absolute path
                that will be used as-is for local protocol.

        Returns:
            Local ``pathlib.Path`` to the (cached) file.
        """
        path_str = str(path)

        if self.protocol == "local":
            return pathlib.Path(path_str)

        # If sync mode completed, files should already be local
        if self.mode == "sync":
            cached = self._cache_path_structured(path_str)
            if cached.exists():
                return cached
            # Fall through to download if somehow missing

        cached = self._cache_path_structured(path_str)
        if cached.exists():
            return cached

        # Download
        remote = self._remote_path(path_str)
        fs = self._get_fs()
        try:
            fs.get(remote, str(cached))
            logger.debug("Downloaded %s -> %s", remote, cached)
        except Exception as e:
            logger.error("Failed to download %s: %s", remote, e)
            raise

        return cached

    def open(self, path: str | pathlib.Path, mode: str = "rb"):
        """Open a file, downloading from remote if needed.

        For streaming mode, downloads to cache first then opens locally
        (better for random access patterns like numpy.load).
        """
        local = self.get_local_path(str(path))
        return open(local, mode, encoding="utf-8" if "b" not in mode else None)  # noqa: WPS515

    def exists(self, path: str) -> bool:
        """Check if a remote file exists."""
        if self.protocol == "local":
            return pathlib.Path(path).exists()
        remote = self._remote_path(path)
        fs = self._get_fs()
        return fs.exists(remote)

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    def glob(self, pattern: str) -> list[str]:
        """Glob for files on the remote (or local) filesystem.

        Returns paths relative to ``remote_root``.

        .. note::
            Remote glob can be slow.  Prefer structure-aware discovery
            methods in :class:`~dataset.base_dataset.BaseDataset` when
            the dataset structure is known.
        """
        if self.protocol == "local":
            root = pathlib.Path(self.remote_root) if self.remote_root else pathlib.Path(".")
            return [str(p.relative_to(root)) for p in sorted(root.glob(pattern))]

        # For sync mode, glob the local cache
        if self.mode == "sync" and self.cache_dir.exists():
            return [
                str(p.relative_to(self.cache_dir)) for p in sorted(self.cache_dir.glob(pattern))
            ]

        # Remote glob via fsspec
        fs = self._get_fs()
        base = self._remote_path("")
        full_pattern = f"{base}/{pattern}".replace("//", "/")
        results = fs.glob(full_pattern)

        # Return paths relative to remote_root
        relative = []
        for r in sorted(results):
            r = str(r)
            if r.startswith(base):
                r = r[len(base) :].lstrip("/")
            relative.append(r)
        return relative

    def list_dir(self, path: str = "") -> list[str]:
        """List entries in a remote directory.

        Returns names (not full paths) of entries in *path* relative to
        ``remote_root``.
        """
        if self.protocol == "local":
            root = pathlib.Path(self.remote_root) / path if self.remote_root else pathlib.Path(path)
            return sorted(e.name for e in root.iterdir())

        # Sync mode: list from cache
        if self.mode == "sync":
            local = self.cache_dir / path
            if local.exists():
                return sorted(e.name for e in local.iterdir())

        fs = self._get_fs()
        remote = self._remote_path(path)
        entries = fs.ls(remote, detail=False)
        return sorted(e.split("/")[-1] for e in entries)

    # ------------------------------------------------------------------
    # Prefetching (streaming mode)
    # ------------------------------------------------------------------

    def prefetch(self, paths: list[str], num_workers: int | None = None) -> None:
        """Download files in the background using a thread pool.

        Non-blocking — returns immediately.  Already-cached files are
        skipped.  Use :meth:`wait_prefetch` to block until complete.

        Args:
            paths: Relative paths to prefetch.
            num_workers: Override the default prefetch_workers.
        """
        if self.protocol == "local" or self.mode == "sync":
            return  # Nothing to prefetch

        workers = num_workers or self.prefetch_workers

        # Filter out already-cached files
        to_fetch = [p for p in paths if not self._cache_path_structured(p).exists()]
        if not to_fetch:
            logger.info("All %d files already cached.", len(paths))
            return

        logger.info(
            "Prefetching %d/%d files (%d workers)...",
            len(to_fetch),
            len(paths),
            workers,
        )

        self._prefetch_executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="prefetch"
        )

        progress = tqdm.tqdm(
            total=len(to_fetch),
            desc="Prefetching",
            unit="file",
            leave=False,
        )

        def _download_one(p: str) -> str | None:
            try:
                self.get_local_path(p)
                progress.update(1)
                return None
            except Exception as e:
                progress.update(1)
                return f"{p}: {e}"

        self._prefetch_futures = [
            self._prefetch_executor.submit(_download_one, p) for p in to_fetch
        ]

        # Start a monitor thread to log completion
        def _monitor():
            errors = []
            for fut in as_completed(self._prefetch_futures):
                err = fut.result()
                if err:
                    errors.append(err)
            progress.close()
            if errors:
                logger.warning(
                    "Prefetch completed with %d errors:\n  %s",
                    len(errors),
                    "\n  ".join(errors[:10]),
                )
            else:
                logger.info("Prefetch complete — all %d files cached.", len(to_fetch))

        threading.Thread(target=_monitor, daemon=True, name="prefetch-monitor").start()

    def wait_prefetch(self) -> None:
        """Block until all background prefetch tasks complete."""
        if self._prefetch_executor is not None:
            self._prefetch_executor.shutdown(wait=True)
            self._prefetch_executor = None

    # ------------------------------------------------------------------
    # Directory sync (sync mode)
    # ------------------------------------------------------------------

    def sync_directory(self, remote_dir: str | None = None, local_dir: str | None = None) -> None:
        """Synchronize a remote directory to a local directory.

        Uses ``rsync`` over SSH when available, falls back to ``scp -r``,
        then to fsspec-based copy.

        Args:
            remote_dir: Override remote source (default: ``remote_root``).
            local_dir: Override local destination (default: ``cache_dir``).
        """
        remote = remote_dir or self.remote_root
        local = pathlib.Path(local_dir or str(self.cache_dir))
        local.mkdir(parents=True, exist_ok=True)

        if self.protocol == "ssh":
            self._sync_ssh(remote, local)
        elif self.protocol == "s3":
            self._sync_s3(remote, local)
        else:
            self._sync_fsspec(remote, local)

    def _sync_ssh(self, remote: str, local: pathlib.Path) -> None:
        """Sync via rsync or scp over SSH."""
        ssh_opts = []
        if self.ssh_key_path:
            ssh_opts += ["-i", os.path.expanduser(self.ssh_key_path)]
        if self.port != 22:
            ssh_opts += ["-p", str(self.port)]

        # Ensure remote ends with / for rsync directory sync
        if not remote.endswith("/"):
            remote += "/"

        if shutil.which("rsync"):
            ssh_cmd = " ".join(["ssh"] + ssh_opts)
            cmd = [
                "rsync",
                "-az",
                "--info=progress2",
                "-e",
                ssh_cmd,
                remote,
                str(local) + "/",
            ]
            logger.info("Syncing via rsync: %s -> %s", remote, local)
        else:
            # Fall back to scp
            scp_opts = []
            if self.ssh_key_path:
                scp_opts += ["-i", os.path.expanduser(self.ssh_key_path)]
            if self.port != 22:
                scp_opts += ["-P", str(self.port)]
            cmd = ["scp", "-r"] + scp_opts + [remote, str(local)]
            logger.info("Syncing via scp: %s -> %s", remote, local)

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            logger.error("Sync failed: %s", result.stderr)
            raise RuntimeError(f"Directory sync failed: {result.stderr}")
        logger.info("Sync complete: %s", local)

    def _sync_s3(self, remote: str, local: pathlib.Path) -> None:
        """Sync from S3 using aws cli or fsspec."""
        if shutil.which("aws"):
            cmd = ["aws", "s3", "sync", remote, str(local)]
            logger.info("Syncing via aws s3 sync: %s -> %s", remote, local)
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                raise RuntimeError(f"S3 sync failed: {result.stderr}")
            logger.info("S3 sync complete: %s", local)
        else:
            self._sync_fsspec(remote, local)

    def _sync_fsspec(self, remote: str, local: pathlib.Path) -> None:
        """Generic sync via fsspec (downloads all files)."""
        fs = self._get_fs()
        base = self._remote_path("")
        files = fs.ls(base, detail=False) if not remote else fs.find(remote)

        logger.info("Syncing %d files via fsspec: %s -> %s", len(files), remote, local)
        for f in tqdm.tqdm(files, desc="Downloading", unit="file"):
            rel = str(f)
            if rel.startswith(base):
                rel = rel[len(base) :].lstrip("/")
            dest = local / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                fs.get(f, str(dest))

        logger.info("Sync complete: %d files in %s", len(files), local)
