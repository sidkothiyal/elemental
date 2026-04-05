"""Base dataset class with remote file access and smart file discovery.

All Elemental datasets should inherit from :class:`BaseDataset` instead of
``torch.utils.data.Dataset`` directly.  When no ``remote_config`` is
supplied, the class is a transparent pass-through with zero overhead.

File discovery
--------------
Remote directory listings (glob) are expensive.  :class:`BaseDataset`
provides several discovery strategies that subclasses can choose from
based on their dataset structure:

* :meth:`list_files` — glob-based (local: fast, remote: fallback).
* :meth:`discover_files_from_range` — numbered files (zero network calls).
* :meth:`discover_files_from_manifest` — JSON/text manifest file
  (single file download).
* :meth:`discover_files_from_structure` — known directory layout with
  provided subdirectory names.

File access
-----------
In ``__getitem__``, call :meth:`resolve_path` to transparently download
and cache remote files.  Call :meth:`start_prefetch` after discovery to
download files in the background ahead of training.
"""

import json
import logging
import pathlib

from torch.utils.data import Dataset

from dataset.remote_fs import RemoteFileSystem

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """Base dataset with optional remote file access.

    Args:
        remote_config: Optional dict with remote storage settings.
            When *None*, the dataset behaves identically to a plain
            ``torch.utils.data.Dataset``.

            Supported keys:

            * ``protocol`` (str): ``"ssh"``, ``"s3"``, ``"http"``, ``"local"``.
            * ``remote_root`` (str): Root path on the remote system.
            * ``cache_dir`` (str): Local cache directory.
            * ``mode`` (str): ``"stream"`` (default) or ``"sync"``.
            * ``prefetch_workers`` (int): Thread pool size (default 4).
            * ``ssh_key_path`` (str): SSH key path (SSH only).
            * ``port`` (int): SSH port (SSH only, default 22).
            * Additional keys are forwarded to fsspec as storage_options.
    """

    def __init__(self, remote_config: dict | None = None):
        super().__init__()
        self._remote_fs = None

        if remote_config is not None:
            # Convert DictConfig to plain dict if needed (Hydra)
            if hasattr(remote_config, "__iter__") and not isinstance(remote_config, dict):
                remote_config = dict(remote_config)

            # Extract known keys, pass rest as storage_options
            known_keys = {
                "protocol",
                "remote_root",
                "cache_dir",
                "mode",
                "prefetch_workers",
                "ssh_key_path",
                "port",
            }
            fs_kwargs = {k: v for k, v in remote_config.items() if k in known_keys}
            storage_options = {k: v for k, v in remote_config.items() if k not in known_keys}
            self._remote_fs = RemoteFileSystem(**fs_kwargs, **storage_options)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_remote(self) -> bool:
        """Whether this dataset uses a remote filesystem."""
        return self._remote_fs is not None and self._remote_fs.is_remote

    @property
    def remote_mode(self) -> str | None:
        """Current remote mode (``"stream"``, ``"sync"``, or *None*)."""
        if self._remote_fs is None:
            return None
        return self._remote_fs.mode

    # ------------------------------------------------------------------
    # File discovery strategies
    # ------------------------------------------------------------------

    def list_files(self, directory: str | pathlib.Path, pattern: str) -> list[pathlib.Path]:
        """List files matching *pattern* in *directory*.

        Works for both local and remote filesystems.  For local, uses
        ``pathlib.Path.glob`` (fast).  For remote, uses fsspec glob
        (slower — prefer structure-aware methods when possible).

        Args:
            directory: Directory to search in.
            pattern: Glob pattern (e.g., ``"*.png"``).

        Returns:
            Sorted list of paths.  For remote stream mode, these are
            relative paths that must be resolved via :meth:`resolve_path`
            before use.
        """
        if self._remote_fs is not None and self._remote_fs.is_remote:
            if self._remote_fs.mode == "sync":
                # After sync, files are in cache_dir
                cache_dir = self._remote_fs.cache_dir
                results = sorted(cache_dir.glob(pattern))
                return results
            # Stream mode — remote glob
            rel_paths = self._remote_fs.glob(pattern)
            return [pathlib.Path(p) for p in rel_paths]
        # Local — standard glob
        directory = pathlib.Path(directory)
        return sorted(directory.glob(pattern))

    def discover_files_from_range(
        self,
        directory: str | pathlib.Path,
        template: str,
        start: int,
        end: int,
        ext: str,
    ) -> list[pathlib.Path]:
        """Generate file paths from a numbered range.

        Zero network calls — constructs paths purely from the naming
        convention.  Ideal for datasets with predictable numbering
        (e.g., DIV2K: ``0001.png`` to ``0900.png``).

        Args:
            directory: Base directory containing the files.
            template: Printf-style template for the numeric part
                (e.g., ``"%04d"`` for zero-padded 4-digit numbers).
            start: First number (inclusive).
            end: Last number (inclusive).
            ext: File extension including dot (e.g., ``".png"``).

        Returns:
            List of ``pathlib.Path`` objects.
        """
        directory = pathlib.Path(directory)

        if self.is_remote and self.remote_mode == "sync":
            assert self._remote_fs is not None
            directory = pathlib.Path(self._remote_fs.cache_dir)

        return [directory / f"{template % i}{ext}" for i in range(start, end + 1)]

    def discover_files_from_manifest(
        self,
        manifest_path: str | pathlib.Path,
        root_dir: str | pathlib.Path | None = None,
    ) -> list[pathlib.Path]:
        """Load file list from a manifest file.

        The manifest is a JSON file containing a list of relative paths,
        or a plain text file with one path per line.  For remote
        datasets, only the manifest file itself is downloaded — avoiding
        a full directory listing.

        Args:
            manifest_path: Path to the manifest file (relative to
                remote_root for remote datasets).
            root_dir: Optional root directory to prepend to each path.
                Defaults to the manifest's parent directory.

        Returns:
            List of ``pathlib.Path`` objects.
        """
        # Download manifest if remote
        local_manifest = self.resolve_path(manifest_path)
        content = local_manifest.read_text(encoding="utf-8")

        # Try JSON first, fall back to line-per-file
        try:
            entries = json.loads(content)
            if not isinstance(entries, list):
                raise ValueError("Manifest JSON must be a list of paths")
        except json.JSONDecodeError:
            entries = [line.strip() for line in content.splitlines() if line.strip()]

        if root_dir is None:
            root_dir = pathlib.Path(manifest_path).parent

        root_dir = pathlib.Path(root_dir)

        if self.is_remote and self.remote_mode == "sync":
            assert self._remote_fs is not None
            root_dir = self._remote_fs.cache_dir

        return [root_dir / entry for entry in entries]

    def discover_files_from_structure(
        self,
        root: str | pathlib.Path,
        subdirs: list[str],
        pattern: str,
    ) -> list[pathlib.Path]:
        """Build file paths from a known directory structure.

        When the list of subdirectories is known in advance (e.g.,
        ImageNet class names), this avoids listing the parent directory.
        Only globs within each subdirectory (which is typically small
        and fast, even remotely).

        For truly known structures (where even the files are
        predictable), prefer :meth:`discover_files_from_range`.

        Args:
            root: Root directory.
            subdirs: Known subdirectory names.
            pattern: File glob pattern within each subdirectory.

        Returns:
            Flat list of all matching paths across all subdirectories.
        """
        root = pathlib.Path(root)

        if self.is_remote and self.remote_mode == "sync":
            assert self._remote_fs is not None
            root = pathlib.Path(self._remote_fs.cache_dir)

        all_files: list[pathlib.Path] = []
        for subdir in subdirs:
            subdir_path = root / subdir
            if self.is_remote and self.remote_mode == "stream":
                # Remote glob within subdirectory
                sub_pattern = f"{subdir}/{pattern}"
                assert self._remote_fs is not None
                rel_paths = self._remote_fs.glob(sub_pattern)
                all_files.extend(pathlib.Path(p) for p in rel_paths)
            else:
                # Local glob
                all_files.extend(sorted(subdir_path.glob(pattern)))

        return all_files

    # ------------------------------------------------------------------
    # File access
    # ------------------------------------------------------------------

    def resolve_path(self, path: str | pathlib.Path) -> pathlib.Path:
        """Resolve a file path, downloading from remote if needed.

        For local datasets, returns the path as-is.  For remote stream
        mode, downloads the file to the local cache on first access and
        returns the cached path.  For sync mode, returns the cached path
        directly (sync already completed in ``__init__``).

        This is the primary method used in ``__getitem__`` to transparently
        handle local and remote files.

        Args:
            path: File path (relative for remote, absolute for local).

        Returns:
            Local ``pathlib.Path`` to the file.
        """
        if self._remote_fs is None or not self._remote_fs.is_remote:
            return pathlib.Path(path)

        return self._remote_fs.get_local_path(str(path))

    def open_file(self, path: str | pathlib.Path, mode: str = "rb"):
        """Open a file, transparently handling remote paths.

        Downloads the file to cache if remote, then opens the local copy.

        Args:
            path: File path.
            mode: File open mode (default ``"rb"``).

        Returns:
            File handle.
        """
        local = self.resolve_path(path)
        return open(local, mode, encoding="utf-8" if "b" not in mode else None)  # noqa: WPS515

    # ------------------------------------------------------------------
    # Prefetching
    # ------------------------------------------------------------------

    def start_prefetch(
        self, paths: list[str | pathlib.Path], num_workers: int | None = None
    ) -> None:
        """Start background downloading of files.

        No-op when the dataset is local or in sync mode (files already
        present).  Call this after file discovery to download files ahead
        of the first training epoch.

        Args:
            paths: List of file paths to prefetch.
            num_workers: Override default number of download threads.
        """
        if self._remote_fs is None or not self._remote_fs.is_remote:
            return
        if self._remote_fs.mode == "sync":
            return

        str_paths = [str(p) for p in paths]
        self._remote_fs.prefetch(str_paths, num_workers=num_workers)

    def wait_prefetch(self) -> None:
        """Block until all background prefetch operations complete."""
        if self._remote_fs is not None:
            self._remote_fs.wait_prefetch()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_cache_dir(self) -> pathlib.Path | None:
        """Return the local cache directory, or None if not remote."""
        if self._remote_fs is None:
            return None
        return self._remote_fs.cache_dir

    def remote_root_as_local(self) -> pathlib.Path:
        """For sync mode, return the local cache dir as the 'root_dir'.

        This allows datasets to transparently use the cache dir as their
        root directory after a sync operation.
        """
        if self._remote_fs is not None and self._remote_fs.mode == "sync":
            return self._remote_fs.cache_dir
        raise ValueError("remote_root_as_local() only valid in sync mode")
