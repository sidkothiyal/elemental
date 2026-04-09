"""Unit tests for RemoteFileSystem."""
# pylint: disable=missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,protected-access

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dataset.remote_fs import RemoteFileSystem


@pytest.mark.unit
class TestRemoteFileSystemLocal:
    def test_local_get_local_path_returns_relative(self):
        """For local protocol, get_local_path returns Path(path) as-is."""
        with tempfile.TemporaryDirectory() as tmp:
            rfs = RemoteFileSystem(
                protocol="local",
                remote_root=tmp,
                cache_dir=str(Path(tmp) / "cache"),
            )
            result = rfs.get_local_path("some/file.txt")
            assert result == Path("some/file.txt")

    def test_local_exists_with_absolute_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            test_file = Path(tmp) / "exists.txt"
            test_file.touch()
            rfs = RemoteFileSystem(
                protocol="local", remote_root=tmp, cache_dir=str(Path(tmp) / "c")
            )
            assert rfs.exists(str(test_file)) is True
            assert rfs.exists(str(Path(tmp) / "missing.txt")) is False

    def test_cache_dir_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "new_cache"
            RemoteFileSystem(protocol="local", cache_dir=str(cache_dir))
            assert cache_dir.is_dir()


@pytest.mark.unit
class TestRemoteFileSystemSSH:
    @patch("dataset.remote_fs.fsspec")
    def test_ssh_get_fs_creates_filesystem(self, mock_fsspec):
        mock_fs = MagicMock()
        mock_fsspec.filesystem.return_value = mock_fs

        with tempfile.TemporaryDirectory() as tmp:
            rfs = RemoteFileSystem(
                protocol="ssh",
                remote_root="user@host:/data",
                cache_dir=str(Path(tmp) / "cache"),
            )
            fs = rfs._get_fs()
            assert fs is not None
