"""SSH/SCP-based remote backup strategy."""

import logging
import os
import re
import shutil
import subprocess

from utils.remote_backup.base import RemoteBackupStrategy

logger = logging.getLogger(__name__)

_DATETIME_SUFFIX_RE = re.compile(r"_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")

WANDB_RUN_ID_FILENAME = "wandb_run_id.txt"

# Files to sync alongside the checkpoint
_METADATA_FILES = [WANDB_RUN_ID_FILENAME]


class SSHBackupStrategy(RemoteBackupStrategy):
    """Upload/download checkpoints over SSH using ``scp`` (or ``rsync``).

    Remote directory layout::

        remote_dir/
          base_experiment_name/
            last.pt
            best.pt
            wandb_run_id.txt
    """

    def __init__(
        self,
        host: str,
        username: str,
        remote_dir: str,
        experiment_name: str,
        ssh_key_path: str | None = None,
        port: int = 22,
        backup_on: list[str] | None = None,
    ):
        self.host = host
        self.username = username
        self.remote_dir = remote_dir
        self.ssh_key_path = ssh_key_path
        self.port = port
        self.base_name = _DATETIME_SUFFIX_RE.sub("", experiment_name)
        self.backup_on = backup_on or ["last", "best"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def _remote_base(self) -> str:
        """Remote directory for this experiment (without host prefix)."""
        return f"{self.remote_dir}/{self.base_name}"

    def _ssh_opts(self) -> list[str]:
        opts: list[str] = []
        if self.ssh_key_path:
            opts += ["-i", self.ssh_key_path]
        if self.port != 22:
            opts += ["-P", str(self.port)]
        return opts

    def _ssh_opts_for_ssh_cmd(self) -> list[str]:
        """SSH options using ``-p`` (lowercase) for the ``ssh`` command."""
        opts: list[str] = []
        if self.ssh_key_path:
            opts += ["-i", self.ssh_key_path]
        if self.port != 22:
            opts += ["-p", str(self.port)]
        return opts

    def _remote_target(self) -> str:
        return f"{self.username}@{self.host}"

    def _use_rsync(self) -> bool:
        return shutil.which("rsync") is not None

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    def upload(
        self, local_path: str, checkpoint_name: str  # pylint: disable=unused-argument
    ) -> None:
        """Upload checkpoint (and metadata files) to the remote host."""
        remote_dest = self._remote_base

        # Ensure remote directory exists
        ssh_cmd = [
            "ssh",
            *self._ssh_opts_for_ssh_cmd(),
            self._remote_target(),
            f"mkdir -p {remote_dest}",
        ]
        subprocess.run(ssh_cmd, check=True, capture_output=True)

        self._put_file(local_path, f"{remote_dest}/{os.path.basename(local_path)}")

        # Also upload metadata files from the same directory
        local_dir = os.path.dirname(local_path)
        for fname in _METADATA_FILES:
            fpath = os.path.join(local_dir, fname)
            if os.path.isfile(fpath):
                self._put_file(fpath, f"{remote_dest}/{fname}")

    def _put_file(self, local_path: str, remote_path: str) -> None:
        target = f"{self._remote_target()}:{remote_path}"
        if self._use_rsync():
            cmd = [
                "rsync",
                "-az",
                "-e",
                " ".join(["ssh", *self._ssh_opts_for_ssh_cmd()]),
                local_path,
                target,
            ]
        else:
            cmd = ["scp", *self._ssh_opts(), local_path, target]
        logger.info("Uploading %s -> %s", local_path, target)
        subprocess.run(cmd, check=True, capture_output=True)

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download_latest(self, local_dir: str) -> str | None:
        """Download the latest checkpoint (and metadata) from the remote host."""
        os.makedirs(local_dir, exist_ok=True)
        remote_src = self._remote_base

        # Try checkpoint files in priority order
        for name_suffix in self.backup_on:
            remote_file = f"{remote_src}/{name_suffix}.pt"
            local_dest = os.path.join(local_dir, f"{name_suffix}.pt")
            try:
                self._get_file(remote_file, local_dest)
                logger.info("Downloaded remote checkpoint to %s", local_dest)

                # Also download metadata files
                for fname in _METADATA_FILES:
                    try:
                        self._get_file(
                            f"{remote_src}/{fname}",
                            os.path.join(local_dir, fname),
                        )
                    except subprocess.CalledProcessError:
                        logger.debug("Metadata file %s not found on remote", fname)

                return local_dest
            except subprocess.CalledProcessError:
                logger.debug("Remote file %s not found, trying next", remote_file)

        logger.info("No remote checkpoint found for %s", self.base_name)
        return None

    def _get_file(self, remote_path: str, local_path: str) -> None:
        source = f"{self._remote_target()}:{remote_path}"
        if self._use_rsync():
            cmd = [
                "rsync",
                "-az",
                "-e",
                " ".join(["ssh", *self._ssh_opts_for_ssh_cmd()]),
                source,
                local_path,
            ]
        else:
            cmd = ["scp", *self._ssh_opts(), source, local_path]
        logger.info("Downloading %s -> %s", source, local_path)
        subprocess.run(cmd, check=True, capture_output=True)
