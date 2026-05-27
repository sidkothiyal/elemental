"""Abstract base class for remote backup strategies."""

from abc import ABC, abstractmethod


class RemoteBackupStrategy(ABC):
    """Base class for remote checkpoint backup strategies.

    Subclasses implement upload/download logic for a specific backend
    (e.g. WandB artifacts, SSH/SCP).
    """

    backup_on: list[str]

    def should_backup(self, checkpoint_name: str) -> bool:
        """Return whether *checkpoint_name* should be backed up remotely."""
        return checkpoint_name in self.backup_on

    @abstractmethod
    def upload(self, local_path: str, checkpoint_name: str) -> None:
        """Upload a checkpoint file to remote storage.

        Args:
            local_path: Absolute path to the local checkpoint file.
            checkpoint_name: Logical name without extension (e.g. "last", "best").
        """

    @abstractmethod
    def download_latest(self, local_dir: str) -> str | None:
        """Download the latest checkpoint (and associated metadata) into *local_dir*.

        Returns:
            Path to the downloaded checkpoint file, or ``None`` if nothing
            was found on the remote.
        """
