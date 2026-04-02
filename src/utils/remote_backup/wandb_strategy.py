"""WandB artifact-based remote backup strategy."""

import glob
import logging
import os
import re
import shutil

from utils.remote_backup.base import RemoteBackupStrategy
from utils.wandb_utils import WANDB_RUN_ID_FILENAME, download_artifact, upload_artifact

logger = logging.getLogger(__name__)

_DATETIME_SUFFIX_RE = re.compile(r"_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")


class WandbBackupStrategy(RemoteBackupStrategy):
    """Upload/download checkpoints as WandB artifacts.

    Artifact naming convention (fully-qualified reference)::

        entity/project/base_experiment_name-last:latest   -> last.pt
        entity/project/base_experiment_name-best:latest   -> best.pt

    ``base_experiment_name`` is the experiment name with the Hydra datetime
    suffix stripped (e.g. ``compression_simple_ae_32x``).

    The wandb run ID is **not** stored as a separate artifact.  On download
    it is recovered via ``artifact.logged_by().id`` from the checkpoint
    artifact itself and written to ``wandb_run_id.txt`` so the existing
    ``WandbCallback`` resume logic picks it up.
    """

    def __init__(
        self,
        entity: str,
        project: str,
        experiment_name: str,
        backup_on: list[str] | None = None,
    ):
        self.entity = entity
        self.project = project
        self.base_name = _DATETIME_SUFFIX_RE.sub("", experiment_name)
        self.backup_on = backup_on or ["last", "best"]

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    def upload(self, local_path: str, checkpoint_name: str) -> None:
        """Upload a checkpoint as a WandB artifact."""
        artifact_name = f"{self.base_name}-{checkpoint_name}"
        upload_artifact(
            name=artifact_name,
            artifact_type="checkpoint",
            path=local_path,
        )
        self._cleanup_artifact_cache()

    @staticmethod
    def _cleanup_artifact_cache() -> None:
        """Clear wandb artifact staging and local cache after upload.

        Wandb accumulates large files in two locations during artifact uploads:
          - staging dir: files buffered before upload completes
          - md5 cache:   content-addressed cache of downloaded/uploaded files

        Both are safe to remove after the upload finishes; wandb will
        re-download artifacts on demand if they are needed again.
        """
        # Clear staging directory (files buffered mid-upload)
        wandb_data_dir = os.environ.get(
            "WANDB_DATA_DIR",
            os.path.join(os.path.expanduser("~"), ".local", "share", "wandb"),
        )
        staging_dir = os.path.join(wandb_data_dir, "artifacts", "staging")
        if os.path.isdir(staging_dir):
            for entry in glob.glob(os.path.join(staging_dir, "*")):
                try:
                    if os.path.isdir(entry):
                        shutil.rmtree(entry)
                    else:
                        os.remove(entry)
                except Exception:  # pylint: disable=broad-exception-caught
                    logger.debug("Could not remove staging entry %s", entry, exc_info=True)
            logger.info("Cleared wandb artifact staging dir: %s", staging_dir)

        # Clear md5 object cache
        wandb_cache_dir = os.environ.get(
            "WANDB_CACHE_DIR",
            os.path.join(os.path.expanduser("~"), ".cache", "wandb"),
        )
        md5_cache_dir = os.path.join(wandb_cache_dir, "artifacts", "obj", "md5")
        if os.path.isdir(md5_cache_dir):
            try:
                shutil.rmtree(md5_cache_dir)
                os.makedirs(md5_cache_dir, exist_ok=True)
            except Exception:  # pylint: disable=broad-exception-caught
                logger.debug("Could not clear md5 cache dir %s", md5_cache_dir, exc_info=True)
            logger.info("Cleared wandb artifact md5 cache: %s", md5_cache_dir)

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download_latest(self, local_dir: str) -> str | None:
        """Download the latest checkpoint from WandB and recover the run ID.

        Files are placed directly into *local_dir* so that the existing
        ``find_latest_checkpoint`` and ``_find_wandb_run_id_for_resume``
        helpers discover them automatically.
        """
        os.makedirs(local_dir, exist_ok=True)

        checkpoint_path: str | None = None
        for name_suffix in self.backup_on:
            artifact_ref = f"{self.entity}/{self.project}/{self.base_name}-{name_suffix}:latest"
            try:
                download_dir, artifact = download_artifact(artifact_ref, root=local_dir)
                candidate = os.path.join(download_dir, f"{name_suffix}.pt")
                if not os.path.isfile(candidate):
                    for f in os.listdir(download_dir):
                        if f.endswith(".pt"):
                            candidate = os.path.join(download_dir, f)
                            break
                if os.path.isfile(candidate):
                    dest = os.path.join(local_dir, f"{name_suffix}.pt")
                    if os.path.abspath(candidate) != os.path.abspath(dest):
                        os.replace(candidate, dest)
                    checkpoint_path = dest
                    logger.info("Downloaded remote checkpoint to %s", checkpoint_path)

                    # Recover the wandb run ID from the artifact's source run
                    self._recover_run_id(artifact, local_dir)
                    break
            except Exception:  # pylint: disable=broad-exception-caught
                logger.debug("Artifact %s not found, trying next", artifact_ref, exc_info=True)

        if checkpoint_path is None:
            logger.info("No remote checkpoint found for %s", self.base_name)

        return checkpoint_path

    @staticmethod
    def _recover_run_id(artifact, local_dir: str) -> None:
        """Write ``wandb_run_id.txt`` by inspecting which run logged *artifact*."""
        try:
            source_run = artifact.logged_by()
            if source_run is None:
                logger.debug("artifact.logged_by() returned None, skipping run ID recovery")
                return
            run_id = source_run.id
            dest = os.path.join(local_dir, WANDB_RUN_ID_FILENAME)
            with open(dest, "w", encoding="utf-8") as f:
                f.write(run_id)
            logger.info("Recovered wandb run ID %s -> %s", run_id, dest)
        except Exception:  # pylint: disable=broad-exception-caught
            logger.debug("Failed to recover wandb run ID from artifact", exc_info=True)
