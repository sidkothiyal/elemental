"""Standalone W&B utility functions.

These helpers are independent of any callback class and can be used
anywhere that needs to interact with W&B artifacts or run-ID files.
"""

import logging
import os

import wandb
from utils import checkpoint as ckpt_utils

logger = logging.getLogger(__name__)

# Filename for storing wandb run id
WANDB_RUN_ID_FILENAME = "wandb_run_id.txt"


def find_wandb_run_id_for_resume(
    results_dir: str,
    experiment_name: str,
    load_checkpoint_path: str | None = None,
) -> str | None:
    """Find wandb run ID from a previous run for resumption.

    If ``load_checkpoint_path`` is ``"latest"``, uses ``find_latest_checkpoint``
    to locate the most recent checkpoint and reads the wandb run ID from its
    directory. If a specific checkpoint path is provided, uses that directory.

    Args:
        results_dir: Root results directory (e.g. ``results/autoencoder/``).
        experiment_name: Current experiment name with datetime stamp.
        load_checkpoint_path: Path to checkpoint file or ``"latest"``.

    Returns:
        The wandb run ID if found, otherwise ``None``.
    """
    checkpoint_dir: str | None = None

    # Resolve checkpoint path to get the directory containing wandb_run_id.txt
    if load_checkpoint_path:
        if str(load_checkpoint_path).lower() == "latest":
            # Use find_latest_checkpoint to locate the most recent checkpoint
            try:
                resolved_path = ckpt_utils.find_latest_checkpoint(
                    results_dir=results_dir,
                    experiment_name=experiment_name,
                )
                checkpoint_dir = os.path.dirname(resolved_path)
                logger.info("Resolved 'latest' checkpoint to directory: %s", checkpoint_dir)
            except FileNotFoundError as e:
                logger.info("Could not find latest checkpoint for wandb resume: %s", e)
                return None
        else:
            # Use the directory of the specified checkpoint path
            checkpoint_dir = os.path.dirname(load_checkpoint_path)
            logger.info("Using checkpoint directory: %s", checkpoint_dir)

    if checkpoint_dir is None:
        logger.info("No checkpoint path provided, cannot find wandb run ID")
        return None

    # Look for wandb run ID file in the checkpoint directory
    run_id_path = os.path.join(checkpoint_dir, WANDB_RUN_ID_FILENAME)
    if not os.path.isfile(run_id_path):
        logger.info("No wandb run ID file found at: %s", run_id_path)
        return None

    # Read the run ID from the file
    try:
        with open(run_id_path, "r", encoding="utf-8") as f:
            run_id = f.read().strip()
        if run_id:
            logger.info("Loaded wandb run ID for resumption: %s", run_id)
            return run_id
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning("Failed to read wandb run ID file: %s", e)

    return None


def upload_artifact(
    name: str,
    artifact_type: str,
    path: str,
    aliases: list[str] | None = None,
    delete_previous: bool = True,
) -> wandb.Artifact:
    """Upload a file or directory as a wandb artifact.

    Uses the current active run (``wandb.run``) to log the artifact.

    Args:
        name: Artifact name (e.g. "my-dataset", "model-ckpt").
        artifact_type: Artifact type (e.g. "dataset", "model").
        path: Local path to a file or directory to upload.
        aliases: Optional list of aliases to assign (e.g. ["latest", "v1"]).
        delete_previous: If ``True`` (default), delete all previous versions
            of this artifact after the new version is uploaded.

    Returns:
        The created and logged ``wandb.Artifact``.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If no active run is available.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")

    if wandb.run is None:
        raise ValueError("No active wandb run (wandb.run is None). Call wandb.init() first.")

    artifact = wandb.Artifact(name=name, type=artifact_type)
    if os.path.isfile(path):
        artifact.add_file(path)
    else:
        artifact.add_dir(path)

    wandb.run.log_artifact(artifact, aliases=aliases)
    logger.info("Uploaded artifact %s (%s) from %s", name, artifact_type, path)

    if delete_previous:
        _delete_previous_artifact_versions(name, artifact_type)

    return artifact


def _delete_previous_artifact_versions(name: str, artifact_type: str) -> None:
    """Delete all versions of an artifact except the latest."""
    try:
        api = wandb.Api()
        artifact_path = f"{wandb.run.entity}/{wandb.run.project}/{name}"
        for version in api.artifacts(type_name=artifact_type, name=artifact_path):
            if "latest" in version.aliases:
                continue
            logger.info("Deleting old artifact version %s", version.name)
            version.delete()
    except Exception:  # pylint: disable=broad-exception-caught
        logger.warning("Failed to delete previous versions of artifact %s", name, exc_info=True)


def download_artifact(
    artifact_ref: str,
    root: str | None = None,
) -> tuple[str, wandb.Artifact]:
    """Download a wandb artifact to disk via the public API.

    Always uses ``wandb.Api()`` so that no active run is required.  This
    makes it safe to call before ``wandb.init()``.

    Args:
        artifact_ref: Fully-qualified artifact reference, e.g.
            ``"entity/project/name:alias"``.
        root: Directory to download into.  If ``None``, uses wandb's
            default artifact cache.

    Returns:
        A ``(download_path, artifact)`` tuple where *download_path* is
        the local directory containing the artifact files and *artifact*
        is the resolved ``wandb.Artifact`` object (useful for
        inspecting metadata such as ``artifact.logged_by()``).
    """
    logger.info("Downloading artifact %s to %s", artifact_ref, root)
    api = wandb.Api()
    artifact = api.artifact(artifact_ref)
    download_path = artifact.download(root=root)
    logger.info("Downloaded artifact %s to %s", artifact_ref, download_path)
    return download_path, artifact
