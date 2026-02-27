"""Wandb callback for Trainer"""

import logging
import os
import random

import numpy as np
import torch
from torch.nn import functional as F

import wandb
from callback import callback
from data_processor.tokenizer import Tokenizer
from utils import checkpoint as ckpt_utils, torch_np_utils

# from transformers import AutoTokenizer


logger = logging.getLogger(__name__)

os.environ["WANDB_SILENT"] = "True"

# Filename for storing wandb run id
WANDB_RUN_ID_FILENAME = "wandb_run_id.txt"


def _find_wandb_run_id_for_resume(
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
    print(f"Downloading artifact {artifact_ref} to {root}")
    api = wandb.Api()
    artifact = api.artifact(artifact_ref)
    download_path = artifact.download(root=root)
    logger.info("Downloaded artifact %s to %s", artifact_ref, download_path)
    return download_path, artifact


class WandbCallback(callback.Callback):
    """Wandb callback for Trainer"""

    def __init__(
        self,
        exp_name: str,
        project: str | None = "elemental",
        entity: str | None = "elemental-ai",
        logging_interval: int = 100,
        num_samples_to_log: int = 1,
        class_colors: dict[int, tuple[int, int, int]] | None = None,
        num_classes: int | None = None,
        mode: str | None = None,
        log_text_data: bool = False,
        tokenizer: str | None = None,  # pylint: disable=unused-argument
        results_dir: str | None = None,
        resume_training: bool = False,
        load_checkpoint_path: str | None = None,
    ):
        """Initialize the callback.

        Args:
            exp_name: The name of the experiment.
            project: The name of the project.
            entity: The name of the entity.
            logging_interval: The logging interval.
            num_samples_to_log: The number of images to log.
            class_colors: The colors for each class for segmentation images.
            num_classes: The number of classes for segmentation images.
            mode: The mode of training.
            log_text_data: Whether to log text data.
            tokenizer: The tokenizer to use for text logging.
            results_dir: Directory where results and checkpoints are saved.
                Required for resume_training to find previous wandb run ID.
            resume_training: Whether to resume training from a checkpoint.
                If True, will attempt to resume the wandb run from the previous
                training session.
            load_checkpoint_path: Path to the checkpoint file or "latest".
                Used to locate the wandb run ID file when resuming.
        """
        logger.info(
            "Initializing WandbCallback with project: %s, entity: %s, exp_name: %s",
            project,
            entity,
            exp_name,
        )
        self._logging_interval = logging_interval
        self._num_samples_to_log = num_samples_to_log
        self._mode = mode
        self._log_text_data = log_text_data
        self._results_dir = results_dir
        self._exp_name = exp_name
        if self._num_samples_to_log != 1:
            raise NotImplementedError(
                f"num_samples_to_log is not 1, but {self._num_samples_to_log}"
            )

        self._class_colors = class_colors
        self._num_classes = num_classes
        if self._num_classes:
            if self._class_colors is None:
                self._class_colors = {
                    i: (
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255),
                    )
                    for i in range(self._num_classes)
                }
        if log_text_data:
            # if tokenizer is None:
            #     raise ValueError("tokenizer is required when log_text_data is True")
            self._log_table = wandb.Table(columns=["step", "phase", "tag", "text"])
            self._tokenizer = Tokenizer("data/tiny_shakespeare/")
            # self._tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        # Determine wandb init parameters for resume
        wandb_init_kwargs: dict = {
            "project": project,
            "entity": entity,
            "name": exp_name,
        }

        # Attempt to resume wandb run if resume_training is enabled
        if resume_training and results_dir:
            run_id = _find_wandb_run_id_for_resume(
                results_dir=results_dir,
                experiment_name=exp_name,
                load_checkpoint_path=load_checkpoint_path,
            )
            if run_id:
                wandb_init_kwargs["id"] = run_id
                wandb_init_kwargs["resume"] = "allow"
                logger.info("Resuming wandb run with ID: %s", run_id)
            else:
                logger.info("No previous wandb run found, starting new run")
        elif resume_training and not results_dir:
            logger.warning(
                "resume_training is True but results_dir is not provided. "
                "Cannot resume wandb run without results_dir."
            )

        self._wandb_logger = wandb.init(**wandb_init_kwargs)

        # Save wandb run ID to results directory for future resumption
        self._save_wandb_run_id()

    def _save_wandb_run_id(self) -> None:
        """Save the wandb run ID to the results directory for future resumption."""
        if self._results_dir is None:
            logger.debug("results_dir not set, skipping wandb run ID save")
            return

        if self._wandb_logger is None or self._wandb_logger.id is None:
            logger.warning("wandb logger not initialized, cannot save run ID")
            return

        # Construct the full results path (results_dir + experiment_name)
        full_results_dir = os.path.join(self._results_dir, self._exp_name)
        os.makedirs(full_results_dir, exist_ok=True)

        run_id_path = os.path.join(full_results_dir, WANDB_RUN_ID_FILENAME)
        try:
            with open(run_id_path, "w", encoding="utf-8") as f:
                f.write(self._wandb_logger.id)
            logger.info("Saved wandb run ID to: %s", run_id_path)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("Failed to save wandb run ID: %s", e)

    def update_config(self, config: dict, *args, **kwargs):  # pylint: disable=unused-argument
        """Update the config of the callback.

        Args:
            config: The config to update.
        """
        # allow_val_change=True is needed when resuming a run, as the config
        # may already be set from the previous run
        self._wandb_logger.config.update(config, allow_val_change=True)

    def filter_label(self, label: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor | None:
        """Filter the label.

        Args:
            label: The label to filter.
        """
        if isinstance(label, torch.Tensor):
            return label[0]
        if isinstance(label, dict):
            return {k: self.filter_label(v) for k, v in label.items()}
        return label

    def _log_image(
        self,
        step: int,
        phase: str,
        tag: str,
        image: torch.Tensor | dict[str, torch.Tensor],
        target_label: torch.Tensor | None = None,
        pred_label: torch.Tensor | None = None,
    ):
        """Log the image.

        Args:
            step: The step of the image.
            phase: The phase of the image.
            image: The image to log.
        """
        if isinstance(image, torch.Tensor):
            image = torch_np_utils.to_numpy(image)
        elif isinstance(image, dict):
            for key, img_value in image.items():
                target = target_label[key] if target_label is not None else None
                pred = pred_label[key] if pred_label is not None else None
                self._log_image(step, phase, f"{tag}/{key}", img_value, target, pred)
            return

        if image.ndim == 4:
            image = np.transpose(image, (0, 2, 3, 1))
        image = image[0]
        target_label = self.filter_label(target_label)
        pred_label = self.filter_label(pred_label)

        if image.ndim not in [2, 3]:
            return

        if image.ndim == 3 and image.shape[2] not in [1, 3] and self._num_classes:
            image = np.argmax(image, axis=2)
            # Convert class index to class color
            colored_image = np.zeros((*image.shape, 3))
            for i in range(self._num_classes):
                colored_image[image == i] = self._class_colors[i]  # type: ignore
            image = colored_image

        wandb_image_kwargs = {"caption": ""}
        if target_label is not None:
            wandb_image_kwargs["caption"] += f"Target: {target_label} "
        if pred_label is not None:
            wandb_image_kwargs["caption"] += f"Pred: {pred_label} "
        wandb_image = wandb.Image(image, **wandb_image_kwargs)

        self._wandb_logger.log({f"{phase}/{tag}": wandb_image, f"{phase}/step": step})

    def _log_text(
        self, step: int, phase: str, tag: str, text: torch.Tensor | dict[str, torch.Tensor]
    ):
        """Log the text.

        Args:
            step: The step of the text.
            phase: The phase of the text.
            tag: The tag of the text.
            text: The text to log.
        """
        if isinstance(text, torch.Tensor):
            if text.ndim in [1, 2]:
                text = self._tokenizer.decode(text[0, ...])
            elif text.ndim == 3:
                text = F.softmax(text, dim=-1)
                text = torch.argmax(text, dim=-1)
                text = self._tokenizer.decode(text[0, ...])
            else:
                raise ValueError(f"Text tensor has invalid dimensions: {text.ndim}")
        elif isinstance(text, dict):
            for key, text_value in text.items():
                self._log_text(step, phase, f"{tag}/{key}", text_value)
            return

        print(f"Logging step: {step}, phase: {phase}, tag: {tag}, text: {text}")

        # self._log_table.add_data(step, phase, tag, text)
        # push_table = wandb.Table(data=self._log_table.data, columns=self._log_table.columns)
        # wandb.log({f"{phase}/text": push_table})

    def on_batch_end(
        self,
        step: int,
        phase: str,
        input_data: torch.Tensor | dict[str, torch.Tensor],
        target_data: torch.Tensor | dict[str, torch.Tensor],
        pred_data: torch.Tensor | dict[str, torch.Tensor],
    ):
        """Log the metrics.

        Args:
            step: The step of the metrics.
            phase: The phase of the metrics.
            input_data: The input of the metrics.
            target_data: The target of the metrics.
            pred_data: The predicted of the metrics.
        """
        # Only log at specified intervals to reduce GPU-CPU sync overhead
        if step % self._logging_interval == 0:
            if self._log_text_data:
                self._log_text(step, phase, "input", input_data)
                self._log_text(step, phase, "target", target_data)
                self._log_text(step, phase, "pred", pred_data)
            elif self._mode == "classification":
                self._log_image(
                    step,
                    phase,
                    "input",
                    input_data,
                    target_label=target_data,
                    pred_label=pred_data,
                )
            else:
                self._log_image(step, phase, "input", input_data)
                self._log_image(step, phase, "target", target_data)
                self._log_image(step, phase, "pred", pred_data)

    def log_metrics(self, step: int, phase: str, metrics: dict[str, float]):
        """Log the metrics.

        Args:
            step: The step of the metrics.
            metrics: The metrics to log.
        """
        metrics = {f"{phase}/{key}": value for key, value in metrics.items()}
        metrics[f"{phase}/step"] = step
        self._wandb_logger.log(metrics)

    def on_lr_update(self, epoch: int, lr: float, scheduler_type: str = ""):
        """Log the learning rate.

        Args:
            epoch: The epoch of the learning rate.
            lr: The learning rate.
            scheduler_type: The type of the learning rate.
        """
        self._wandb_logger.log({"epoch": epoch, f"learning_rate/{scheduler_type}": lr})
