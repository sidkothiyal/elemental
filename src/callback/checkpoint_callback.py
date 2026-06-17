"""Checkpoint callback for Trainer."""

import logging
import os
from collections import deque
from typing import Any

import torch
from torch import nn

from callback import callback
from utils import checkpoint as ckpt_utils

logger = logging.getLogger(__name__)


class CheckpointCallback(callback.Callback):  # pylint: disable=too-many-instance-attributes
    """Checkpoint callback for saving model, optimizer, scheduler, and loss states."""

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        save_dir: str,
        save_every_n_epochs: int | None = 10,
        save_every_n_steps: int | None = None,
        save_best: bool = True,
        save_last: bool = True,
        monitor_metric: str = "loss_total",
        mode: str = "min",
        max_checkpoints_saved: int | None = None,
    ):
        """Initialize the checkpoint callback.

        Args:
            save_dir: Directory to save checkpoints.
            save_every_n_epochs: Save checkpoint every n epochs (default: 10).
            save_every_n_steps: Save checkpoint every n steps (default: None).
            save_best: Whether to save the best model based on validation loss (default: True).
            save_last: Whether to save the last model at the end of training (default: True).
            monitor_metric: Metric to monitor for best model (default: "loss_total").
            mode: Mode for monitoring metric - "min" or "max" (default: "min").
            max_checkpoints_saved: Maximum number of periodic checkpoints to keep
                (default: None, unlimited). Best and last checkpoints are always kept and don't
                count towards this limit.
        """
        super().__init__()
        self.save_dir = save_dir
        self.save_every_n_epochs = save_every_n_epochs
        self.save_every_n_steps = save_every_n_steps
        self.save_best = save_best
        self.save_last = save_last
        self.monitor_metric = monitor_metric
        self.mode = mode
        self.max_checkpoints_saved = max_checkpoints_saved

        # Will be populated via update_config
        self.model: nn.Module | nn.ModuleDict | None = None
        self.optimizer: torch.optim.Optimizer | dict[str, torch.optim.Optimizer] | None = None
        self.scheduler: Any | None = None
        self.criterion: nn.Module | None = None
        self.training_mode: str | None = None
        # Train dataloader (for streaming resume) + the latest optimizer-step
        # count, both persisted so resume restores the LR schedule and stream
        # position correctly. See Trainer._restore_dataloader_state.
        self.train_loader: Any | None = None
        self.current_global_step: int = 0

        # Track best metric value
        self.best_metric = float("inf") if mode == "min" else float("-inf")
        self.current_epoch = 0
        self.current_val_metrics: dict[str, float] = {}

        # Track periodic checkpoints (excludes "best" and "last")
        self.periodic_checkpoints: deque[str] = deque()
        self.total_checkpoints_saved = 0

        # Remote backup strategy (set via update_config)
        self.remote_backup = None

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        logger.info(
            "CheckpointCallback initialized with save_dir: %s, max_checkpoints: %s",
            save_dir,
            max_checkpoints_saved if max_checkpoints_saved else "unlimited",
        )

    def update_config(self, config: dict, *args, **kwargs):  # pylint: disable=unused-argument
        """Update the config of the callback.

        Args:
            config: The config to update. Expected keys:
                - model: The model to checkpoint
                - optimizer: The optimizer to checkpoint
                - scheduler: The learning rate scheduler to checkpoint
                - criterion: The loss function to checkpoint
                - mode: Training mode (e.g., 'classification', 'gan', 'autoencoder')
        """
        self.model = config.get("model", self.model)
        self.optimizer = config.get("optimizer", self.optimizer)
        self.scheduler = config.get("scheduler", self.scheduler)
        self.criterion = config.get("criterion", self.criterion)
        self.training_mode = config.get("mode", self.training_mode)
        self.remote_backup = config.get("remote_backup", self.remote_backup)
        self.train_loader = config.get("train_loader", self.train_loader)

        logger.info("CheckpointCallback config updated")

    def _save_checkpoint(
        self,
        name: str,
        epoch: int | None = None,
        step: int | None = None,
        is_periodic: bool = False,
    ):
        """Save checkpoint to disk.

        Args:
            name: Name of the checkpoint file (without extension).
            epoch: Current epoch number.
            step: Current step number.
            is_periodic: Whether this is a periodic checkpoint (subject to
                max_checkpoints_saved limit).
        """
        if self.model is None:
            logger.warning("Model not set, skipping checkpoint save")
            return

        # Save checkpoint using utility function. ``global_optim_step`` and the
        # streaming dataloader state are saved as extra metadata so resume can
        # restore the LR schedule and the stream position exactly.
        dataloader_state = None
        if self.train_loader is not None and hasattr(self.train_loader, "state_dict"):
            try:
                dataloader_state = self.train_loader.state_dict()
            except Exception:  # pylint: disable=broad-exception-caught
                logger.warning("Failed to capture dataloader state for checkpoint", exc_info=True)
        checkpoint_path = os.path.join(self.save_dir, f"{name}.pt")
        ckpt_utils.save_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            criterion=self.criterion,
            epoch=epoch,
            step=step,
            global_optim_step=self.current_global_step,
            dataloader_state=dataloader_state,
        )
        self.total_checkpoints_saved += 1
        logger.info(
            "Saved checkpoint: %s (total: %d)", checkpoint_path, self.total_checkpoints_saved
        )

        # Track and manage periodic checkpoints
        if is_periodic:
            self.periodic_checkpoints.append(checkpoint_path)

            # Remove oldest checkpoint if limit exceeded
            if self.max_checkpoints_saved is not None:
                while len(self.periodic_checkpoints) > self.max_checkpoints_saved:
                    oldest_checkpoint = self.periodic_checkpoints.popleft()
                    if os.path.exists(oldest_checkpoint):
                        os.remove(oldest_checkpoint)
                        logger.info(
                            "Removed old checkpoint: %s (keeping %d/%d periodic checkpoints)",
                            oldest_checkpoint,
                            len(self.periodic_checkpoints),
                            self.max_checkpoints_saved,
                        )

        # Upload to remote backup if the strategy is configured for this name
        if self.remote_backup and self.remote_backup.should_backup(name):
            try:
                self.remote_backup.upload(checkpoint_path, name)
            except Exception:  # pylint: disable=broad-exception-caught
                logger.warning("Remote backup upload failed for %s", name, exc_info=True)

    def on_epoch_start(self, epoch: int, phase: str):
        """Called when the epoch starts.

        Args:
            epoch: Current epoch number.
            phase: Current phase (train/val/test).
        """
        if phase == "train":
            self.current_epoch = epoch

            # Save periodic checkpoints at the start of each epoch (after completing previous epoch)
            # Skip epoch 0 since no training has occurred yet
            if self.save_every_n_epochs is not None and epoch > 0:
                if epoch % self.save_every_n_epochs == 0:
                    self._save_checkpoint(
                        f"checkpoint-epoch-{epoch}", epoch=epoch - 1, is_periodic=True
                    )

    def on_batch_end(
        self,
        step: int,
        phase: str,
        data: dict,  # pylint: disable=unused-argument
    ):
        """Called when the batch ends.

        Args:
            step: Current step number.
            phase: Current phase (train/val/test).
            data: Unified data container (unused by this callback).
        """
        # Track the latest optimizer-step count so any checkpoint (periodic,
        # best, last) records the correct resume position.
        if phase == "train":
            gs = data.get("global_optim_step")
            if gs is not None:
                self.current_global_step = int(gs)

        # Save periodic checkpoints by step during training
        if phase == "train" and self.save_every_n_steps is not None:
            if step > 0 and step % self.save_every_n_steps == 0:
                self._save_checkpoint(
                    f"checkpoint-step-{step}", epoch=self.current_epoch, step=step, is_periodic=True
                )

    def log_metrics(
        self, step: int, phase: str, metrics: dict[str, float]  # pylint: disable=unused-argument
    ):
        """Store validation metrics and check for best model.

        Best-model saving lives here (rather than in ``on_epoch_end``)
        because ``on_epoch_end`` is now called *before* ``log_metrics``
        inside the batch loop, ensuring that MetricsCallback has a chance
        to inject its computed metrics into *metrics* first.

        Args:
            step: Current step number.
            phase: Current phase (train/val/test).
            metrics: Metrics dictionary.
        """
        if phase == "val":
            self.current_val_metrics = metrics

            if self.save_best:
                metric_value = metrics.get(self.monitor_metric)
                if metric_value is not None:
                    is_better = (
                        metric_value < self.best_metric
                        if self.mode == "min"
                        else metric_value > self.best_metric
                    )
                    if is_better:
                        self.best_metric = metric_value
                        self._save_checkpoint("best", epoch=self.current_epoch, is_periodic=False)
                        logger.info(
                            "New best model saved with %s: %.6f",
                            self.monitor_metric,
                            metric_value,
                        )

    def on_phase_end(self, phase: str):
        """Called when the phase ends.

        Args:
            phase: Current phase (train/val/test).
        """
        # Save last model at the end of training
        if phase == "train" and self.save_last:
            self._save_checkpoint("last", epoch=self.current_epoch, is_periodic=False)

    @staticmethod
    def load_checkpoint(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        checkpoint_path: str,
        model: nn.Module | nn.ModuleDict,
        optimizer: torch.optim.Optimizer | dict[str, torch.optim.Optimizer] | None = None,
        scheduler: Any | None = None,
        criterion: nn.Module | None = None,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        """Load checkpoint from disk.

        Args:
            checkpoint_path: Path to the checkpoint file.
            model: Model to load state into.
            optimizer: Optimizer to load state into (optional).
            scheduler: Scheduler to load state into (optional).
            criterion: Loss function to load state into (optional).
            map_location: Device to map tensors to (default: "cpu").

        Returns:
            Dictionary containing checkpoint metadata (epoch, step, etc.).
        """
        return ckpt_utils.load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            map_location=map_location,
        )
