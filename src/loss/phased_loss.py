"""Phased Loss allows different losses to be applied in different phases of training."""

import logging
from typing import List, Optional

import torch
from torch import nn

from loss.utils import call_loss_with_epoch

logger = logging.getLogger(__name__)


class PhasedLoss(nn.Module):
    """Loss that switches between different loss functions based on training progress.

    Uses percentage-based phase definitions for cleaner configuration.
    Stateless design - epoch is passed to forward() directly.
    """

    def __init__(
        self,
        total_epochs: int,
        phases: List[dict],
    ):
        """Initialize the PhasedLoss.

        Args:
            total_epochs: Total number of training epochs
            phases: List of phase configurations. Each phase is a dict with:
                - 'run_until': float in [0, 1], percentage of training when this phase ends
                               (e.g., 0.5 means run until 50% of training)
                - 'loss': nn.Module, the loss function to use in this phase

        Example:
            >>> # 3 phases: L1 (0-25%), MSE (25-75%), SSIM (75-100%)
            >>> phases = [
            ...     {'run_until': 0.25, 'loss': nn.L1Loss()},
            ...     {'run_until': 0.75, 'loss': nn.MSELoss()},
            ...     {'run_until': 1.0, 'loss': SSIMLoss()},
            ... ]
            >>> loss_fn = PhasedLoss(total_epochs=200, phases=phases)
        """
        super().__init__()

        if not phases:
            raise ValueError("At least one phase must be provided")
        if total_epochs <= 0:
            raise ValueError(f"total_epochs must be positive, got {total_epochs}")

        self.total_epochs = total_epochs

        # Validate and process phases
        processed_phases = []
        for i, phase in enumerate(phases):
            if "run_until" not in phase:
                raise ValueError(f"Phase {i} missing 'run_until'")
            if "loss" not in phase:
                raise ValueError(f"Phase {i} missing 'loss'")

            run_until = phase["run_until"]
            if not 0 < run_until <= 1.0:
                raise ValueError(f"Phase {i} 'run_until' must be in (0, 1], got {run_until}")
            if not isinstance(phase["loss"], nn.Module):
                raise ValueError(f"Phase {i} 'loss' must be an nn.Module")

            # Calculate actual epoch boundary
            end_epoch = int(run_until * total_epochs)

            processed_phases.append(
                {
                    "end_epoch": end_epoch,
                    "loss": phase["loss"],
                }
            )

        # Sort phases by end_epoch
        self.phases = sorted(processed_phases, key=lambda p: p["end_epoch"])

        # Store loss functions as ModuleList for proper registration
        self.loss_fns = nn.ModuleList([p["loss"] for p in self.phases])

        # Log configuration
        logger.info(
            "PhasedLoss initialized for %d epochs with %d phases:", total_epochs, len(self.phases)
        )
        prev_end = 0
        for i, phase in enumerate(self.phases):
            logger.info(
                "  Phase %d: epochs [%d, %d), loss=%s",
                i,
                prev_end,
                phase["end_epoch"],
                phase["loss"].__class__.__name__,
            )
            prev_end = phase["end_epoch"]

    def _get_phase_for_epoch(self, epoch: int) -> Optional[int]:
        """Get the index of the phase for the given epoch.

        Args:
            epoch: The epoch to check

        Returns:
            The index of the phase, or None if no phase matches
        """
        for i, phase in enumerate(self.phases):
            if epoch < phase["end_epoch"]:
                return i
        return None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        epoch: int,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            pred: The predicted tensor
            target: The target tensor
            epoch: Current training epoch (required)
            *args: Additional arguments to pass to the loss function
            **kwargs: Additional keyword arguments to pass to the loss function

        Returns:
            The computed loss value

        Raises:
            ValueError: If no phase matches the epoch and no default loss is set
        """
        phase_idx = self._get_phase_for_epoch(epoch)

        if phase_idx is None:
            raise ValueError(f"No phase found for epoch {epoch} (total_epochs={self.total_epochs})")

        loss_fn = self.loss_fns[phase_idx]

        loss = call_loss_with_epoch(loss_fn, pred, target, epoch=epoch, *args, **kwargs)
        return loss
