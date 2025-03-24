"""Wandb callback for Trainer"""
import logging
import os

import wandb
from callback import callback

logger = logging.getLogger(__name__)


class WandbCallback(callback.Callback):
    """Wandb callback for Trainer"""

    def __init__(
        self,
        exp_name: str,
        project: str = "elemental",
        entity: str = "elemental-ai",
    ):
        """Initialize the callback.

        Args:
            exp_name (str): The name of the experiment.
            project (str): The name of the project.
            entity (str): The name of the entity.
        """
        logger.info(
            "Initializing WandbCallback with "
            f"project: {project}, entity: {entity}, exp_name: {exp_name}"
        )
        self._wandb_logger = wandb.init(
            project=project,
            entity=entity,
            name=exp_name,
        )

    def update_config(self, *args, **kwargs):
        """Update the config of the callback."""
        self._wandb_logger.config.update(*args, **kwargs)
