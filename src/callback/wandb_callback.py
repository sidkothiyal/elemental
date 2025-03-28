"""Wandb callback for Trainer"""
import logging
import random
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch

import wandb
from callback import callback
from utils import torch_np_utils

logger = logging.getLogger(__name__)


class WandbCallback(callback.Callback):
    """Wandb callback for Trainer"""

    def __init__(
        self,
        exp_name: str,
        project: Optional[str] = "elemental",
        entity: Optional[str] = "elemental-ai",
        logging_interval: int = 100,
        num_images_to_log: int = 1,
        class_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
        num_classes: Optional[int] = None,
    ):
        """Initialize the callback.

        Args:
            exp_name: The name of the experiment.
            project: The name of the project.
            entity: The name of the entity.
            logging_interval: The logging interval.
            num_images_to_log: The number of images to log.
        """
        logger.info(
            (
                "Initializing WandbCallback with "
                "project: %s, entity: %s, exp_name: %s"
            ),
            project,
            entity,
            exp_name,
        )
        self._logging_interval = logging_interval
        self._num_images_to_log = num_images_to_log
        if self._num_images_to_log != 1:
            raise NotImplementedError(
                "num_images_to_log is not 1, but %d", self._num_images_to_log
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

        self._wandb_logger = wandb.init(
            project=project,
            entity=entity,
            name=exp_name,
        )

    def update_config(self, *args, **kwargs):
        """Update the config of the callback."""
        self._wandb_logger.config.update(*args, **kwargs)

    def _log_image(
        self,
        step: int,
        phase: str,
        tag: str,
        image: Union[torch.Tensor, Dict[str, torch.Tensor]],
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
                self._log_image(step, phase, f"{tag}/{key}", img_value)
            return

        if image.ndim == 4:
            image = np.transpose(image, (0, 2, 3, 1))
        image = image[0]

        if image.ndim == 3 and image.shape[2] != 1 and image.shape[2] != 3:
            image = np.argmax(image, axis=2)
            # Convert class index to class color
            colored_image = np.zeros((*image.shape, 3))
            for i in range(self._num_classes):
                colored_image[image == i] = self._class_colors[i]
            image = colored_image

        self._wandb_logger.log(
            {f"{phase}/{tag}": wandb.Image(image)}, step=step
        )

    def on_batch_end(
        self,
        step: int,
        phase: str,
        input_data: Union[torch.Tensor, Dict[str, torch.Tensor]],
        target_data: Union[torch.Tensor, Dict[str, torch.Tensor]],
        pred_data: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ):
        """Log the metrics.

        Args:
            step: The step of the metrics.
            phase: The phase of the metrics.
            input_data: The input of the metrics.
            target_data: The target of the metrics.
            pred_data: The predicted of the metrics.
        """
        if step % self._logging_interval == 0:
            self._log_image(step, phase, "input", input_data)
            self._log_image(step, phase, "target", target_data)
            self._log_image(step, phase, "pred", pred_data)

    def log_metrics(self, step: int, phase: str, metrics: Dict[str, float]):
        """Log the metrics.

        Args:
            step: The step of the metrics.
            metrics: The metrics to log.
        """
        metrics = {f"{phase}/{key}": value for key, value in metrics.items()}
        self._wandb_logger.log(metrics, step=step)
