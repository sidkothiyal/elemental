"""Wandb callback for Trainer"""
import logging
import os
import random
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import wandb
from torch.nn import functional as F

from callback import callback
from data.tokenizer import Tokenizer
from utils import torch_np_utils

# from transformers import AutoTokenizer


logger = logging.getLogger(__name__)

os.environ["WANDB_SILENT"] = "True"


class WandbCallback(callback.Callback):
    """Wandb callback for Trainer"""

    def __init__(
        self,
        exp_name: str,
        project: Optional[str] = "elemental",
        entity: Optional[str] = "elemental-ai",
        logging_interval: int = 100,
        num_samples_to_log: int = 1,
        class_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
        num_classes: Optional[int] = None,
        mode: Optional[str] = None,
        log_text_data: bool = False,
        tokenizer: Optional[str] = None,
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
        self._wandb_logger = wandb.init(
            project=project,
            entity=entity,
            name=exp_name,
        )

    def update_config(self, *args, **kwargs):
        """Update the config of the callback."""
        self._wandb_logger.config.update(*args, **kwargs)

    def filter_label(
        self, label: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> Optional[torch.Tensor]:
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
        image: Union[torch.Tensor, Dict[str, torch.Tensor]],
        target_label: Optional[torch.Tensor] = None,
        pred_label: Optional[torch.Tensor] = None,
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
        self, step: int, phase: str, tag: str, text: Union[torch.Tensor, Dict[str, torch.Tensor]]
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

    def log_metrics(self, step: int, phase: str, metrics: Dict[str, float]):
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
