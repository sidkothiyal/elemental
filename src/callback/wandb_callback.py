"""Wandb callback for Trainer."""

import logging
import os
import random
from typing import Any, cast

import numpy as np
import plotly.graph_objects as go
import torch
from torch.nn import functional as F

import wandb
from callback import callback
from data_processor.tokenizer import Tokenizer
from utils import torch_np_utils
from utils.wandb_utils import (  # noqa: F401 — re-export for backward compat
    WANDB_RUN_ID_FILENAME,
    find_wandb_run_id_for_resume,
)

logger = logging.getLogger(__name__)

os.environ["WANDB_SILENT"] = "True"


class WandbCallback(callback.Callback):
    """Wandb callback for Trainer."""

    def __init__(
        self,
        exp_name: str,
        project: str | None = "elemental",
        entity: str | None = "elemental-ai",
        logging_interval: int = 100,
        class_colors: dict[int, tuple[int, int, int]] | None = None,
        num_classes: int | None = None,
        mode: str | None = None,
        log_text_data: bool = False,
        tokenizer: str | None = None,
        results_dir: str | None = None,
        resume_training: bool = False,
        load_checkpoint_path: str | None = None,
        field_loggers: dict[str, str] | None = None,
        video_fps: int = 8,
        video_value_range: tuple[float, float] = (-1.0, 1.0),
        max_videos_per_log: int = 1,
        image_value_range: tuple[float, float] = (0.0, 1.0),
    ):
        """Initialize the callback.

        Args:
            exp_name: The name of the experiment.
            project: The name of the project.
            entity: The name of the entity.
            logging_interval: The logging interval.
            class_colors: The colors for each class for segmentation images.
            num_classes: The number of classes for segmentation images.
            mode: The mode of training (e.g. ``"classification"``).
            log_text_data: Whether to log text data.
            tokenizer: The tokenizer path to use for text logging.
            results_dir: Directory where results and checkpoints are saved.
                Required for resume_training to find previous wandb run ID.
            resume_training: Whether to resume training from a checkpoint.
                If True, will attempt to resume the wandb run from the previous
                training session.
            load_checkpoint_path: Path to the checkpoint file or ``"latest"``.
                Used to locate the wandb run ID file when resuming.
            field_loggers: Optional dict mapping data-container keys to logger
                type strings (e.g. ``"point3d"``). For those keys, the specified
                logger is used instead of the default image/text logging.
            video_fps: Playback fps for ``wandb.Video`` clips.
            video_value_range: ``(lo, hi)`` value range of incoming 5-D video
                tensors. Maps to ``[0, 255]`` uint8 before logging. Defaults to
                BVI-AOM convention ``(-1.0, 1.0)``.
            max_videos_per_log: Number of batch elements to log per video step.
            image_value_range: ``(lo, hi)`` value range of incoming image
                tensors. Defaults to ``(0.0, 1.0)`` to preserve historical
                behaviour; set to ``(-1.0, 1.0)`` for image AEs whose data is
                normalised to that range.
        """
        logger.info(
            "Initializing WandbCallback with project: %s, entity: %s, exp_name: %s",
            project,
            entity,
            exp_name,
        )
        self._logging_interval = logging_interval
        self._mode = mode
        self._log_text_data = log_text_data
        self._to_log: list[str] | None = None
        self._results_dir = results_dir
        self._exp_name = exp_name

        self._class_colors = class_colors
        self._num_classes = num_classes
        self._field_loggers = dict(field_loggers) if field_loggers else None

        self._video_fps = video_fps
        # Hydra/OmegaConf delivers list values for sequence params; coerce to a
        # 2-tuple so downstream helpers can rely on the static type.
        self._video_value_range = cast(tuple[float, float], tuple(video_value_range))
        self._max_videos_per_log = max_videos_per_log
        self._image_value_range = cast(tuple[float, float], tuple(image_value_range))
        self._warned_shape_keys: set[str] = set()

        if self._num_classes and self._class_colors is None:
            self._class_colors = self._generate_class_colors(self._num_classes)

        if log_text_data:
            self._setup_text_logging(tokenizer)

        wandb_init_kwargs = self._build_wandb_init_kwargs(
            project=project,
            entity=entity,
            exp_name=exp_name,
            resume_training=resume_training,
            results_dir=results_dir,
            load_checkpoint_path=load_checkpoint_path,
        )
        self._wandb_logger = wandb.init(**wandb_init_kwargs)

        # Save wandb run ID to results directory for future resumption
        self._save_wandb_run_id()

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_class_colors(num_classes: int) -> dict[int, tuple[int, int, int]]:
        """Generate random RGB colours for *num_classes* classes."""
        return {
            i: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for i in range(num_classes)
        }

    def _setup_text_logging(self, tokenizer: str | None) -> None:
        """Initialise the tokenizer and wandb table for text logging."""
        if tokenizer is None:
            raise ValueError("tokenizer path is required when log_text_data is True")
        self._log_table = wandb.Table(columns=["step", "phase", "tag", "text"])
        self._tokenizer = Tokenizer(tokenizer)

    @staticmethod
    def _build_wandb_init_kwargs(
        project: str | None,
        entity: str | None,
        exp_name: str,
        resume_training: bool,
        results_dir: str | None,
        load_checkpoint_path: str | None,
    ) -> dict:
        """Build the keyword arguments dict for ``wandb.init``."""
        kwargs: dict = {
            "project": project,
            "entity": entity,
            "name": exp_name,
        }

        if resume_training and results_dir:
            run_id = find_wandb_run_id_for_resume(
                results_dir=results_dir,
                experiment_name=exp_name,
                load_checkpoint_path=load_checkpoint_path,
            )
            if run_id:
                kwargs["id"] = run_id
                kwargs["resume"] = "allow"
                logger.info("Resuming wandb run with ID: %s", run_id)
            else:
                logger.info("No previous wandb run found, starting new run")
        elif resume_training and not results_dir:
            logger.warning(
                "resume_training is True but results_dir is not provided. "
                "Cannot resume wandb run without results_dir."
            )

        return kwargs

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

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def update_config(self, config: dict):
        """Update the config of the callback.

        Args:
            config: The config to update.  Recognised keys:
                - ``to_log``: optional list of data-container keys that
                  should be logged by :meth:`on_batch_end`.  ``None``
                  (the default) means log everything.
        """
        to_log = config.get("to_log")
        if to_log is not None:
            self._to_log = list(to_log)

        # allow_val_change=True is needed when resuming a run, as the config
        # may already be set from the previous run
        self._wandb_logger.config.update(config, allow_val_change=True)

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _first_sample(
        label: torch.Tensor | dict[str, torch.Tensor] | None,
    ) -> torch.Tensor | dict | None:
        """Return the first batch element from *label*.

        Works recursively on nested dicts.
        """
        if label is None:
            return None
        if isinstance(label, torch.Tensor):
            return label[0]
        if isinstance(label, dict):
            return {k: WandbCallback._first_sample(v) for k, v in label.items()}
        return label

    # -- Value-range helpers -------------------------------------------

    @staticmethod
    def _denormalize_to_uint8(x: torch.Tensor, value_range: tuple[float, float]) -> torch.Tensor:
        """Map a tensor in ``value_range`` to ``uint8`` ``[0, 255]`` (clamped).

        Casts bf16/fp16 → fp32 first so the numpy hop downstream is safe.
        """
        lo, hi = value_range
        x = x.float()
        if hi != lo:
            x = (x - lo) / (hi - lo)
        return (x.clamp(0.0, 1.0) * 255).to(torch.uint8)

    @staticmethod
    def _denormalize_to_unit_float(x: np.ndarray, value_range: tuple[float, float]) -> np.ndarray:
        """Map a numpy array in ``value_range`` to float ``[0, 1]`` (clamped).

        No-op when ``value_range == (0.0, 1.0)`` so the default path stays
        bit-identical to the historical behaviour.
        """
        lo, hi = value_range
        if (lo, hi) == (0.0, 1.0):
            return x
        return np.clip((x - lo) / (hi - lo), 0.0, 1.0)

    def _warn_unsupported_shape(self, tag: str, shape) -> None:
        """Emit a one-time warning for an unsupported logging shape."""
        if tag in self._warned_shape_keys:
            return
        logger.warning(
            "WandbCallback: unsupported tensor shape for key %r: %s — skipping",
            tag,
            tuple(shape),
        )
        self._warned_shape_keys.add(tag)

    # -- Image logging -------------------------------------------------

    def _log_image(
        self,
        step: int,
        phase: str,
        tag: str,
        image: torch.Tensor | dict[str, torch.Tensor],
        target_label: torch.Tensor | None = None,
        pred_label: torch.Tensor | None = None,
    ):
        """Log an image (or dict of images) to wandb.

        Handles dicts by recursing, then processes a single image tensor:
        transposes NCHW → NHWC, takes the first sample, and optionally
        converts class-index maps to RGB using ``_class_colors``. 5-D
        ``(B, T, C, H, W)`` tensors are auto-routed to :meth:`_log_video`.
        """
        if isinstance(image, dict):
            for key, img_value in image.items():
                target = target_label[key] if target_label is not None else None
                pred = pred_label[key] if pred_label is not None else None
                self._log_image(step, phase, f"{tag}/{key}", img_value, target, pred)
            return

        if isinstance(image, torch.Tensor) and image.ndim == 5:
            self._log_video(step, phase, tag, image)
            return

        if isinstance(image, torch.Tensor):
            image = torch_np_utils.to_numpy(image)

        if image.ndim == 4:
            image = np.transpose(image, (0, 2, 3, 1))
        image = image[0]
        target_label = self._first_sample(target_label)
        pred_label = self._first_sample(pred_label)

        if image.ndim not in [2, 3]:
            self._warn_unsupported_shape(tag, image.shape)
            return

        if image.ndim == 3 and image.shape[2] not in [1, 3] and self._num_classes:
            image = np.argmax(image, axis=2)
            # Convert class index to class color
            colored_image = np.zeros((*image.shape, 3))
            for i in range(self._num_classes):
                colored_image[image == i] = self._class_colors[i]  # type: ignore
            image = colored_image
        else:
            image = self._denormalize_to_unit_float(image, self._image_value_range)

        caption = ""
        if target_label is not None:
            caption += f"Target: {target_label} "
        if pred_label is not None:
            caption += f"Pred: {pred_label} "
        wandb_image = wandb.Image(image, caption=caption)

        self._wandb_logger.log({f"{phase}/{tag}": wandb_image, f"{phase}/step": step})

    # -- Video logging -------------------------------------------------

    def _log_video(
        self,
        step: int,
        phase: str,
        tag: str,
        video: torch.Tensor,
    ) -> None:
        """Log a 5-D ``(B, T, C, H, W)`` tensor as ``wandb.Video`` clip(s).

        Logs the first ``self._max_videos_per_log`` batch elements after
        mapping ``self._video_value_range`` → ``uint8`` ``[0, 255]``.
        """
        if video.shape[0] == 0:
            return
        n = min(self._max_videos_per_log, video.shape[0])
        clips = self._denormalize_to_uint8(video[:n].detach(), self._video_value_range)
        arr = torch_np_utils.to_numpy(clips)  # (n, T, C, H, W) uint8
        if n == 1:
            payload: Any = wandb.Video(arr[0], fps=self._video_fps)
        else:
            payload = [wandb.Video(arr[i], fps=self._video_fps) for i in range(n)]
        self._wandb_logger.log({f"{phase}/{tag}": payload, f"{phase}/step": step})

    # -- Text logging --------------------------------------------------

    def _log_text(
        self, step: int, phase: str, tag: str, text: torch.Tensor | dict[str, torch.Tensor]
    ):
        """Log text data to wandb.

        Tensor inputs are decoded via the configured tokenizer.  Dicts are
        handled recursively.
        """
        if isinstance(text, dict):
            for key, text_value in text.items():
                self._log_text(step, phase, f"{tag}/{key}", text_value)
            return

        if isinstance(text, torch.Tensor):
            if text.ndim in [1, 2]:
                text = self._tokenizer.decode(text[0, ...])
            elif text.ndim == 3:
                text = F.softmax(text, dim=-1)
                text = torch.argmax(text, dim=-1)
                text = self._tokenizer.decode(text[0, ...])
            else:
                raise ValueError(f"Text tensor has invalid dimensions: {text.ndim}")

        logger.info("Logging step: %d, phase: %s, tag: %s, text: %s", step, phase, tag, text)

    # -- 3D scatter / line logging -------------------------------------

    def _prepare_3d_data(
        self, value: torch.Tensor | np.ndarray, logger_name: str
    ) -> np.ndarray | None:
        """Convert *value* to a ``(N, 3)`` numpy array, or ``None`` on failure.

        Handles tensor → numpy conversion, takes the first batch element if
        the array is 3-D, and validates the final shape.
        """
        if isinstance(value, torch.Tensor):
            value = torch_np_utils.to_numpy(value)
        if value.ndim == 3:
            value = value[0]
        if value.ndim != 2 or value.shape[1] != 3:
            logger.warning(
                "%s logger expects shape (N, 3) or (B, N, 3), got %s",
                logger_name,
                value.shape,
            )
            return None
        return value

    def _log_point3d(
        self, step: int, phase: str, tag: str, value: torch.Tensor | np.ndarray
    ) -> None:
        """Log a ``(N, 3)`` tensor as a 3-D point cloud (markers only)."""
        data = self._prepare_3d_data(value, "point3d")
        if data is None:
            return
        xs, ys, zs = data[:, 0], data[:, 1], data[:, 2]
        fig = go.Figure(data=[go.Scatter3d(x=xs, y=ys, z=zs, mode="markers", marker={"size": 4})])
        fig.update_layout(
            title=tag,
            scene={
                "xaxis_title": "x",
                "yaxis_title": "y",
                "zaxis_title": "z",
                "aspectmode": "data",
            },
        )
        self._wandb_logger.log({f"{phase}/{tag}": wandb.Plotly(fig), f"{phase}/step": step})

    def _log_line3d(
        self, step: int, phase: str, tag: str, value: torch.Tensor | np.ndarray
    ) -> None:
        """Log a ``(N, 3)`` tensor as a 3-D line with directional color gradient."""
        data = self._prepare_3d_data(value, "line3d")
        if data is None:
            return
        xs, ys, zs = data[:, 0], data[:, 1], data[:, 2]
        point_indices = np.arange(len(xs), dtype=np.float64)
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="lines+markers",
                    line={
                        "width": 4,
                        "color": point_indices,
                        "colorscale": "Bluered",
                        "showscale": True,
                        "colorbar": {"title": "Point order"},
                    },
                    marker={
                        "size": 3,
                        "color": point_indices,
                        "colorscale": "Bluered",
                        "showscale": False,
                    },
                )
            ]
        )
        fig.update_layout(
            title=tag,
            scene={
                "xaxis_title": "x",
                "yaxis_title": "y",
                "zaxis_title": "z",
                "aspectmode": "data",
            },
        )
        self._wandb_logger.log({f"{phase}/{tag}": wandb.Plotly(fig), f"{phase}/step": step})

    # -- Dispatch helpers ----------------------------------------------

    _FIELD_LOGGER_DISPATCH: dict[str, str] = {
        "point3d": "_log_point3d",
        "line3d": "_log_line3d",
    }

    def _dispatch_field_logger(
        self, step: int, phase: str, key: str, value: torch.Tensor | np.ndarray
    ) -> bool:
        """Call the configured logger for *key* if present. Returns ``True`` if dispatched."""
        if self._field_loggers is None or key not in self._field_loggers:
            return False
        logger_type = self._field_loggers[key]
        method_name = self._FIELD_LOGGER_DISPATCH.get(logger_type)
        if method_name is None:
            logger.warning("Unknown field logger type %r for key %s", logger_type, key)
            return False
        getattr(self, method_name)(step, phase, key, value)
        return True

    def _log_dict_scalars(self, step: int, phase: str, key: str, value: dict) -> bool:
        """If *value* is a dict of scalars, log each to wandb and return ``True``."""
        if not isinstance(value, dict) or len(value) == 0:
            return False
        if not all(isinstance(v, (int, float)) for v in value.values()):
            return False
        metrics = {f"{phase}/{key}/{subkey}": val for subkey, val in value.items()}
        metrics[f"{phase}/step"] = step
        self._wandb_logger.log(metrics)
        return True

    # -- Unified value logger ------------------------------------------

    def _log_value(
        self,
        step: int,
        phase: str,
        key: str,
        value,
        target_label=None,
        pred_label=None,
    ) -> None:
        """Log a single data-container value through the dispatch chain.

        Order: field_loggers → dict-of-scalars → text → image.
        """
        if value is None:
            return
        if self._dispatch_field_logger(step, phase, key, value):
            return
        if self._log_dict_scalars(step, phase, key, value):
            return
        if self._log_text_data:
            self._log_text(step, phase, key, value)
        else:
            self._log_image(
                step, phase, key, value, target_label=target_label, pred_label=pred_label
            )

    def _resolve_default_keys(self, data: dict) -> list[str]:
        """Build the default list of keys to log when ``_to_log`` is ``None``."""
        keys = ["input/input", "target", "pred"]
        # Include conditioning images (e.g. observation image in VLA diffusion)
        cond_data = data.get("input/cond")
        if isinstance(cond_data, dict):
            for cond_key, value in cond_data.items():
                if isinstance(value, torch.Tensor) and value.ndim >= 3:
                    keys.append(f"input/cond/{cond_key}")
        return keys

    # ------------------------------------------------------------------
    # Callback hooks
    # ------------------------------------------------------------------

    def on_batch_end(self, step: int, phase: str, data: dict):
        """Log visual data (images / text) from the unified data container.

        Args:
            step: The current step number.
            phase: The current phase (train/val/test).
            data: Unified data container with namespaced keys
                (``input/*``, ``target``, ``pred``, ``post_processed/*``).
        """
        if step % self._logging_interval != 0:
            return

        keys = self._to_log if self._to_log is not None else self._resolve_default_keys(data)

        for key in keys:
            value = data.get(key)
            if value is None:
                continue

            # In classification mode, pass target/pred labels alongside the
            # input image so they appear in the caption.
            if self._mode == "classification" and key == "input/input":
                self._log_value(
                    step,
                    phase,
                    key,
                    value,
                    target_label=data.get("target"),
                    pred_label=data.get("pred"),
                )
            else:
                self._log_value(step, phase, key, value)

    def log_metrics(self, step: int, phase: str, metrics: dict[str, float]):
        """Log scalar metrics to wandb.

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
