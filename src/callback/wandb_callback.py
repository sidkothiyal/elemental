"""Wandb callback for Trainer."""

import logging
import os
import random
from typing import Any, Callable, cast

import numpy as np
import plotly.graph_objects as go
import torch
from omegaconf import OmegaConf

import wandb
from callback import callback
from data_processor.bpe_tokenizer import BPETokenizer
from data_processor.tokenizer import Tokenizer
from utils import torch_np_utils
from utils.wandb_utils import (  # noqa: F401 — re-export for backward compat
    WANDB_RUN_ID_FILENAME,
    find_wandb_run_id_for_resume,
)

logger = logging.getLogger(__name__)

os.environ["WANDB_SILENT"] = "True"

# Default data-container keys logged (as images) when ``to_log`` is ``None``.
_DEFAULT_LOG_KEYS = ("input/input", "target", "pred")
# Modality strings accepted by ``to_log`` dict values.
_SUPPORTED_MODALITIES = frozenset({"text", "image", "video", "point3d", "line3d", "scalars"})
# Largest plausible side length for a 2-D tensor logged as a grayscale image;
# anything bigger (e.g. a (seq_len, vocab) logits matrix) is rejected.
_MAX_IMAGE_SIDE = 8192


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
        tokenizer_type: str | None = None,
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
            log_text_data: Deprecated. When True, the default keys are logged
                as ``"text"``. Prefer a ``to_log`` dict ``{key: "text"}``.
            tokenizer: The tokenizer path/dir to use for text logging.
            tokenizer_type: ``"bpe"``, ``"char"``, or ``"hf"`` (HuggingFace
                ``AutoTokenizer``). When ``None`` it is inferred from
                ``tokenizer`` (``.json`` → ``"bpe"``, else ``"char"``); ``"hf"``
                must be set explicitly (a bare name like ``"gpt2"`` is ambiguous).
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
        # Deprecated flag + tokenizer config, retained for text-logging setup
        # which now happens lazily in ``update_config`` (once ``to_log`` is
        # known). ``_to_log`` is the normalized ``{key: modality}`` map.
        self._log_text_data = log_text_data
        self._tokenizer_path = tokenizer
        self._tokenizer_type = tokenizer_type
        self._to_log: dict[str, str] = {}
        self._to_log_was_none = True
        self._decode_ids: Callable[[torch.Tensor], str] | None = None
        self._text_rows: list[list] = []
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

        wandb_init_kwargs = self._build_wandb_init_kwargs(
            project=project,
            entity=entity,
            exp_name=exp_name,
            resume_training=resume_training,
            results_dir=results_dir,
            load_checkpoint_path=load_checkpoint_path,
        )
        self._wandb_logger = wandb.init(**wandb_init_kwargs)

        # Make token/optimizer-step counts the default x-axes for train panels.
        self._define_step_metrics()

        # Save wandb run ID to results directory for future resumption
        self._save_wandb_run_id()

    def _define_step_metrics(self) -> None:
        """Declare cumulative tokens / optimizer step as W&B step metrics.

        The trainer logs ``train/tokens_consumed`` and ``train/optim_step``
        alongside the raw micro-batch ``train/step``. For LM runs the raw step is
        NOT comparable across configs (it differs by ``gradient_accumulation_steps``),
        so point all ``train/*`` panels at ``train/tokens_consumed`` by default —
        then runs with different micro-batching overlay correctly on tokens seen.
        Best-effort: older wandb without ``define_metric`` is silently skipped.
        """
        if self._wandb_logger is None or not hasattr(self._wandb_logger, "define_metric"):
            return
        try:
            self._wandb_logger.define_metric("train/tokens_consumed")
            self._wandb_logger.define_metric("train/optim_step")
            self._wandb_logger.define_metric("train/*", step_metric="train/tokens_consumed")
        except Exception:  # pylint: disable=broad-exception-caught
            logger.warning("wandb.define_metric failed — using default step x-axis", exc_info=True)

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

    def _setup_text_logging(self) -> None:
        """Build the tokenizer + decode adapter used for ``"text"`` keys.

        Idempotent: a second call (e.g. across train/val phases) is a no-op.
        The tokenizer flavour is taken from ``tokenizer_type`` or inferred from
        the path extension (``.json`` → BPE, otherwise the char-level tokenizer).
        """
        if self._decode_ids is not None:
            return
        path = self._tokenizer_path
        if path is None:
            raise ValueError("tokenizer path is required for text logging (to_log: text)")

        tok_type = self._tokenizer_type
        if tok_type is None:
            tok_type = "bpe" if str(path).endswith(".json") else "char"

        if tok_type == "bpe":
            bpe = BPETokenizer(path)
            self._decode_ids = lambda ids: bpe.decode(ids.tolist())
        elif tok_type == "char":
            # char Tokenizer.decode expects an iterable of id tensors.
            self._decode_ids = Tokenizer(path).decode
        elif tok_type == "hf":
            from transformers import AutoTokenizer  # noqa: PLC0415

            hf = AutoTokenizer.from_pretrained(path)
            self._decode_ids = lambda ids: hf.decode(ids.tolist())
        else:
            raise ValueError(
                f"unknown tokenizer_type {tok_type!r}; expected 'bpe', 'char', or 'hf'"
            )
        self._text_rows = []

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
                - ``to_log``: how :meth:`on_batch_end` logs data-container keys.
                  Accepts three shapes:
                    * ``None`` — log the default keys as images.
                    * ``list[str]`` — log each key as an image (legacy).
                    * ``dict[str, str]`` — explicit ``{key: modality}`` where
                      modality is one of ``text``/``image``/``video``/
                      ``point3d``/``line3d``/``scalars``.
        """
        # Track the null default so on_batch_end can dynamically include
        # conditioning images (e.g. VLA observation) discovered in the batch.
        self._to_log_was_none = config.get("to_log") is None
        self._to_log = self._normalize_to_log(config.get("to_log"))
        if "text" in self._to_log.values():
            self._setup_text_logging()

        # allow_val_change=True is needed when resuming a run, as the config
        # may already be set from the previous run
        self._wandb_logger.config.update(config, allow_val_change=True)

    def _normalize_to_log(self, to_log) -> dict[str, str]:
        """Coerce ``to_log`` (None / list / dict) into a ``{key: modality}`` map.

        Legacy ``field_loggers`` and ``log_text_data`` are folded in so
        un-migrated configs keep working.
        """
        # Hydra delivers DictConfig/ListConfig; normalize to native containers
        # so the isinstance checks below behave (a DictConfig is not a dict).
        if OmegaConf.is_config(to_log):
            to_log = OmegaConf.to_container(to_log, resolve=True)

        if to_log is None:
            mapping = {key: "image" for key in _DEFAULT_LOG_KEYS}
        elif isinstance(to_log, dict):
            mapping = {str(k): str(v) for k, v in to_log.items()}
        else:  # list / sequence of keys → image (legacy behaviour)
            mapping = {str(key): "image" for key in to_log}

        # field_loggers (deprecated) override the modality for their keys.
        if self._field_loggers:
            mapping.update({str(k): str(v) for k, v in self._field_loggers.items()})

        # log_text_data (deprecated) forces the default keys to text.
        if self._log_text_data:
            for key in _DEFAULT_LOG_KEYS:
                mapping.setdefault(key, "text")
                if mapping[key] == "image":
                    mapping[key] = "text"

        for key, modality in mapping.items():
            if modality not in _SUPPORTED_MODALITIES:
                logger.warning(
                    "WandbCallback: unknown modality %r for key %r — skipping",
                    modality,
                    key,
                )
        return mapping

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

        # Reject implausibly large 2-D tensors (e.g. a (seq_len, vocab) logits
        # matrix mistakenly routed here) before they become a huge image.
        if image.ndim == 2 and max(image.shape) > _MAX_IMAGE_SIDE:
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
        """Decode token tensors into the current interval's text buffer.

        Accepts token-id tensors (``(B, T)`` or ``(T,)``) or logits
        (``(B, T, vocab)``, reduced via ``argmax``). The first batch element is
        decoded with the configured tokenizer. Dicts are handled recursively.
        Rows are buffered in ``self._text_rows`` and flushed once per interval by
        :meth:`_flush_text_rows` — a single small table, never an accumulating one.
        """
        if isinstance(text, dict):
            for key, text_value in text.items():
                self._log_text(step, phase, f"{tag}/{key}", text_value)
            return

        ids = self._first_token_ids(text)
        if ids is None:
            self._warn_unsupported_shape(tag, getattr(text, "shape", ()))
            return

        assert self._decode_ids is not None  # set by _setup_text_logging for text keys
        self._text_rows.append([step, tag, self._decode_ids(ids)])

    def _flush_text_rows(self, step: int, phase: str) -> None:
        """Log the interval's buffered text rows as one fresh ``wandb.Table``.

        A new table per call (stepped media) keeps each artifact tiny — re-logging
        a growing accumulated table balloons run media into the tens of GB.
        """
        if not self._text_rows:
            return
        table = wandb.Table(columns=["step", "tag", "text"], data=self._text_rows)
        self._wandb_logger.log({f"{phase}/text_samples": table, f"{phase}/step": step})

    @staticmethod
    def _first_token_ids(text: torch.Tensor) -> torch.Tensor | None:
        """Reduce *text* to the first sample's 1-D token-id tensor, or ``None``.

        ``(B, T, vocab)`` logits → ``argmax(-1)`` then first sample;
        ``(B, T)`` ids → first sample; ``(T,)`` ids → as-is.
        """
        if not isinstance(text, torch.Tensor):
            return None
        if text.ndim == 3:
            text = torch.argmax(text, dim=-1)
        if text.ndim == 2:
            text = text[0]
        if text.ndim != 1:
            return None
        return text

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

    # Maps a ``to_log`` modality string to the method that logs it.
    _MODALITY_DISPATCH: dict[str, str] = {
        "image": "_log_image",
        "video": "_log_video",
        "text": "_log_text",
        "point3d": "_log_point3d",
        "line3d": "_log_line3d",
        "scalars": "_log_dict_scalars",
    }

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
        modality: str,
        target_label=None,
        pred_label=None,
    ) -> None:
        """Dispatch a single data-container value to its modality logger."""
        if value is None:
            return
        if modality == "image":
            self._log_image(
                step, phase, key, value, target_label=target_label, pred_label=pred_label
            )
            return
        method_name = self._MODALITY_DISPATCH.get(modality)
        if method_name is None:
            return  # unknown modality already warned during normalization
        getattr(self, method_name)(step, phase, key, value)

    def _discover_cond_keys(self, data: dict) -> list[str]:
        """Dynamic conditioning-image keys to log under the ``None`` default.

        Includes ``input/cond/<key>`` for tensors with ndim >= 3 (e.g. the
        observation image in VLA diffusion).
        """
        cond_data = data.get("input/cond")
        if not isinstance(cond_data, dict):
            return []
        return [
            f"input/cond/{cond_key}"
            for cond_key, value in cond_data.items()
            if isinstance(value, torch.Tensor) and value.ndim >= 3
        ]

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

        effective = dict(self._to_log)
        if self._to_log_was_none:
            for cond_key in self._discover_cond_keys(data):
                effective.setdefault(cond_key, "image")

        # Text rows for all keys this interval go into one fresh table (below).
        self._text_rows = []
        for key, modality in effective.items():
            value = data.get(key)
            if value is None:
                continue

            # In classification mode, pass target/pred labels alongside the
            # input image so they appear in the caption.
            if self._mode == "classification" and modality == "image" and key == "input/input":
                self._log_value(
                    step,
                    phase,
                    key,
                    value,
                    modality,
                    target_label=data.get("target"),
                    pred_label=data.get("pred"),
                )
            else:
                self._log_value(step, phase, key, value, modality)

        self._flush_text_rows(step, phase)

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
