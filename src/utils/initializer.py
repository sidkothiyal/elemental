"""Utility functions for initializing the experiment."""

import logging
import os
import random
from typing import Any

import hydra.utils as hy_utils
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

from callback import callback as callback_mod
from utils import checkpoint, distributed  # pylint: disable=no-name-in-module
from utils.remote_backup.base import RemoteBackupStrategy  # pylint: disable=no-name-in-module
from utils.s3_cache_backend import install_s3_cache_backend

logger = logging.getLogger(__name__)


def move_to_device(
    obj: object,
    device: torch.device,
    channels_last: bool = False,
) -> object:
    """Recursively move torch tensors to the specified device."""
    if isinstance(obj, torch.Tensor):
        if channels_last and obj.ndim == 4:
            return obj.to(device, memory_format=torch.channels_last, non_blocking=True)
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device, channels_last) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(item, device, channels_last) for item in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(item, device, channels_last) for item in obj)
    return obj


def _record_stream_recursive(obj: object, stream: torch.cuda.Stream) -> None:
    """Tell the CUDA caching allocator that *obj*'s tensors are used on *stream*.

    Without this, the allocator only knows the stream the tensor was allocated
    on (the prefetch side-stream) and may recycle the memory before the
    consumer stream (default stream) is done with it.
    """
    if isinstance(obj, torch.Tensor):
        if obj.is_cuda:
            obj.record_stream(stream)
    elif isinstance(obj, dict):
        for v in obj.values():
            _record_stream_recursive(v, stream)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _record_stream_recursive(item, stream)


class CUDAPrefetcher:
    """Prefetches batches to GPU using a side CUDA stream.

    Overlaps Host-to-Device memory copies with compute on the default stream,
    hiding most of the HtoD transfer latency.
    """

    def __init__(
        self,
        loader: DataLoader,
        device: torch.device,
        channels_last: bool = False,
    ):
        self.loader = loader
        self.device = device
        self.channels_last = channels_last
        self.stream = torch.cuda.Stream() if device.type == "cuda" else None

    def __len__(self):
        return len(self.loader)

    @property
    def dataset(self):
        """Get the dataset."""
        return self.loader.dataset

    def __iter__(self):
        loader_iter = iter(self.loader)
        try:
            next_batch = next(loader_iter)
        except StopIteration:
            return

        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                next_batch = move_to_device(
                    next_batch, self.device, channels_last=self.channels_last
                )

        for batch in loader_iter:
            if self.stream is not None:
                torch.cuda.current_stream().wait_stream(self.stream)
                _record_stream_recursive(next_batch, torch.cuda.current_stream())
            current_batch = next_batch

            if self.stream is not None:
                with torch.cuda.stream(self.stream):
                    next_batch = move_to_device(
                        batch, self.device, channels_last=self.channels_last
                    )
            else:
                next_batch = batch

            yield current_batch

        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            _record_stream_recursive(next_batch, torch.cuda.current_stream())
        yield next_batch


def initialize_device(
    device: str, local_rank: int = -1, use_distributed: bool = False
) -> torch.device:
    """Initialize the device.

    Args:
        device: The device to initialize.
        local_rank: The local rank for distributed training.
        use_distributed: Whether to use distributed training.

    Returns:
        Initialized torch device.
    """
    if device == "gpu":
        if torch.cuda.is_available():
            if use_distributed:
                torch.cuda.set_device(local_rank)
                return torch.device(f"cuda:{local_rank}")
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise ValueError("Device selected as gpu but no GPU or MPS available")

    logger.info("Using device: %s", device)
    return torch.device(device)


def initialize_dtype(dtype: str | torch.dtype | DictConfig | None = None) -> torch.dtype:
    """Resolve a dtype spec from a config to a concrete ``torch.dtype``.

    Accepts:
      - ``None`` — defaults to ``torch.bfloat16``.
      - ``torch.dtype`` — passed through.
      - ``str`` — e.g. ``"bfloat16"``, ``"float16"``, ``"float32"``; resolved
        via ``getattr(torch, ...)``.
      - Hydra config (``DictConfig`` with ``_target_``) — instantiated, e.g.
        ``{_target_: hydra.utils.get_object, _args_: [torch.bfloat16]}``.

    Falls back to ``float32`` when bfloat16 is selected on hardware that
    doesn't support it.
    """
    if dtype is None:
        dtype = torch.bfloat16
    elif isinstance(dtype, str):
        resolved = getattr(torch, dtype, None)
        if not isinstance(resolved, torch.dtype):
            raise ValueError(
                f"dtype string {dtype!r} does not resolve to a torch.dtype "
                f"(e.g. 'bfloat16', 'float16', 'float32')."
            )
        dtype = resolved
    elif not isinstance(dtype, torch.dtype):
        dtype = hy_utils.instantiate(dtype)

    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        logger.warning("BFloat16 is not supported on this device. Using float32 instead.")
        dtype = torch.float32

    logger.info("Using dtype: %s", dtype)
    return dtype


def set_seeds(seed: int) -> None:
    """set the seeds for the experiment.

    Args:
        seed: The seed to set.
    """
    logger.info("Setting seeds to: %s", seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def initialize_dataset(
    dataloader_config: DictConfig,
    local_rank: int = -1,
    world_size: int = 1,
    use_distributed: bool = False,
    device: torch.device | None = None,
    channels_last: bool = False,
) -> dict[str, DataLoader | None]:
    """Initialize the dataset.

    Args:
        dataloader_config: The dataloader config.
        local_rank: The local rank for distributed training.
        world_size: The world size for distributed training.
        use_distributed: Whether to use distributed training.
        device: If provided, wraps each non-None dataloader with CUDAPrefetcher
            to overlap Host-to-Device transfers with compute.
        channels_last: If True, converts 4-D tensors to channels_last format
            when prefetching. Only relevant when device is provided.

    Returns:
        A dict with ``"train"``, ``"val"``, and ``"test"`` dataloaders (wrapped
        in :class:`CUDAPrefetcher` when ``device`` is given; ``"test"`` may be
        ``None``).
    """
    dataloaders = hy_utils.instantiate(dataloader_config)

    # Set up distributed samplers if needed for distributed training.
    # IterableDatasets shard themselves inside __iter__, so DistributedSampler
    # is both unnecessary and incompatible — skip them.
    if use_distributed:

        def _maybe_wrap(loader, *, shuffle):
            if loader is None:
                return None
            if isinstance(loader.dataset, IterableDataset):
                return loader
            return distributed.get_distributed_dataloader(
                loader, local_rank, world_size, shuffle=shuffle
            )

        dataloaders["train"] = _maybe_wrap(dataloaders["train"], shuffle=True)
        dataloaders["val"] = _maybe_wrap(dataloaders["val"], shuffle=False)
        dataloaders["test"] = _maybe_wrap(dataloaders["test"], shuffle=False)

    if device is not None:
        dataloaders["train"] = CUDAPrefetcher(dataloaders["train"], device, channels_last)
        dataloaders["val"] = CUDAPrefetcher(dataloaders["val"], device, channels_last)
        if dataloaders["test"] is not None:
            dataloaders["test"] = CUDAPrefetcher(dataloaders["test"], device, channels_last)

    logger.info("Dataloaders initialized successfully")
    return {"train": dataloaders["train"], "val": dataloaders["val"], "test": dataloaders["test"]}


def initialize_callbacks(config: DictConfig) -> callback_mod.CallbackList:
    """Instantiate callbacks from config and seed initial config values."""
    callbacks = hy_utils.instantiate(config.callbacks)
    if isinstance(callbacks, (dict, DictConfig)):
        callbacks = list(callbacks.values())
    callbacks = callback_mod.CallbackList(callbacks or [])
    callbacks.update_config({"lr": config.lr, "num_epochs": config.num_epochs, "seed": config.seed})
    return callbacks


def extract_class_names(config: Any) -> str | dict[str, Any]:
    """Extract the class names from the config.

    Args:
        config: The config to extract the class names from.

    Returns:
        The class names or a dictionary of class names.
    """
    if isinstance(config, (dict, nn.ModuleDict)):
        return {k: extract_class_names(v) for k, v in config.items()}

    return config.__class__.__name__


def get_num_parameters(model: nn.Module) -> tuple[int, int, int]:
    """Get the number of parameters in the model.

    Args:
        model: The model to get the number of parameters from.

    Returns:
        The number of parameters, the number of trainable parameters,
        and the number of non-trainable parameters.
    """
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_non_trainable_parameters = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return num_parameters, num_trainable_parameters, num_non_trainable_parameters


def _warn_if_capturable_false(
    optimizer: torch.optim.Optimizer | dict[str, torch.optim.Optimizer],
) -> None:
    """Warn when compiling an optimizer whose step counter is a Python integer.

    AdamW, Adam, and similar optimizers support a ``capturable`` param-group
    flag. When ``capturable=False`` (the default), the step counter is stored as
    a Python integer. ``torch.compile`` specialises compiled graphs on Python
    scalar values, so every distinct step count (0, 400, 800, …) that hasn't
    been compiled yet triggers a retrace — typically once per epoch at batch 0.
    Setting ``capturable=True`` moves the step counter to a CUDA tensor, making
    the compiled graph invariant to the step count and eliminating per-epoch
    recompilation.
    """
    opts = list(optimizer.values()) if isinstance(optimizer, dict) else [optimizer]
    for opt in opts:
        if not opt.param_groups:
            continue
        pg = opt.param_groups[0]
        if "capturable" in pg and not pg["capturable"]:
            logger.warning(
                "compile_optimizer_step=True but %s has capturable=False. "
                "torch.compile specialises on the Python-integer step counter, "
                "causing a retrace (and S3 cache lookup) on the first batch of "
                "every epoch. Set capturable=True in optimizer_params to fix this.",
                type(opt).__name__,
            )


def _tensorize_optimizer_lr(
    optimizer: torch.optim.Optimizer | dict[str, torch.optim.Optimizer],
) -> None:
    """Convert each param_group's ``lr`` from a Python float to a 0-dim ``torch.Tensor``.

    This is required when compiling ``optimizer.step`` with ``torch.compile``.  Dynamo guards on
    Python scalars, so any LR scheduler that writes a new float into ``param_groups['lr']`` each
    step (e.g. ``CosineAnnealingLR``, ``StepLR``) would cause a guard failure and retrace on every
    scheduler tick.

    Converting to a tensor makes Dynamo treat the LR as a graph input rather than a specialization
    constant.  PyTorch 2.2+ schedulers use ``_update_param_group_val``, which detects an existing
    tensor and calls ``fill_()`` (in-place update) instead of assignment, so the same tensor object
    persists across all scheduler steps — no new guard, no retrace.

    Must be called before ``torch.compile(optimizer.step, ...)`` because the compiled graph is
    built from the initial trace; if the LR is a float at trace time it will be baked in as a
    constant.
    """
    opts = list(optimizer.values()) if isinstance(optimizer, dict) else [optimizer]
    for opt in opts:
        for pg in opt.param_groups:
            if not isinstance(pg["lr"], torch.Tensor):
                pg["lr"] = torch.tensor(pg["lr"], dtype=torch.float32)
    logger.info("Converted optimizer param_group lr values to tensors for torch.compile")


# Keys consumed by elemental itself; everything else in the compile config is
# forwarded verbatim as kwargs to ``torch.compile`` (so e.g. ``options.*``,
# ``mode``, ``fullgraph``, future PyTorch additions all flow through without
# code changes).
_RESERVED_COMPILE_KEYS = {"enabled", "compile_optimizer_step", "remote_cache"}


def compile_model_and_optimizer_step(  # pylint: disable=too-many-locals
    model: nn.Module,
    optimizer: torch.optim.Optimizer | dict[str, torch.optim.Optimizer] | None = None,
    compile_config: DictConfig | dict | None = None,
) -> nn.Module:
    """Compile model and optimizer for training using torch.compile.

    The compile config is split into two parts:
      * Elemental-managed keys: ``enabled``, ``compile_optimizer_step``, ``remote_cache``.
      * Everything else is forwarded as ``**kwargs`` to ``torch.compile`` — including
        ``mode``, ``backend``, ``fullgraph``, ``dynamic``, ``disable``, and any
        Inductor ``options`` dict. This lets new PyTorch knobs be exposed by
        adding a YAML key, without touching this function.

    Args:
        model: The model to compile.
        optimizer: The optimizer or dict of optimizers. If provided and
            compile_optimizer_step is enabled, optimizer.step will be compiled in-place.
        compile_config: Configuration dict for torch.compile.

    Returns:
        The compiled model if enabled, otherwise the original model.
        Note: optimizer.step is modified in-place if compile_optimizer_step is enabled.
    """
    if compile_config is None:
        return model

    if isinstance(compile_config, DictConfig):
        compile_config = OmegaConf.to_container(compile_config, resolve=True)
    assert isinstance(compile_config, dict)

    if not compile_config.get("enabled", False):
        logger.info("torch.compile is disabled")
        return model

    # Install S3 remote cache backend if configured (must happen before torch.compile).
    # Enabled when config says enabled: true OR when TORCH_COMPILE_S3_REMOTE_CACHE=1 is set.
    # Either way, TORCHINDUCTOR_*_REMOTE_CACHE=1 env vars must also be set for inductor
    # to actually call the backend (see configs/torch_compile.yaml for details).
    remote_cache_config = compile_config.get("remote_cache", None) or {}
    remote_cache_enabled = remote_cache_config.get("enabled", False) or (
        os.environ.get("TORCH_COMPILE_S3_REMOTE_CACHE", "0") == "1"
    )
    if remote_cache_enabled:
        install_s3_cache_backend(
            fx_graph=remote_cache_config.get("fx_graph", True),
            autotune=remote_cache_config.get("autotune", True),
            aot_autograd=remote_cache_config.get("aot_autograd", True),
            pgo=remote_cache_config.get("pgo", False),
            triton=remote_cache_config.get("triton", False),
        )

    compile_optim_step = compile_config.get("compile_optimizer_step", False)

    compile_kwargs: dict[str, Any] = {
        k: v for k, v in compile_config.items() if k not in _RESERVED_COMPILE_KEYS and v is not None
    }
    mode = compile_kwargs.get("mode", "default")

    # torch.compile rejects ``mode`` and ``options`` together (mode is a preset bundle of
    # options). When both are present, expand the mode-preset into options first and merge,
    # with explicit ``options.*`` keys taking precedence — that lets users start from a mode
    # and tweak individual options on top.
    if "mode" in compile_kwargs and "options" in compile_kwargs:
        from torch._inductor import list_mode_options  # pylint: disable=import-outside-toplevel

        mode_options = dict(list_mode_options(compile_kwargs["mode"]))
        explicit_options = compile_kwargs.pop("options")
        compile_kwargs.pop("mode")
        merged = {**mode_options, **explicit_options}
        compile_kwargs["options"] = merged
        logger.info("Merged mode=%s preset with explicit options; final options=%s", mode, merged)

    logger.info("Compiling model with torch.compile kwargs=%s", compile_kwargs)
    compiled_model = torch.compile(model, **compile_kwargs)
    logger.info("Model compiled successfully")

    # Compile optimizer.step in-place if requested.
    # Skip when mode is "reduce-overhead" — CUDA graphs from the compiled
    # model forward and compiled optimizer step conflict, causing
    # "accessing tensor output of CUDAGraphs that has been overwritten" errors.
    if compile_optim_step and optimizer is not None and mode != "reduce-overhead":
        _warn_if_capturable_false(optimizer)
        _tensorize_optimizer_lr(optimizer)
        # ``fullgraph`` on optimizer.step regularly trips on parameter-group iteration;
        # drop it to keep the optimizer compile resilient while preserving mode/backend/options.
        opt_compile_kwargs = {k: v for k, v in compile_kwargs.items() if k != "fullgraph"}
        logger.info("Compiling optimizer.step with torch.compile kwargs=%s", opt_compile_kwargs)

        if isinstance(optimizer, dict):
            for name, opt in optimizer.items():
                opt.step = torch.compile(opt.step, **opt_compile_kwargs)
                logger.info("Compiled optimizer.step for '%s'", name)
        else:
            optimizer.step = torch.compile(optimizer.step, **opt_compile_kwargs)
            logger.info("Compiled optimizer.step successfully")
    elif compile_optim_step and mode == "reduce-overhead":
        logger.warning(
            "Skipping optimizer.step compilation: incompatible with reduce-overhead mode "
            "(CUDA graph conflicts)"
        )

    return compiled_model


def initialize_remote_backup(
    config: DictConfig,
    results_dir: str,
) -> RemoteBackupStrategy | None:
    """Instantiate the remote backup strategy and download a checkpoint if needed.

    This must be called **before** callback initialization so that any
    downloaded ``wandb_run_id.txt`` is available when ``WandbCallback``
    starts.

    Args:
        config: Full Hydra config.  Reads ``remote_backup``,
            ``load_checkpoint_path``, and ``resume_training`` keys.
        results_dir: Fully-qualified results directory for the current run
            (e.g. ``results/autoencoder/exp_name_2026-02-27_12-00-00``).

    Returns:
        The instantiated :class:`RemoteBackupStrategy`, or ``None`` if
        remote backup is not configured.
    """
    if not config.get("remote_backup"):
        return None

    remote_backup: RemoteBackupStrategy = hy_utils.instantiate(config.remote_backup)
    logger.info("Remote backup strategy: %s", type(remote_backup).__name__)

    checkpoint_path_cfg = config.get("load_checkpoint_path")
    if (
        checkpoint_path_cfg
        and str(checkpoint_path_cfg).lower() == "latest"
        and config.get("resume_training", False)
        and distributed.is_main_process()
    ):
        try:
            checkpoint.find_latest_checkpoint(config.results_dir, config.experiment_name)
            logger.info("Local checkpoint found, skipping remote download")
        except FileNotFoundError:
            logger.info("No local checkpoint found, attempting remote download...")
            try:
                remote_backup.download_latest(results_dir)
            except Exception:  # pylint: disable=broad-exception-caught
                logger.warning("Remote backup download failed", exc_info=True)

    return remote_backup


def load_from_checkpoint(
    checkpoint_path: str,
    model: nn.Module | nn.ModuleDict,
    optimizer: torch.optim.Optimizer | dict[str, torch.optim.Optimizer] | None = None,
    scheduler: Any | None = None,
    criterion: nn.Module | None = None,
    device: torch.device = torch.device("cpu"),
    load_keys: list[str] | None = None,
    resume_training: bool = False,
    strict: bool = True,
    weights_only: bool | None = None,
    results_dir: str | None = None,
    experiment_name: str | None = None,
) -> dict[str, Any]:
    """Load model, optimizer, loss, and lr scheduler from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file, or the string ``"latest"``
            to automatically resolve the most recent checkpoint from a previous
            run of the same experiment.
        model: Model to load state into.
        optimizer: Optimizer to load state into (optional).
        scheduler: Scheduler to load state into (optional).
        criterion: Loss function to load state into (optional).
        device: Device to map tensors to.
        load_keys: List of keys to load from checkpoint (e.g., ["model", "optimizer"]).
            If None and resume_training is False, only model will be loaded.
        resume_training: Whether to resume training (loads all states and returns epoch/step).
            If True, loads model, optimizer, scheduler, and criterion.
        strict: Whether to strictly enforce state_dict key matching for model.
        weights_only: If True, only load tensor weights (safer, prevents arbitrary
            code execution). If None (default), automatically set to True when only
            loading model weights, False otherwise. Recommended True for inference/testing.
        results_dir: Root results directory (e.g. ``results/autoencoder/``).
            Required when ``checkpoint_path`` is ``"latest"``.
        experiment_name: Current experiment name including the datetime stamp.
            Required when ``checkpoint_path`` is ``"latest"``.

    Returns:
        Dictionary containing checkpoint metadata (epoch, step, etc.).

    Raises:
        ValueError: If both load_keys and resume_training are provided, or if
            ``checkpoint_path`` is ``"latest"`` but ``results_dir`` or
            ``experiment_name`` are not provided.
    """
    # Resolve "latest" to the most recent run's checkpoint
    if str(checkpoint_path).lower() == "latest":
        if results_dir is None or experiment_name is None:
            raise ValueError(
                "Both 'results_dir' and 'experiment_name' must be provided "
                "when checkpoint_path is 'latest'."
            )
        checkpoint_path = checkpoint.find_latest_checkpoint(
            results_dir=results_dir,
            experiment_name=experiment_name,
        )
        logger.info("Resolved 'latest' checkpoint to: %s", checkpoint_path)
    # Validate that only one parameter is provided
    if load_keys is not None and resume_training:
        raise ValueError("Cannot provide both 'load_keys' and 'resume_training'. Use only one.")

    # Determine what to load
    if resume_training:
        # Load everything when resuming training - need weights_only=False for optimizer state
        logger.info("Resuming training from checkpoint: %s", checkpoint_path)
        checkpoint_data = checkpoint.load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            map_location=device,
            strict=strict,
            weights_only=False,  # Optimizer/scheduler state may contain non-tensor data
        )
        logger.info(
            "Resumed from epoch %s, step %s",
            checkpoint_data.get("epoch"),
            checkpoint_data.get("step"),
        )
        return checkpoint_data

    # Load specific keys or just model by default
    keys_to_load = load_keys if load_keys is not None else ["model"]
    logger.info("Loading checkpoint from: %s (keys: %s)", checkpoint_path, keys_to_load)

    # Create dict to pass only the requested components
    components = {}
    if "model" in keys_to_load:
        components["model"] = model
    if "optimizer" in keys_to_load:
        components["optimizer"] = optimizer
    if "scheduler" in keys_to_load:
        components["scheduler"] = scheduler
    if "criterion" in keys_to_load:
        components["criterion"] = criterion

    # Auto-determine weights_only if not specified:
    # Use weights_only=True when only loading model (safer for inference/testing)
    if weights_only is None:
        weights_only = keys_to_load == ["model"]

    # Load the checkpoint
    checkpoint_data = checkpoint.load_checkpoint(
        checkpoint_path=checkpoint_path,
        map_location=device,
        strict=strict,
        weights_only=weights_only,
        **components,
    )

    return checkpoint_data
