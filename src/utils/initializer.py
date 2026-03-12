"""Utility functions for initializing the experiment."""

import logging
import random
from typing import Any

import hydra.utils as hy_utils
import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

from utils import checkpoint, distributed  # pylint: disable=no-name-in-module
from utils.remote_backup.base import RemoteBackupStrategy  # pylint: disable=no-name-in-module

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


def initialize_dtype(dtype: str | None = None) -> torch.dtype:
    """Initialize the dtype.

    Args:
        dtype: The dtype to initialize.
    """
    dtype = dtype or torch.bfloat16
    if not isinstance(dtype, torch.dtype):
        dtype = hy_utils.instantiate(dtype)

    if dtype is torch.bfloat16:
        if not torch.cuda.is_bf16_supported():
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
) -> tuple[DataLoader, DataLoader, DataLoader]:
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
        The train, val, and test dataloaders (wrapped in CUDAPrefetcher when
        device is given).
    """
    dataloaders = hy_utils.instantiate(dataloader_config)

    # Set up distributed samplers if needed for distributed training
    if use_distributed:
        dataloaders["train"] = distributed.get_distributed_dataloader(
            dataloaders["train"], local_rank, world_size, shuffle=True
        )
        dataloaders["val"] = distributed.get_distributed_dataloader(
            dataloaders["val"], local_rank, world_size, shuffle=False
        )
        dataloaders["test"] = (
            distributed.get_distributed_dataloader(
                dataloaders["test"], local_rank, world_size, shuffle=False
            )
            if dataloaders["test"] is not None
            else None
        )

    if device is not None:
        dataloaders["train"] = CUDAPrefetcher(dataloaders["train"], device, channels_last)
        dataloaders["val"] = CUDAPrefetcher(dataloaders["val"], device, channels_last)
        if dataloaders["test"] is not None:
            dataloaders["test"] = CUDAPrefetcher(dataloaders["test"], device, channels_last)

    return dataloaders["train"], dataloaders["val"], dataloaders["test"]


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


def compile_model_and_optimizer_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | dict[str, torch.optim.Optimizer] | None = None,
    compile_config: DictConfig | dict | None = None,
) -> nn.Module:
    """Compile model and optimizer for training using torch.compile.

    Args:
        model: The model to compile.
        optimizer: The optimizer or dict of optimizers. If provided and
            compile_optimizer_step is enabled, optimizer.step will be compiled in-place.
        compile_config: Configuration dict for torch.compile with the following options:
            - enabled (bool): Whether to enable compilation. Default: False.
            - mode (str): Compilation mode. Options: "default", "reduce-overhead",
              "max-autotune", "max-autotune-no-cudagraphs". Default: "default".
            - backend (str): Backend to use. Default: "inductor".
            - fullgraph (bool): Whether to require full graph compilation. Default: False.
            - dynamic (bool): Whether to use dynamic shapes. Default: None.
            - disable (bool): Whether to disable dynamo tracing. Default: False.
            - compile_optimizer_step (bool): Whether to compile optimizer.step. Default: False.

    Returns:
        The compiled model if enabled, otherwise the original model.
        Note: optimizer.step is modified in-place if compile_optimizer_step is enabled.
    """
    if compile_config is None:
        return model

    # Convert DictConfig to dict if needed
    if isinstance(compile_config, DictConfig):
        compile_config = dict(compile_config)

    enabled = compile_config.get("enabled", False)
    if not enabled:
        logger.info("torch.compile is disabled")
        return model

    # Extract compile options
    mode = compile_config.get("mode", "default")
    backend = compile_config.get("backend", "inductor")
    fullgraph = compile_config.get("fullgraph", False)
    dynamic = compile_config.get("dynamic", None)
    disable = compile_config.get("disable", False)
    compile_optim_step = compile_config.get("compile_optimizer_step", False)

    # Compile model
    logger.info(
        "Compiling model with torch.compile (mode=%s, backend=%s, fullgraph=%s, dynamic=%s)",
        mode,
        backend,
        fullgraph,
        dynamic,
    )

    compiled_model = torch.compile(
        model,
        mode=mode,
        backend=backend,
        fullgraph=fullgraph,
        dynamic=dynamic,
        disable=disable,
    )
    logger.info("Model compiled successfully")

    # Compile optimizer.step in-place if requested.
    # Skip when mode is "reduce-overhead" — CUDA graphs from the compiled
    # model forward and compiled optimizer step conflict, causing
    # "accessing tensor output of CUDAGraphs that has been overwritten" errors.
    if compile_optim_step and optimizer is not None and mode != "reduce-overhead":
        logger.info(
            "Compiling optimizer.step with torch.compile (mode=%s, backend=%s)", mode, backend
        )

        if isinstance(optimizer, dict):
            for name, opt in optimizer.items():
                opt.step = torch.compile(opt.step, mode=mode, backend=backend)
                logger.info("Compiled optimizer.step for '%s'", name)
        else:
            optimizer.step = torch.compile(optimizer.step, mode=mode, backend=backend)
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
