"""Builder for Trainer instances.

``build_trainer`` composes the Hydra config, sets up environment (seeds, device,
DDP), instantiates dataloaders, callbacks, and remote backup, then dispatches to
the appropriate ``Trainer`` subclass whose ``_build_*`` classmethods construct
model, optimizer, scheduler, loss, and scaler. Finally it loads any checkpoint,
compiles the model, wires post-processor and profiler, and returns a ready-to-run
``Trainer`` instance.
"""
# build_trainer's job is to call the Trainer subclasses' internal _build_* hooks.
# pylint: disable=protected-access

import logging
import os
import shutil
from typing import Type

import hydra
import hydra.utils as hy_utils
import torch
from omegaconf import DictConfig, open_dict

from training.customizable_trainer import CustomizableTrainer
from training.gan_trainer import GANTrainer
from training.graph_trainer import GraphTrainer
from training.helpers import setup_profiler
from training.rl_trainer import RLTrainer
from training.trainer import Trainer
from utils import distributed, initializer

logger = logging.getLogger(__name__)


def _attach_file_logging(results_dir: str) -> None:
    """Mirror all logs to ``<results_dir>/run.log`` and log free disk space.

    Long unattended runs lose their console scrollback, and wandb's stdout/stderr
    redirect can mangle a late traceback — so a crash (disk full, CUDA OOM, ...)
    can vanish without a record. A root-level FileHandler guarantees the final
    traceback (logged by the entrypoint) lands on disk. Also surfaces free space
    up front, since a near-full disk is a common silent killer.
    """
    log_path = os.path.abspath(os.path.join(results_dir, "run.log"))
    root = logging.getLogger()
    # Idempotent: don't double-attach if build_trainer runs twice in a process.
    already = any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == log_path
        for h in root.handlers
    )
    if not already:
        handler = logging.FileHandler(log_path)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        root.addHandler(handler)
        logger.info("Mirroring logs to: %s", log_path)

    usage = shutil.disk_usage(results_dir)
    free_gb, total_gb = usage.free / 1e9, usage.total / 1e9
    log = logger.warning if free_gb < 20 else logger.info
    log("Disk space at %s: %.1f GB free / %.1f GB total", results_dir, free_gb, total_gb)


def _trainer_class_for(mode: str) -> Type[Trainer]:
    """Return the Trainer subclass for a given config.mode string."""
    return {
        "gan": GANTrainer,
        "rl": RLTrainer,
        "customizable": CustomizableTrainer,
    }.get(mode, Trainer)


def _init_env(config: DictConfig, local_rank: int) -> tuple:
    """Set up distributed, seeds, TF32, device, dtype, and results_dir.

    Returns:
        (device, dtype, use_distributed, world_size, cuda_empty_cache_interval,
         results_dir)
    """
    # Set up distributed training if requested
    use_distributed = config.get("distributed", False) and local_rank >= 0
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if use_distributed:
        backend = "nccl" if __import__("torch").cuda.is_available() else "gloo"
        distributed.setup_distributed(local_rank, world_size, backend)

    # Allow TF32 for float32 matmuls on Ampere+ GPUs (A100, RTX 30xx, etc.).
    # This is a no-op when using bfloat16 autocast, but speeds up any float32
    # operations (e.g. loss computation, metrics) by using TF32 tensor cores.
    torch.set_float32_matmul_precision("high")

    # Set up environment: set seeds, initialize device, create results directory
    initializer.set_seeds(config.seed)
    if distributed.is_main_process():
        logger.info("Configuration: %s", config)

    device = initializer.initialize_device(config.device, local_rank, use_distributed)
    dtype = initializer.initialize_dtype(config.get("dtype"))
    cuda_empty_cache_interval = config.get("cuda_empty_cache_interval", 0)

    results_dir = os.path.join(config.results_dir, config.experiment_name)
    if distributed.is_main_process():
        os.makedirs(results_dir, exist_ok=True)
        logger.info("Results will be saved to: %s", results_dir)
        _attach_file_logging(results_dir)

    return (
        device,
        dtype,
        use_distributed,
        world_size,
        cuda_empty_cache_interval,
        results_dir,
    )


def _build_profiler(config: DictConfig, results_dir: str) -> tuple:
    """Build the profiler; return (profiler, profiler_step_per)."""
    profiler = setup_profiler(config.get("profiler", None), results_dir)
    profiler_step_per = "batch"
    if profiler is not None:
        profiler.start()
        prof_cfg = config.get("profiler", {})
        profiler_step_per = prof_cfg.get("step_per", "batch")
    return profiler, profiler_step_per


def _maybe_load_checkpoint(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    config: DictConfig,
    model,
    optim,
    scheduler,
    loss,
    scaler,
    device,
) -> dict:
    """Load from checkpoint if configured, returning epoch/step metadata."""
    checkpoint_data = {"epoch": None, "step": None}
    checkpoint_path = config.get("load_checkpoint_path", None)
    if checkpoint_path is None:
        return checkpoint_data

    load_keys = config.get("load_keys", None)
    resume_training = config.get("resume_training", False)
    strict = config.get("strict", True)

    return initializer.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optim,
        scheduler=scheduler,
        criterion=loss,
        scaler=scaler,
        device=device,
        load_keys=load_keys,
        resume_training=resume_training,
        strict=strict,
        results_dir=config.results_dir,
        experiment_name=config.experiment_name,
    )


def build_trainer(  # pylint: disable=too-many-locals
    config_file: str,
    local_rank: int = -1,
    overrides: list[str] | None = None,
    no_remote_logging: bool = False,
) -> Trainer | GraphTrainer:
    """Build and return a fully initialized ``Trainer`` (or subclass) instance.

    Args:
        config_file: Hydra config name (e.g. ``"experiment/autoencoder/compression_ae"``).
        local_rank: Local rank for DDP (-1 for single-GPU).
        overrides: Hydra override strings (e.g. ``["lr=0.001", "num_epochs=100"]``).

    Returns:
        A configured ``Trainer`` instance with ``.train()`` / ``.test()`` ready to call.
        For ``mode: graph``, returns a ``GraphTrainer`` instead.
    """
    with hydra.initialize(version_base=None, config_path="../../configs"):
        config = hydra.compose(config_name=config_file, overrides=overrides or [])

    if no_remote_logging:
        with open_dict(config):
            if config.get("remote_backup") is not None:
                config.remote_backup = None
            callbacks_cfg = config.get("callbacks")
            if callbacks_cfg is not None and "wandb" in callbacks_cfg:
                config.callbacks.wandb = None
        logger.info("Remote logging disabled: cleared remote_backup and callbacks.wandb")

    # Graph mode uses a completely different initialization path.
    if config.get("mode", None) == "graph":
        return GraphTrainer(config)

    (
        device,
        dtype,
        use_distributed,
        world_size,
        cuda_empty_cache_interval,
        results_dir,
    ) = _init_env(config, local_rank)

    # Remote backup must be initialized BEFORE callbacks so WandbCallback finds
    # the downloaded run ID.
    remote_backup = initializer.initialize_remote_backup(config, results_dir)

    callbacks = initializer.initialize_callbacks(config)
    # RL experiments omit `dataloader:` — env + rollout buffer replace the supervised loader.
    dataloader_cfg = config.get("dataloader", None)
    if dataloader_cfg is not None:
        loaders = initializer.initialize_dataset(
            dataloader_cfg,
            local_rank,
            world_size,
            use_distributed,
            device=device,
            channels_last=config.get("channels_last", True),
        )
    else:
        loaders = {}

    cls = _trainer_class_for(config.mode)

    model = cls._build_model(config, device, use_distributed)
    (
        num_parameters,
        num_trainable_parameters,
        num_non_trainable_parameters,
    ) = initializer.get_num_parameters(model)
    if distributed.is_main_process():
        logger.info("Number of parameters: %d", num_parameters)
        logger.info("Number of trainable parameters: %d", num_trainable_parameters)
        logger.info("Number of non-trainable parameters: %d", num_non_trainable_parameters)

    optim = cls._build_optimizer(config, model)
    scheduler = cls._build_scheduler(config, optim)
    loss = cls._build_loss(config, device)
    scaler = cls._build_scaler(config, device, dtype)

    checkpoint_data = _maybe_load_checkpoint(config, model, optim, scheduler, loss, scaler, device)

    # Compile after checkpoint load so compiled graph is built against loaded weights.
    compile_config = config.get("torch_compile", None)
    model = initializer.compile_model_and_optimizer_step(model, optim, compile_config)

    pp_config = config.get("postprocessors", None)
    postprocessor_pipeline = hy_utils.instantiate(pp_config) if pp_config is not None else None

    profiler, profiler_step_per = _build_profiler(config, results_dir)

    callbacks.update_config(
        {
            "num_parameters": num_parameters,
            "num_trainable_parameters": num_trainable_parameters,
            "num_non_trainable_parameters": num_non_trainable_parameters,
            "config": config,
            "dtype": dtype,
            "scheduler": scheduler,
            "criterion": loss,
            "model": model,
            "optimizer": optim,
            "scaler": scaler,
            "device": device,
            "mode": config.mode,
            "remote_backup": remote_backup,
            "to_log": config.get("to_log", None),
            "train_loader": loaders.get("train"),
        }
    )

    resume_training = config.get("resume_training", False)

    extra_kwargs: dict = {}
    if cls is CustomizableTrainer:
        extra_kwargs["graph"] = cls._build_graph(config)
    if cls is RLTrainer:
        extra_kwargs["env"] = cls._build_env(config, device)
        extra_kwargs["policy_wrapper"] = cls._build_policy_wrapper(config, model, device)
        extra_kwargs["rollout_buffer"] = cls._build_rollout_buffer(config, device)
        extra_kwargs["rl_algorithm"] = cls._build_algorithm(config)
        cls._attach_policy_params(optim, extra_kwargs["policy_wrapper"], config)

    return cls(
        config=config,
        device=device,
        dtype=dtype,
        model=model,
        optim=optim,
        loss=loss,
        scheduler=scheduler,
        scaler=scaler,
        callbacks=callbacks,
        loaders=loaders,
        postprocessor_pipeline=postprocessor_pipeline,
        profiler=profiler,
        profiler_step_per=profiler_step_per,
        results_dir=results_dir,
        use_distributed=use_distributed,
        checkpoint_data=checkpoint_data,
        cuda_empty_cache_interval=cuda_empty_cache_interval,
        resume_training=resume_training,
        **extra_kwargs,
    )
