"""Utilities for distributed training with PyTorch DDP."""

import logging
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger(__name__)


def setup_distributed(rank: int, world_size: int, backend: str = "nccl") -> None:
    """Initialize distributed training environment.

    Args:
        rank: The rank of the current process.
        world_size: Total number of processes.
        backend: Backend to use ('nccl' for GPU, 'gloo' for CPU).
    """
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12355")

    # Bind this process to its own GPU BEFORE initializing the process group.
    # NCCL otherwise defaults every rank to cuda:0, which trips the
    # "Duplicate GPU detected" check when DDP runs its first collective.
    if backend == "nccl" and torch.cuda.is_available():
        torch.cuda.set_device(rank)

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    logger.info("Initialized distributed training: rank %d/%d", rank, world_size)


def cleanup_distributed() -> None:
    """Clean up distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_model_ddp(
    model: nn.Module | nn.ModuleDict,
    device: torch.device,
    find_unused_parameters: bool = False,
) -> nn.Module | nn.ModuleDict:
    """Wrap model with DistributedDataParallel if distributed training is active.

    Args:
        model: The model to wrap. Can be single model or ModuleDict (for GAN).
        device: The device the model is on.
        find_unused_parameters: Whether to find unused parameters during backward.

    Returns:
        DDP-wrapped model if distributed, otherwise original model.
    """
    if not dist.is_initialized():
        return model

    if isinstance(model, nn.ModuleDict):
        # Wrap each sub-model in the dict
        wrapped = nn.ModuleDict()
        for key, submodel in model.items():
            wrapped[key] = DDP(
                submodel,
                device_ids=[device.index] if device.type == "cuda" else None,
                find_unused_parameters=find_unused_parameters,
            )
        return wrapped

    return DDP(
        model,
        device_ids=[device.index] if device.type == "cuda" else None,
        find_unused_parameters=find_unused_parameters,
    )


def get_distributed_dataloader(
    dataloader: DataLoader,
    rank: int,
    world_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """Create a distributed sampler for the dataloader.

    Args:
        dataloader: The original dataloader.
        rank: Current process rank.
        world_size: Total number of processes.

    Returns:
        DataLoader with DistributedSampler.
    """
    sampler = DistributedSampler(
        dataloader.dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
    )

    return DataLoader(
        dataloader.dataset,
        batch_size=dataloader.batch_size,
        sampler=sampler,
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory,
        drop_last=dataloader.drop_last,
    )


def is_main_process() -> bool:
    """Check if current process is the main process (rank 0 or not distributed)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size() -> int:
    """Number of distributed processes (1 when not distributed)."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()
