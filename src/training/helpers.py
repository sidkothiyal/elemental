"""Stateless trainer helpers: data-container builders, loss accumulators, profiler setup.

These functions do not depend on any `Trainer` state. They operate only on their
arguments, which makes them independently unit-testable (see `tests/src/test_trainer.py`).
"""

import inspect
import logging
import os
from typing import Any, Callable

import hydra.utils as hy_utils
import torch
from omegaconf import DictConfig
from torch import nn
from torch.profiler import ProfilerActivity, schedule, tensorboard_trace_handler

logger = logging.getLogger(__name__)

_signature_cache: dict[Callable, inspect.Signature] = {}


def _get_cached_signature(fn: Callable) -> inspect.Signature:
    """Return inspect.signature(fn), cached to avoid repeated introspection."""
    sig = _signature_cache.get(fn)
    if sig is None:
        sig = inspect.signature(fn)
        _signature_cache[fn] = sig
    return sig


def collect_module_losses(
    model: nn.Module,
    dataset: object | None = None,
    pred: torch.Tensor | None = None,
    target: torch.Tensor | None = None,
    model_input_args: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Trigger compute_losses hooks and collect losses from model submodules and dataset.

    For each object (model submodules + dataset), this function:
    1. Calls ``compute_losses()`` if present, passing whichever of ``pred``, ``target``,
       and ``model_input_args`` values match its signature via introspection.
    2. Collects entries from the ``losses`` dict attribute, if present.

    This allows nested modules (e.g., VQVAE inside PatcherUnpatcher) to contribute
    their custom losses to the training loop, and allows datasets (e.g., DiffusionDataset)
    to update internal state such as per-timestep loss histories.

    Args:
        model: The root model to collect losses from.
        dataset: Optional dataset object (e.g., dataloader.dataset). If it exposes
                ``compute_losses`` or ``losses``, they will be handled.
        pred: Model predictions tensor, passed to ``compute_losses`` if in its signature.
        target: Target tensor, passed to ``compute_losses`` if in its signature.
        model_input_args: Dict of extra batch tensors (e.g. ``{"t": ..., "cond": ...}``).
                         Values whose keys match ``compute_losses`` parameter names are passed.

    Returns:
        Dictionary mapping loss names to loss tensors. Names are prefixed with
        the module path to avoid collisions (e.g., "model/dictionary_loss").
    """
    all_losses = {}

    available_kwargs: dict[str, torch.Tensor] = {}
    if pred is not None:
        available_kwargs["pred"] = pred
    if target is not None:
        available_kwargs["target"] = target
    if model_input_args:
        available_kwargs.update(model_input_args)

    objects_to_check: list[tuple[str, object]] = list(model.named_modules())
    if dataset is not None:
        objects_to_check.append(("dataset", dataset))

    for name, obj in objects_to_check:
        if hasattr(obj, "compute_losses") and available_kwargs:
            sig = _get_cached_signature(obj.compute_losses)
            matched_kwargs = {k: v for k, v in available_kwargs.items() if k in sig.parameters}
            obj.compute_losses(**matched_kwargs)

        if hasattr(obj, "losses") and obj.losses:
            for loss_name, loss_value in obj.losses.items():
                full_name = f"{name}/{loss_name}" if name else loss_name
                all_losses[full_name] = loss_value

    return all_losses


def clear_module_losses(model: nn.Module) -> None:
    """Clear losses from all submodules that have a 'losses' attribute."""
    for module in model.modules():
        if hasattr(module, "losses"):
            module.losses.clear()


def setup_profiler(
    profiler_config: dict | DictConfig | None,
    results_dir: str,
) -> torch.profiler.profile | None:
    """Create a PyTorch profiler from config.

    Args:
        profiler_config: Profiler configuration dict with keys:
            - enabled (bool): Whether profiling is active.
            - activities (list[str]): Which activities to trace ("CPU", "CUDA").
            - schedule (dict): Schedule params (wait, warmup, active, repeat).
            - record_shapes (bool): Capture input tensor shapes per op.
            - profile_memory (bool): Track tensor allocations/deallocations.
            - with_stack (bool): Capture Python call stack for each op.
            - with_flops (bool): Estimate FLOPs for matmul/conv ops.
            - output (dict): Output settings (print_table, export_chrome_trace,
              tensorboard, table_sort_by, table_row_limit).
        results_dir: Directory to save profiler outputs.

    Returns:
        Configured profiler instance, or None if disabled.
    """
    if profiler_config is None:
        return None

    if isinstance(profiler_config, DictConfig):
        profiler_config = dict(profiler_config)

    if not profiler_config.get("enabled", False):
        logger.info("Profiler is disabled")
        return None

    activity_map = {
        "CPU": ProfilerActivity.CPU,
        "CUDA": ProfilerActivity.CUDA,
    }
    activities = [
        activity_map[a] for a in profiler_config.get("activities", ["CPU"]) if a in activity_map
    ]

    sched_cfg = profiler_config.get("schedule", {})
    if isinstance(sched_cfg, DictConfig):
        sched_cfg = dict(sched_cfg)
    prof_schedule = schedule(
        wait=sched_cfg.get("wait", 1),
        warmup=sched_cfg.get("warmup", 1),
        active=sched_cfg.get("active", 3),
        repeat=sched_cfg.get("repeat", 1),
    )

    output_cfg = profiler_config.get("output", {})
    if isinstance(output_cfg, DictConfig):
        output_cfg = dict(output_cfg)

    profiler_dir = os.path.join(results_dir, "profiler")
    os.makedirs(profiler_dir, exist_ok=True)

    on_trace_ready = None
    if output_cfg.get("tensorboard", False):
        tb_dir = os.path.join(profiler_dir, "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        on_trace_ready = tensorboard_trace_handler(tb_dir)
        logger.info("Profiler TensorBoard traces will be saved to: %s", tb_dir)

    def trace_handler(prof):
        if output_cfg.get("print_table", True):
            sort_by = output_cfg.get("table_sort_by", "cpu_time_total")
            row_limit = output_cfg.get("table_row_limit", 20)
            logger.info(
                "\n%s",
                prof.key_averages().table(sort_by=sort_by, row_limit=row_limit),
            )

        # tensorboard_trace_handler internally calls export_chrome_trace(),
        # and PyTorch only allows one export per profiling cycle.
        if on_trace_ready is not None:
            on_trace_ready(prof)
        elif output_cfg.get("export_chrome_trace", True):
            trace_path = os.path.join(profiler_dir, f"trace_step_{prof.step_num}.json")
            prof.export_chrome_trace(trace_path)
            logger.info("Chrome trace exported to: %s", trace_path)

    profiler = torch.profiler.profile(
        activities=activities,
        schedule=prof_schedule,
        on_trace_ready=trace_handler,
        record_shapes=profiler_config.get("record_shapes", False),
        profile_memory=profiler_config.get("profile_memory", False),
        with_stack=profiler_config.get("with_stack", False),
        with_flops=profiler_config.get("with_flops", False),
    )

    logger.info(
        "Profiler configured: activities=%s, schedule(wait=%d, warmup=%d, active=%d, repeat=%d)",
        [a.name for a in activities],
        sched_cfg.get("wait", 1),
        sched_cfg.get("warmup", 1),
        sched_cfg.get("active", 3),
        sched_cfg.get("repeat", 1),
    )

    return profiler


def cleanup_memory(
    model: nn.Module,
    optim: torch.optim.Optimizer | dict[str, torch.optim.Optimizer],
    device: torch.device,
    epoch: int | None = None,
    empty_cache_interval: int = 0,
) -> None:
    """Clean up GPU memory: clear gradients, module losses, and optionally the CUDA cache.

    Args:
        model: The model to clear losses from.
        optim: Optimizer or dict of optimizers to clear gradients from.
        device: The torch device being used.
        epoch: Current epoch number, used with empty_cache_interval.
        empty_cache_interval: Interval (in epochs) at which to call
            torch.cuda.empty_cache(). 0 disables it.
    """
    if isinstance(optim, dict):
        for opt in optim.values():
            opt.zero_grad(set_to_none=False)
    else:
        optim.zero_grad(set_to_none=False)

    clear_module_losses(model)

    # torch.cuda.empty_cache() is a synchronization point that stalls the GPU
    # pipeline; use only for large models close to OOM (interval=0 disables).
    if device.type == "cuda" and empty_cache_interval > 0:
        if epoch is None or epoch % empty_cache_interval == 0:
            torch.cuda.empty_cache()


def build_data_container(
    batch: dict,
    data: dict | None = None,
    model: nn.Module | None = None,
    epoch: int | None = None,
) -> dict:
    """Build a unified namespaced data container from a batch dict.

    Every key except ``"target"`` is prefixed with ``"input/"``
    (e.g. ``batch["input"]`` -> ``"input/input"``).  The ``"target"``
    key is kept as-is.

    Optionally stores the *model* reference (under ``"model"``) and
    the current *epoch* (under ``"epoch"``) so that losses
    and post-processors can resolve them via their input / context
    mappings.

    Args:
        batch: Raw batch dictionary from the dataloader.
        data: Optional existing dict to clear and update in-place. If None,
            a new dict is created.
        model: Optional model instance to store in the container.
        epoch: Optional current epoch number.

    Returns:
        The populated dict (same object as *data* when provided).
    """
    if data is None:
        data = {}
    else:
        data.clear()
    for key, value in batch.items():
        if key == "target":
            data["target"] = value
        else:
            data[f"input/{key}"] = value
    if model is not None:
        data["model"] = model
    if epoch is not None:
        data["epoch"] = epoch
    return data


def populate_predictions(pred: torch.Tensor | dict[str, torch.Tensor], data: dict) -> dict:
    """Populate the data container with model predictions.

    If *pred* is a dict (e.g. from ``DenoisingDiffusionModel``), each key
    is stored under ``pred/<key>`` (e.g. ``pred/pred_noise``,
    ``pred/pred_x0``).  The raw dict is also kept at ``pred`` for
    introspection.

    If *pred* is a plain tensor, it is stored at ``pred``.

    Args:
        pred: Model output — a single tensor or a dict of tensors.
        data: The unified data container to populate.

    Returns:
        The same *data* dict, now containing prediction entries.
    """
    if isinstance(pred, dict):
        data["pred"] = pred
        for key, value in pred.items():
            data[f"pred/{key}"] = value
    else:
        data["pred"] = pred
    return data


def should_step_optimizer(batch_idx: int, grad_accum_steps: int, total_batches: int | None) -> bool:
    """Return True when the optimizer should step (accumulation boundary or final batch).

    ``total_batches`` is ``None`` for ``IterableDataset`` loaders whose length is
    unknown. In that case the tail-flush branch is skipped — we step only on
    accumulation boundaries.
    """
    on_boundary = (batch_idx + 1) % grad_accum_steps == 0
    on_final = total_batches is not None and (batch_idx + 1) == total_batches
    return on_boundary or on_final


def backward_step(
    loss: torch.Tensor,
    scaler: torch.amp.GradScaler | None,
    grad_accum_steps: int,
) -> None:
    """Scale *loss* by ``1 / grad_accum_steps`` and run backward.

    When *scaler* is provided and enabled, uses ``scaler.scale(...)`` to handle
    fp16 gradients; otherwise calls ``.backward()`` directly.
    """
    scaled_loss = loss / grad_accum_steps
    if scaler is not None and scaler.is_enabled():
        scaler.scale(scaled_loss).backward()
    else:
        scaled_loss.backward()


def optimizer_step(
    *,
    optim: torch.optim.Optimizer,
    parameters: Any,
    scaler: torch.amp.GradScaler | None,
    clip_grad_norm: float | None,
) -> float | None:
    """Run an optimizer step: unscale (if scaled) → clip → step → scaler.update → zero_grad.

    Callers must gate this with ``should_step_optimizer(...)`` to respect
    gradient accumulation boundaries.

    Returns the total (pre-clip) gradient norm when clipping is enabled — the
    ``clip_grad_norm_`` call computes it anyway, so it is free to surface for
    logging — otherwise ``None``.
    """
    use_scaler = scaler is not None and scaler.is_enabled()
    grad_norm: float | None = None
    if clip_grad_norm is not None:
        if use_scaler:
            assert scaler is not None  # mypy narrowing
            scaler.unscale_(optim)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, max_norm=clip_grad_norm))
    if use_scaler:
        assert scaler is not None  # mypy narrowing
        scaler.step(optim)
        scaler.update()
    else:
        optim.step()
    optim.zero_grad(set_to_none=False)
    return grad_norm


def build_optim_dict_or_single(
    config: DictConfig,
    model: nn.Module,
) -> torch.optim.Optimizer | dict[str, torch.optim.Optimizer]:
    """Build an optimizer — a dict keyed by submodule name when *model* is a ModuleDict.

    Three branches in priority order:

    1. ``config.optimizer_groups`` is set → return a :class:`MultiOptimizer` that
       partitions ``model``'s parameters by predicate (e.g. nanochat's Muon +
       AdamW recipe). Each named group instantiates one sub-optimizer.
    2. ``model`` is an ``nn.ModuleDict`` → return one optimizer per submodule
       (used by GAN / customizable trainers).
    3. Otherwise → a single ``torch.optim.<X>`` over all trainable params.

    Reads ``config.optimizer`` (class name), ``config.optimizer_params`` (kwargs),
    and ``config.lr``.
    """
    if "optimizer_groups" in config and config.optimizer_groups is not None:
        return _build_multi_optimizer(config, model)

    optimizer_class = getattr(torch.optim, config.get("optimizer", "AdamW"))
    optimizer_params = dict(config.get("optimizer_params", {}) or {})
    optimizer_params = {k: v for k, v in optimizer_params.items() if v is not None}

    if isinstance(model, nn.ModuleDict):
        return {
            name: optimizer_class(
                [p for p in submodule.parameters() if p.requires_grad],
                lr=config.lr,
                **optimizer_params,
            )
            for name, submodule in model.items()
        }

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return optimizer_class(trainable_params, lr=config.lr, **optimizer_params)


def _build_multi_optimizer(config: DictConfig, model: nn.Module) -> Any:
    """Construct a MultiOptimizer from ``config.optimizer_groups`` + an optional partition spec.

    ``config.optimizer_partition`` selects the predicate set. Currently the only
    built-in is ``"nanochat_default"`` (Muon for 2-D matrix block params, AdamW
    for everything else). Custom partitions can be supplied as a Hydra
    ``_target_:`` pointing at a callable that returns ``dict[str, predicate]``.
    """
    # Local import to keep helpers.py free of optimizer-package side-effects at import time.
    from optimizer.multi_optimizer import MultiOptimizer  # pylint: disable=import-outside-toplevel
    from optimizer.param_groups import (  # pylint: disable=import-outside-toplevel
        nanochat_default,
        partition_params,
    )

    partition_cfg = config.get("optimizer_partition", "nanochat_default")
    if partition_cfg == "nanochat_default":
        predicates = nanochat_default()
    else:
        predicates = hy_utils.instantiate(partition_cfg)

    partitions = partition_params(model, predicates)

    sub_optimizers: dict = {}
    for group_name, opt_cfg in config.optimizer_groups.items():
        if group_name not in partitions:
            raise ValueError(
                f"optimizer_groups defines '{group_name}' but partition does not "
                f"(known: {list(partitions)})"
            )
        params = partitions[group_name]
        if not params:
            logger.warning("Optimizer group %s has no parameters; skipping", group_name)
            continue
        sub_optimizers[group_name] = hy_utils.instantiate(opt_cfg, params=params)

    default_group = config.get("optimizer_default_group", None)
    return MultiOptimizer(sub_optimizers, default_group=default_group)


def build_scheduler_dict_or_single(
    config: DictConfig,
    optim: torch.optim.Optimizer | dict[str, torch.optim.Optimizer],
) -> torch.optim.lr_scheduler.LRScheduler | dict[str, torch.optim.lr_scheduler.LRScheduler] | None:
    """Build an LR scheduler — a dict when *optim* is a dict, else a single scheduler.

    Two config shapes accepted:

    1. **Legacy** (``config.lr_scheduler`` is a string): looks up the class on
       ``torch.optim.lr_scheduler`` and instantiates with ``lr_scheduler_params``.
    2. **Target** (``config.lr_scheduler`` is a dict with ``_target_``): uses
       Hydra ``instantiate`` and passes the optimizer in as ``optimizer=``.
       Required for custom schedulers (e.g. warmup-cosine).

    Returns None when no ``lr_scheduler`` is configured.
    """
    sched_cfg = config.get("lr_scheduler", None)
    if sched_cfg is None:
        return None

    if isinstance(sched_cfg, str):
        scheduler_class = getattr(torch.optim.lr_scheduler, sched_cfg)
        scheduler_params = dict(config.get("lr_scheduler_params", {}) or {})
        scheduler_params = {k: v for k, v in scheduler_params.items() if v is not None}
        if isinstance(optim, dict):
            return {name: scheduler_class(o, **scheduler_params) for name, o in optim.items()}
        return scheduler_class(optim, **scheduler_params)

    # Target-style: Hydra-instantiated.
    if isinstance(optim, dict):
        return {name: hy_utils.instantiate(sched_cfg, optimizer=o) for name, o in optim.items()}
    return hy_utils.instantiate(sched_cfg, optimizer=optim)


def materialize_and_average_losses(
    running_loss_gpu: dict[str, torch.Tensor],
    total_batches: int,
) -> dict[str, float]:
    """Materialize GPU loss tensors to CPU and divide by batch count."""
    denom = max(total_batches, 1)
    return {key: value.item() / denom for key, value in running_loss_gpu.items()}


def _accumulate_loss(
    running_loss: dict[str, torch.Tensor],
    key: str,
    tensor: torch.Tensor,
) -> None:
    """Accumulate a detached loss tensor into the running loss dict.

    Clones on first insertion to avoid holding a reference to the full
    computation graph; subsequent calls add in-place to minimize allocations.
    """
    if key in running_loss:
        running_loss[key] += tensor
    else:
        running_loss[key] = tensor.clone()
