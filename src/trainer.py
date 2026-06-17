"""Hydra CLI entrypoint for training.

Delegates to :func:`training.builder.build_trainer` which constructs the
appropriate :class:`training.trainer.Trainer` subclass based on ``config.mode``.

Exit codes (consumed by the autoresearch agent):
  0  — successful run
  1  — unexpected error
  10 — CUDA out-of-memory
  11 — torch.compile / dynamo / inductor error
  12 — NaN loss (training diverged)
  13 — timeout (max_wall_time_sec exceeded)
"""

import logging
import os
import sys

import torch
import typer

from training.builder import build_trainer

logger = logging.getLogger(__name__)


def _classify_exit_code(exc: BaseException) -> int:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return 10
    module = type(exc).__module__ or ""
    if "dynamo" in module or "inductor" in module or "torch._compile" in module:
        return 11
    if type(exc).__name__ == "NaNLossError":
        return 12
    if isinstance(exc, TimeoutError):
        return 13
    return 1


def _main(
    config_file: str = typer.Argument(
        ..., help="Hydra config name (e.g. experiment/autoencoder/compression_ae)"
    ),
    overrides: list[str] = typer.Argument(
        default=None, help="Hydra overrides (e.g. lr=0.001 num_epochs=500)"
    ),
    test: bool = typer.Option(False, help="Run in test-only mode"),
    local_rank: int = typer.Option(
        int(os.environ.get("LOCAL_RANK", -1)), help="Local rank for distributed training"
    ),
    no_remote_logging: bool = typer.Option(
        False,
        "--no-remote-logging",
        help="Disable remote loggers (wandb callback, remote_backup) for this run.",
    ),
) -> None:
    """Train a model using the specified Hydra config, with optional overrides."""
    trainer = build_trainer(
        config_file,
        local_rank=local_rank,
        overrides=overrides or [],
        no_remote_logging=no_remote_logging,
    )
    try:
        if test:
            trainer.test()
        else:
            trainer.train()
    except KeyboardInterrupt as exc:
        # Clean message, no scary traceback for a deliberate Ctrl-C.
        logger.warning("Run interrupted by user (KeyboardInterrupt).")
        sys.exit(_classify_exit_code(exc))
    except BaseException as exc:  # pylint: disable=broad-exception-caught
        # _finalize() (incl. TrialSummaryCallback.finalize) already ran inside
        # train()/test() via their try/finally. Log the traceback HERE before
        # collapsing it into an exit code — otherwise the failure reason is lost
        # entirely (this is how a disk-full OSError once died without a trace).
        logger.critical("Run failed with %s: %s", type(exc).__name__, exc, exc_info=exc)
        sys.exit(_classify_exit_code(exc))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    typer.run(_main)
