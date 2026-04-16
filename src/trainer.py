"""Hydra CLI entrypoint for training.

Delegates to :func:`training.builder.build_trainer` which constructs the
appropriate :class:`training.trainer.Trainer` subclass based on ``config.mode``.
"""

import logging
import os

import typer

from training.builder import build_trainer


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
) -> None:
    """Train a model using the specified Hydra config, with optional overrides."""
    trainer = build_trainer(config_file, local_rank=local_rank, overrides=overrides or [])
    if test:
        trainer.test()
    else:
        trainer.train()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    typer.run(_main)
