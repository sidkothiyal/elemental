"""ONNX model export entrypoint.

Accepts either a full experiment config, a model-only config, or a dedicated
export config.  See ``training.model_exporter`` for details.

Usage examples::

    # Full experiment config (export section embedded or via overrides)
    uv run python src/exporter.py experiment/autoencoder/compression_ae \\
        --checkpoint results/autoencoder/run/best.pt \\
        --output-dir exports/compression_ae/

    # Model-only config (no optimizer / loss / dataset needed)
    uv run python src/exporter.py model/vq_vae_transformer \\
        --checkpoint checkpoints/vqvae.pt \\
        --output-dir exports/vqvae/

    # Dedicated export config with sub-model specs
    uv run python src/exporter.py export/vq_vae_transformer \\
        --checkpoint checkpoints/vqvae.pt \\
        --output-dir exports/vqvae/

    # Hydra overrides are appended after positional args
    uv run python src/exporter.py export/vq_vae_transformer \\
        --output-dir exports/ export.opset_version=18
"""

import logging

import typer

from training.model_exporter import build_model_exporter


def _main(
    config_file: str = typer.Argument(
        ...,
        help=(
            "Hydra config name. Can be a full experiment config "
            "(e.g. experiment/autoencoder/compression_ae), a model-only config "
            "(e.g. model/vq_vae_transformer), or a dedicated export config "
            "(e.g. export/vq_vae_transformer)."
        ),
    ),
    output_dir: str = typer.Option(
        "exports/",
        "--output-dir",
        "-o",
        help="Directory to write exported ONNX files to.",
    ),
    checkpoint: str
    | None = typer.Option(
        None,
        "--checkpoint",
        "-c",
        help=(
            "Path to a checkpoint file to load weights from. "
            "Overrides load_checkpoint_path in the config when provided."
        ),
    ),
    overrides: list[str] = typer.Argument(
        default=None,
        help="Hydra overrides (e.g. export.opset_version=18 device=cpu).",
    ),
) -> None:
    """Export a trained model to ONNX format."""
    exporter = build_model_exporter(
        config_file=config_file,
        output_dir=output_dir,
        checkpoint=checkpoint,
        overrides=overrides or [],
    )
    exporter.export()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    typer.run(_main)
