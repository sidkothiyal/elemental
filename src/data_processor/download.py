"""Dataset download script.

Usage:
    python -m src.data_processor.download --download_dir=<path> --dataset_name=<name>

Examples:
    python -m src.data_processor.download --download_dir=./data --dataset_name=div2k
    python -m src.data_processor.download --download_dir=./data --dataset_name=cityscapes
    python -m src.data_processor.download --download_dir=./data --dataset_name=tiny_imagenet
    python -m src.data_processor.download --download_dir=./data --dataset_name=tiny_shakespeare
    python -m src.data_processor.download --download_dir=./data --dataset_name=wikitext
"""

import logging
from functools import partial
from pathlib import Path

import fire
import yaml

from data_processor.downloaders import kaggle_downloader, url_downloader
from data_processor.post_download_hooks.reorganizer import reorganize_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[0] / "dataset_config.yaml"

DOWNLOADERS = {
    "kaggle": kaggle_downloader.KaggleDownloader(),
    "url": url_downloader.URLDownloader(),
}


POST_DOWNLOAD_HOOKS = {
    "reorganize_cityscapes": partial(
        reorganize_dataset,
        train_pattern="data/train",
        val_pattern="data/val",
    ),
    "reorganize_div2k": partial(
        reorganize_dataset,
        train_pattern="DIV2K_train_HR/DIV2K_train_HR",
        val_pattern="DIV2K_valid_HR/DIV2K_valid_HR",
    ),
}


def download(download_dir: str, dataset_name: str) -> None:
    """Download a dataset to the specified directory.

    Args:
        download_dir: Root directory where the dataset will be stored.
            The dataset will be placed under <download_dir>/<dataset_name>/.
        dataset_name: Name of the dataset to download (must exist in dataset_config.yaml).
    """
    # Load config
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Dataset config not found at '{CONFIG_PATH}'. Make sure dataset_config.yaml exists."
        )

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    datasets = config.get("datasets", {})

    if dataset_name not in datasets:
        available = ", ".join(sorted(datasets.keys()))
        raise ValueError(
            f"Dataset '{dataset_name}' not found in dataset_config.yaml. "
            f"Available datasets: {available}"
        )

    dataset_config = datasets[dataset_name]
    download_method = dataset_config["download_method"]

    if download_method not in DOWNLOADERS:
        raise ValueError(
            f"Unknown download method '{download_method}' for dataset '{dataset_name}'. "
            f"Supported methods: {', '.join(DOWNLOADERS.keys())}"
        )

    download_path = Path(download_dir) / dataset_name
    logger.info(
        "Downloading dataset '%s' (method: %s) to '%s'",
        dataset_name,
        download_method,
        download_path,
    )

    downloader = DOWNLOADERS[download_method]
    downloader.download(dataset_config, download_path)

    # Run any post-download hooks
    hook_name = dataset_config.get("post_download")
    if hook_name:
        if hook_name not in POST_DOWNLOAD_HOOKS:
            raise ValueError(f"Unknown post-download hook: '{hook_name}'")
        logger.info("Running post-download hook '%s'", hook_name)
        POST_DOWNLOAD_HOOKS[hook_name](download_path)

    logger.info("Done! Dataset '%s' is ready at '%s'", dataset_name, download_path)


if __name__ == "__main__":
    fire.Fire(download)
