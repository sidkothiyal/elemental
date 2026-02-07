"""Kaggle dataset downloader.

Requires the kaggle package to be installed and authenticated.
Set up credentials via ~/.kaggle/kaggle.json or KAGGLE_USERNAME / KAGGLE_KEY env vars.
"""

import logging
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

logger = logging.getLogger(__name__)


class KaggleDownloader:  # pylint: disable=too-few-public-methods
    """Downloads datasets from Kaggle using the kaggle Python API."""

    def download(self, dataset_config: dict, download_dir: Path) -> Path:
        """Download a Kaggle dataset.

        Args:
            dataset_config: Dataset configuration dict containing 'kaggle_slug'.
            download_dir: Directory where the dataset will be downloaded.

        Returns:
            Path to the downloaded dataset directory.
        """
        kaggle_slug = dataset_config["kaggle_slug"]
        target_dir = download_dir

        logger.info("Downloading Kaggle dataset '%s' to '%s'", kaggle_slug, target_dir)

        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(
                kaggle_slug,
                path=str(target_dir),
                unzip=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download Kaggle dataset '{kaggle_slug}'. "
                "Make sure the kaggle package is installed and credentials are configured "
                "(~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY env vars)."
            ) from exc

        logger.info("Successfully downloaded '%s'", kaggle_slug)
        return target_dir
