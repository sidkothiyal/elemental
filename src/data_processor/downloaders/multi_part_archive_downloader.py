"""Multi-part archive downloader.

Downloads a list of archive files from a common base URL and (optionally)
extracts each one. Useful for datasets split across multiple `.tar.gz` parts
hosted on a public bucket / web server (e.g. the BVI AOM dataset on Netflix's
public S3).
"""

import logging
import tarfile
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)


class MultiPartArchiveDownloader:  # pylint: disable=too-few-public-methods
    """Downloads a configurable subset of archive parts from a common base URL."""

    def download(self, dataset_config: dict, download_dir: Path) -> Path:
        """Download (and optionally extract) the configured archive parts.

        Args:
            dataset_config: Dataset configuration dict containing 'base_url',
                'parts' (list of filenames), and optionally 'extract'
                (bool, default True).
            download_dir: Directory where archives are downloaded and extracted.

        Returns:
            Path to the dataset directory.
        """
        base_url = dataset_config["base_url"].rstrip("/") + "/"
        parts = dataset_config["parts"]
        extract = dataset_config.get("extract", True)

        if not parts:
            raise ValueError(
                "No 'parts' specified for multi_part_archive download. "
                "Provide a list of filenames in the dataset config or via --parts."
            )

        download_dir.mkdir(parents=True, exist_ok=True)

        for part in parts:
            url = base_url + part
            dest_path = download_dir / part

            logger.info("Downloading '%s' -> '%s'", url, dest_path)
            urllib.request.urlretrieve(url, str(dest_path), reporthook=self._progress_hook)
            print()  # newline after progress

            if extract:
                if not tarfile.is_tarfile(dest_path):
                    raise RuntimeError(
                        f"Expected '{dest_path}' to be a tar archive but it isn't. "
                        "Set 'extract: false' in the dataset config to keep raw files."
                    )
                logger.info("Extracting '%s'", dest_path)
                with tarfile.open(dest_path, "r:*") as tf:
                    tf.extractall(download_dir)
                dest_path.unlink()
                logger.info("Extracted and removed archive '%s'", part)

        logger.info("Successfully downloaded %d part(s) to '%s'", len(parts), download_dir)
        return download_dir

    @staticmethod
    def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
        """Print download progress."""
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 // total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(
                f"\r  Progress: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)",
                end="",
                flush=True,
            )
        else:
            mb_downloaded = downloaded / (1024 * 1024)
            print(f"\r  Downloaded: {mb_downloaded:.1f} MB", end="", flush=True)
