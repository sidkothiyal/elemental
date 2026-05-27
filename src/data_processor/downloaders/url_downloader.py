"""URL-based dataset downloader.

Supports downloading single files and zip archives from direct URLs.
"""

import logging
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)


class URLDownloader:  # pylint: disable=too-few-public-methods
    """Downloads datasets from direct URLs."""

    def download(self, dataset_config: dict, download_dir: Path) -> Path:
        """Download a dataset from a URL.

        Args:
            dataset_config: Dataset configuration dict containing 'url' and
                optionally 'filename' and 'extract'.
            download_dir: Directory where the dataset will be downloaded.

        Returns:
            Path to the downloaded dataset directory.
        """
        url = dataset_config["url"]
        extract = dataset_config.get("extract", False)
        filename = dataset_config.get("filename", url.split("/")[-1])

        download_dir.mkdir(parents=True, exist_ok=True)
        dest_path = download_dir / filename

        logger.info("Downloading '%s' -> '%s'", url, dest_path)

        # Download with progress reporting
        urllib.request.urlretrieve(url, str(dest_path), reporthook=self._progress_hook)
        print()  # newline after progress

        if extract and zipfile.is_zipfile(dest_path):
            logger.info("Extracting '%s'", dest_path)
            with zipfile.ZipFile(dest_path, "r") as zf:
                zf.extractall(download_dir)
            # Remove the zip after extraction
            dest_path.unlink()
            logger.info("Extracted and removed archive '%s'", filename)

        logger.info("Successfully downloaded to '%s'", download_dir)
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
