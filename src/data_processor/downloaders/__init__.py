"""Downloaders for the data processor."""

from data_processor.downloaders.kaggle_downloader import KaggleDownloader
from data_processor.downloaders.url_downloader import URLDownloader

__all__ = ["KaggleDownloader", "URLDownloader"]
