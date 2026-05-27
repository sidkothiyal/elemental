"""Tokenizer for the data processor."""

import pathlib

import dill
import lz4.frame as lz4
import torch


class Tokenizer:
    """Tokenizer for the data processor."""

    def __init__(self, data_path: str, split: str = "train"):
        """Tokenizer constructor.

        Args:
            data_path: Path to the data directory.
            split: Split to use ('train' or 'val').
        """
        self.data_path = pathlib.Path(data_path)
        self.split = split
        self.chars, self.stoi, self.itos, self.data = self._load_data()

    def _load_data(self) -> tuple[list[str], dict[str, int], dict[int, str], torch.Tensor]:
        """Load the data from the data directory."""
        split_data_path = self.data_path / f"{self.split}_data.pkl.lz4"
        with lz4.open(split_data_path, "rb") as f:
            return dill.load(f)

    def encode(self, string: str) -> list[int]:
        """Encode a string to a list of integers."""
        return [self.stoi[character] for character in string]

    def decode(self, letters: list[int]) -> str:
        """Decode a list of integers to a string."""
        return "".join([self.itos[letter.item()] for letter in letters])
