import pathlib

import dill
import lz4.frame as lz4


class Tokenizer:
    def __init__(self, data_path: str, split: str = "train"):
        self.data_path = pathlib.Path(data_path)
        self.split = split

        split_data_path = self.data_path / f"{split}_data.pkl.lz4"
        with lz4.open(split_data_path, "rb") as f:
            self.chars, self.stoi, self.itos, self.data = dill.load(f)

    def encode(self, string: str) -> list[int]:
        return [self.stoi[character] for character in string]

    def decode(self, letters: list[int]) -> str:
        return "".join([self.itos[letter.item()] for letter in letters])
