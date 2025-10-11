import pathlib

import dill
import lz4.frame as lz4
from torch.utils.data import Dataset


class TinyShakespeareDataset(Dataset):
    def __init__(self, dataset_dir: str, split: str, block_size: int):
        self.dataset_dir = pathlib.Path(dataset_dir)
        self.split = split
        self.block_size = block_size

        split_dataset_dir = self.dataset_dir / f"{split}_data.pkl.lz4"
        with lz4.open(split_dataset_dir, "rb") as f:
            self.chars, self.stoi, self.itos, self.data = dill.load(f)

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]

        assert len(x) == len(y) == self.block_size, f"x: {x}, y: {y}, block_size: {self.block_size}"

        return {
            "input": x,
            "target": y,
        }


if __name__ == "__main__":
    dataset = TinyShakespeareDataset("data/tiny_shakespeare", "train", 8)
    print(dataset[0])
