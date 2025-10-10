"""Imagenet dataset."""
import pathlib
from typing import Any, Dict, List, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def read_classes(dataset_dir: Union[str, pathlib.Path]) -> List[str]:
    """Read the classes from the file."""
    dataset_dir = pathlib.Path(dataset_dir)
    classes = []
    for class_dir in (dataset_dir / "train").glob("*"):
        classes.append(class_dir.name)
    return classes


class ImagenetDataset(Dataset):
    """Imagenet dataset."""

    def __init__(
        self,
        root_dir: Union[str, pathlib.Path],
        split: str,
        classes: List[str],
        normalize: bool = True,
    ):
        """Imagenet dataset constructor.

        Args:
            root_dir: Path to the root directory of the dataset.
            split: Split to use.
        """
        self.root_dir = pathlib.Path(root_dir)
        self.split = split
        self.classes = classes
        self.normalize = normalize

        self.classes.sort()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.sample_info: Dict[str, Dict[str, Any]] = {}

        getattr(self, f"_collect_{self.split}_data")()

        self._image_names = list(self.sample_info.keys())

    def _collect_train_data(self):
        """Collect train data from the root directory."""
        for class_name in self.classes:
            for image_path in (self.root_dir / class_name / "images").glob("*.JPEG"):
                self.sample_info[image_path.name] = {
                    "class_idx": self.class_to_idx[class_name],
                    "image_path": image_path,
                }

    def _collect_val_data(self):
        """Collect val data from the root directory."""
        val_sample_mapping = self._read_sample_mapping(self.root_dir / "val_annotations.txt")

        for image_path in (self.root_dir / "images").glob("*.JPEG"):
            if image_path.name in val_sample_mapping:
                self.sample_info[image_path.name] = {
                    "class_idx": self.class_to_idx[val_sample_mapping[image_path.name]["class"]],
                    "image_path": image_path,
                }

    def _read_sample_mapping(self, file_path: str) -> Dict[str, Dict[str, str]]:
        """Read the sample mapping from the file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return {line.split("\t")[0]: {"class": line.split("\t")[1]} for line in f}

    def __len__(self):
        return len(self._image_names)

    def __getitem__(self, idx):
        image_name = self._image_names[idx]
        sample_info = self.sample_info[image_name]
        image_path = sample_info["image_path"]

        image = Image.open(image_path).convert("RGB")
        image = np.array(image).transpose(2, 0, 1)
        if self.normalize:
            image = image / 255.0

        return {
            "input": torch.from_numpy(image).float(),
            "target": torch.tensor(sample_info["class_idx"]).long(),
        }


if __name__ == "__main__":
    tiny_imagenet_dataset_dir = pathlib.Path("data/tiny_imagenet")
    tiny_imagenet_classes = read_classes(tiny_imagenet_dataset_dir)
    print(len(tiny_imagenet_classes))

    train_dataset = ImagenetDataset(
        tiny_imagenet_dataset_dir / "train", "train", tiny_imagenet_classes
    )
    print(len(train_dataset))
    print(train_dataset[0])

    val_dataset = ImagenetDataset(tiny_imagenet_dataset_dir / "val", "val", tiny_imagenet_classes)
    print(len(val_dataset))
    print(val_dataset[0])
