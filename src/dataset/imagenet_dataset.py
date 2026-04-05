"""Imagenet dataset."""
import pathlib
from typing import Any

import numpy as np
import torch
from PIL import Image

from dataset.base_dataset import BaseDataset


def read_classes(dataset_dir: str | pathlib.Path) -> list[str]:
    """Read the classes from the file."""
    dataset_dir = pathlib.Path(dataset_dir)
    classes = []
    for class_dir in (dataset_dir / "train").glob("*"):
        classes.append(class_dir.name)
    return classes


class ImagenetDataset(BaseDataset):
    """Imagenet dataset."""

    def __init__(
        self,
        root_dir: str | pathlib.Path,
        split: str,
        classes: list[str],
        normalize: bool = True,
        remote_config: dict | None = None,
    ):
        """Imagenet dataset constructor.

        Args:
            root_dir: Path to the root directory of the dataset.
            split: Split to use.
            remote_config: Optional remote storage configuration.
        """
        super().__init__(remote_config=remote_config)
        self.root_dir = pathlib.Path(root_dir)
        self.split = split
        self.classes = classes
        self.normalize = normalize

        # For sync mode, point root_dir at the local cache
        if self.is_remote and self.remote_mode == "sync":
            self.root_dir = self.remote_root_as_local()

        self.classes.sort()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.sample_info: dict[str, dict[str, Any]] = {}

        getattr(self, f"_collect_{self.split}_data")()

        self._image_names = list(self.sample_info.keys())

        # Prefetch all image files
        self.start_prefetch([info["image_path"] for info in self.sample_info.values()])

    def _collect_train_data(self):
        """Collect train data from the root directory.

        Uses structure-aware discovery: classes are known, so we iterate
        each class subdirectory rather than globbing the entire tree.
        """
        for class_name in self.classes:
            # Use discover_files_from_structure for remote-efficient listing
            image_paths = self.discover_files_from_structure(
                self.root_dir / class_name, subdirs=["images"], pattern="*.JPEG"
            )
            for image_path in image_paths:
                self.sample_info[image_path.name] = {
                    "class_idx": self.class_to_idx[class_name],
                    "image_path": image_path,
                }

    def _collect_val_data(self):
        """Collect val data from the root directory."""
        annotations_path = self.resolve_path(self.root_dir / "val_annotations.txt")
        val_sample_mapping = self._read_sample_mapping(annotations_path)

        image_paths = self.list_files(self.root_dir / "images", "*.JPEG")
        for image_path in image_paths:
            if image_path.name in val_sample_mapping:
                self.sample_info[image_path.name] = {
                    "class_idx": self.class_to_idx[val_sample_mapping[image_path.name]["class"]],
                    "image_path": image_path,
                }

    def _read_sample_mapping(self, file_path: str | pathlib.Path) -> dict[str, dict[str, str]]:
        """Read the sample mapping from the file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return {line.split("\t")[0]: {"class": line.split("\t")[1]} for line in f}

    def __len__(self):
        return len(self._image_names)

    def __getitem__(self, idx):
        image_name = self._image_names[idx]
        sample_info = self.sample_info[image_name]
        image_path = sample_info["image_path"]

        image = Image.open(self.resolve_path(image_path)).convert("RGB")
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
