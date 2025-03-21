import pathlib
import typing

import numpy as np
import torch
from torch.utils import data


class CityscapesDataset(data.Dataset):
    def __init__(
        self,
        root_dir: typing.Union[str, pathlib.Path],
        segmentation_enabled: bool = True,
        depth_enabled: bool = True,
    ):
        """Cityscapes dataset constructor.

        Args:
            root_dir: Path to the root directory of the dataset.
            segmentation_enabled: Whether to load the segmentation masks.
            depth_enabled: Whether to load the depth maps.
        """
        self.root_dir = pathlib.Path(root_dir)
        self.segmentation_enabled = segmentation_enabled
        self.depth_enabled = depth_enabled

        self.image_dir = self.root_dir / "image"
        self.seg_mask_dir = self.root_dir / "label"
        self.depth_dir = self.root_dir / "depth"

        self.images = list(self.image_dir.glob("*.npy"))

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.images)

    def __getitem__(self, idx) -> typing.Dict[str, torch.Tensor]:
        """Get an item from the dataset.

        Args:
            idx: Index of the item to get.
        """
        image_path = self.images[idx]
        file_name = image_path.name

        seg_mask_path = self.seg_mask_dir / file_name
        depth_path = self.depth_dir / file_name
        print(seg_mask_path)

        image = np.load(image_path).transpose(2, 0, 1)
        seg_mask = np.load(seg_mask_path)
        depth = np.load(depth_path).transpose(2, 0, 1)

        item_data = {
            "image": torch.from_numpy(image).float(),
        }

        if self.segmentation_enabled:
            item_data["seg_mask"] = torch.from_numpy(seg_mask).long()

        if self.depth_enabled:
            item_data["depth"] = torch.from_numpy(depth).float()

        return item_data


if __name__ == "__main__":
    dataset = CityscapesDataset(
        root_dir="data/cityscapes/train",
        segmentation_enabled=True,
        depth_enabled=True,
    )

    print("Length of dataset:", len(dataset))

    dataloader = data.DataLoader(dataset, batch_size=2, shuffle=True)

    for batch in dataloader:
        for k, v in batch.items():
            print(k, v.shape)
        break
