"""Cityscapes dataset."""
import pathlib
from typing import Dict
from typing import Optional
from typing import Union

import numpy as np
import torch
from torch.utils import data


class CityscapesDataset(data.Dataset):
    """Cityscapes dataset."""

    def __init__(
        self,
        root_dir: Union[str, pathlib.Path],
        segmentation_enabled: bool = True,
        depth_enabled: bool = True,
        segmentation_class_id_mapping: Optional[Dict[int, int]] = None,
    ):
        """Cityscapes dataset constructor.

        Args:
            root_dir: Path to the root directory of the dataset.
            segmentation_enabled: Whether to load the segmentation masks.
            depth_enabled: Whether to load the depth maps.
            class_id_mapping: Mapping of class ids.
        """
        self.root_dir = pathlib.Path(root_dir)
        self.segmentation_enabled = segmentation_enabled
        self.depth_enabled = depth_enabled
        self.segmentation_class_id_mapping = segmentation_class_id_mapping
        if (
            self.segmentation_class_id_mapping is None
            and self.segmentation_enabled
        ):
            self.segmentation_class_id_mapping = {-1: 30}

        self.image_dir = self.root_dir / "image"
        self.seg_mask_dir = self.root_dir / "label"
        self.depth_dir = self.root_dir / "depth"

        self.images = list(self.image_dir.glob("*.npy"))

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.images)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """Get an item from the dataset.

        Args:
            idx: Index of the item to get.
        """
        image_path = self.images[idx]
        file_name = image_path.name

        seg_mask_path = self.seg_mask_dir / file_name
        depth_path = self.depth_dir / file_name

        image = np.load(image_path).transpose(2, 0, 1)
        seg_mask = np.load(seg_mask_path)
        if self.segmentation_class_id_mapping is not None:
            for old_id, new_id in self.segmentation_class_id_mapping.items():
                seg_mask[seg_mask == old_id] = new_id
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
    from tqdm import tqdm

    dataset = CityscapesDataset(
        root_dir="data/cityscapes/train",
        segmentation_enabled=True,
        depth_enabled=True,
    )

    print("Length of dataset:", len(dataset))

    dataloader = data.DataLoader(dataset, batch_size=2, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    min_class_id = 100
    max_class_id = 0
    for idx, batch in tqdm(enumerate(dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        if idx == 0:
            for k, v in batch.items():
                print(k, v.shape)

        seg_mask = batch["seg_mask"]
        min_class_id = min(min_class_id, seg_mask.min().item())
        max_class_id = max(max_class_id, seg_mask.max().item())

    print(min_class_id, max_class_id)
    # from PIL import Image
    # from tqdm import tqdm
    # import os

    # for idx in tqdm(range(len(dataset))):
    #     item = dataset[idx]
    #     output_dir = "data/cityscapes/output/"
    #     os.makedirs(output_dir, exist_ok=True)
    #     image = item["image"].cpu().numpy()
    #     image = image.transpose(1, 2, 0)
    #     image = image * 255.0
    #     image = image.astype(np.uint8)
    #     image = Image.fromarray(image)
    #     image.save(output_dir + f"{idx}.png")

    #     seg_mask = item["seg_mask"].cpu().numpy()
    #     seg_mask = seg_mask.astype(np.uint8) * 8
    #     seg_mask = Image.fromarray(seg_mask)
    #     seg_mask.save(output_dir + f"{idx}_seg_mask.png")

    #     depth = item["depth"].squeeze(0).cpu().numpy()
    #     depth = depth * 255.0
    #     depth = depth.astype(np.uint8)
    #     depth = Image.fromarray(depth)
    #     depth.save(output_dir + f"{idx}_depth.png")
