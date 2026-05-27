"""Cityscapes dataset."""
import pathlib

import numpy as np
import torch
from torch.utils import data

from dataset.base_dataset import BaseDataset


class CityscapesDataset(BaseDataset):
    """Cityscapes dataset."""

    def __init__(
        self,
        root_dir: str | pathlib.Path,
        segmentation_enabled: bool = True,
        depth_enabled: bool = True,
        segmentation_class_id_mapping: dict[int, int] | None = None,
        rename_output_keys: dict[str, str] | None = None,
        remote_config: dict | None = None,
    ):
        """Cityscapes dataset constructor.

        Args:
            root_dir: Path to the root directory of the dataset.
            segmentation_enabled: Whether to load the segmentation masks.
            depth_enabled: Whether to load the depth maps.
            class_id_mapping: Mapping of class ids.
            remote_config: Optional remote storage configuration.
        """
        super().__init__(remote_config=remote_config)
        self.root_dir = pathlib.Path(root_dir)
        self.segmentation_enabled = segmentation_enabled
        self.depth_enabled = depth_enabled
        self.segmentation_class_id_mapping = segmentation_class_id_mapping
        if self.segmentation_class_id_mapping is None and self.segmentation_enabled:
            self.segmentation_class_id_mapping = {-1: 30}

        self.rename_output_keys = rename_output_keys

        # For sync mode, point root_dir at the local cache
        if self.is_remote and self.remote_mode == "sync":
            self.root_dir = self.remote_root_as_local()

        self.image_dir = self.root_dir / "image"
        self.seg_mask_dir = self.root_dir / "label"
        self.depth_dir = self.root_dir / "depth"

        # Discover image files
        self.images = self.list_files(self.image_dir, "*.npy")

        # Prefetch all files (images + labels + depth share names)
        self.start_prefetch(self._all_file_paths())

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.images)

    def _rename_output(self, item_data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Rename the output keys."""
        if not self.rename_output_keys:
            return item_data

        for old_key, new_key in self.rename_output_keys.items():
            item_data[new_key] = item_data[old_key]
        return item_data

    def _all_file_paths(self) -> list[pathlib.Path]:
        """Build complete list of files to prefetch (all subdirs)."""
        paths = list(self.images)
        for img_path in self.images:
            paths.append(self.seg_mask_dir / img_path.name)
            paths.append(self.depth_dir / img_path.name)
        return paths

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        """Get an item from the dataset.

        Args:
            idx: Index of the item to get.
        """
        image_path = self.resolve_path(self.images[idx])
        file_name = image_path.name

        seg_mask_path = self.resolve_path(self.seg_mask_dir / file_name)
        depth_path = self.resolve_path(self.depth_dir / file_name)

        image = np.load(image_path).transpose(2, 0, 1)
        seg_mask = np.load(seg_mask_path)
        if self.segmentation_class_id_mapping is not None:
            for old_id, new_id in self.segmentation_class_id_mapping.items():
                seg_mask[seg_mask == old_id] = new_id
        depth = np.load(depth_path).transpose(2, 0, 1)

        item_data = {
            "input": torch.from_numpy(image).float(),
        }

        if self.segmentation_enabled:
            item_data["seg_mask"] = torch.from_numpy(seg_mask).long()

        if self.depth_enabled:
            item_data["depth"] = torch.from_numpy(depth).float()

        item_data = self._rename_output(item_data)

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

    MIN_CLASS_ID = float("inf")
    MAX_CLASS_ID = float("-inf")
    for batch_idx, batch in tqdm(enumerate(dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        if batch_idx == 0:
            for k, v in batch.items():
                print(k, v.shape)

        seg_mask_data = batch["seg_mask"]
        MIN_CLASS_ID = min(MIN_CLASS_ID, seg_mask_data.min().item())
        MAX_CLASS_ID = max(MAX_CLASS_ID, seg_mask_data.max().item())

    print(MIN_CLASS_ID, MAX_CLASS_ID)
