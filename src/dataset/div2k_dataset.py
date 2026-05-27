"""DIV2K dataset."""
import pathlib
from collections import defaultdict
from typing import Dict, Optional, Tuple, Union

import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms

from dataset.base_dataset import BaseDataset


class DIV2KDataset(BaseDataset):
    """DIV2K dataset for image autoencoder training."""

    def __init__(
        self,
        root_dir: Union[str, pathlib.Path],
        target_shape: Optional[Tuple[int, int]] = None,
        max_images: Optional[int] = None,
        remote_config: dict | None = None,
    ):
        """DIV2K dataset constructor.

        Args:
            root_dir: Path to the root directory of the dataset.
            target_shape: Optional tuple of (height, width) for random cropping.
                         If None, images are not cropped.
            max_images: Optional maximum number of images to load. If None, loads all images.
            remote_config: Optional remote storage configuration. See
                :class:`~dataset.base_dataset.BaseDataset` for details.
        """
        super().__init__(remote_config=remote_config)
        self.root_dir = pathlib.Path(root_dir)
        self.target_shape = target_shape
        self.max_images = max_images

        # For sync mode, point root_dir at the local cache
        if self.is_remote and self.remote_mode == "sync":
            self.root_dir = self.remote_root_as_local()

        # Discover images — use structure-aware discovery for remote
        if self.is_remote and self.remote_mode == "stream":
            # DIV2K has numbered PNGs (0001.png - 0900.png for train,
            # 0801.png - 0900.png for val). Use glob as fallback since
            # the exact range depends on the split.
            self.images = self.list_files(self.root_dir, "*.png")
        else:
            # Local or sync mode: fast local glob
            self.images = list(self.root_dir.glob("*.png"))
            self.images.sort()  # Sort to ensure consistent ordering

        # Start background prefetching for remote stream mode
        self.start_prefetch(self.images)

        # Define transform pipeline
        transform_list = []

        # Add random crop if target shape is specified
        if self.target_shape is not None:
            height, width = self.target_shape
            transform_list.append(transforms.RandomCrop((height, width)))

        # Convert PIL image to tensor (outputs [0, 1] range)
        transform_list.append(transforms.ToTensor())

        # Normalize to [-1, 1] range for diffusion models
        # ToTensor outputs [0, 1], so we do: (x - 0.5) / 0.5 = x * 2 - 1 = [-1, 1]
        transform_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.images)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """Get an item from the dataset.

        Args:
            idx: Index of the item to get.

        Returns:
            Dictionary with 'input' and 'target' keys, both containing the same tensor.
        """
        if self.max_images:
            # Limiting the number of images while allowing indices to wrap around increases the
            # number of steps per epoch, reducing the overhead of epoch transitions.
            idx = idx % self.max_images
        image_path = self.resolve_path(self.images[idx])

        # Load image using PIL
        image = Image.open(image_path).convert("RGB")

        # Convert to tensor
        image_tensor = self.transform(image)

        # Return the same tensor as both input and target for autoencoder training.
        # Both share the same data — the DataLoader's default collate will stack
        # them into separate batch tensors anyway, so no aliasing issues arise.
        return {
            "input": image_tensor,
            "target": image_tensor,
        }


if __name__ == "__main__":
    from tqdm import tqdm

    # Test the dataset
    dataset = DIV2KDataset(root_dir="data/div2k/train")
    print("Length of dataset:", len(dataset))

    # Test with cropping
    print("\nTesting with target shape (640, 1024):")
    dataset_cropped = DIV2KDataset(root_dir="data/div2k/train", target_shape=(640, 1024))
    print("Length of cropped dataset:", len(dataset_cropped))

    dataloader = data.DataLoader(dataset_cropped, batch_size=2, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unique_shapes: dict[tuple[int, int], int] = defaultdict(int)
    min_height = float("inf")
    min_width = float("inf")
    max_height = float("-inf")
    max_width = float("-inf")
    for batch_idx, batch in tqdm(enumerate(dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        if batch_idx == 0:
            for k, v in batch.items():
                print(f"{k}: {v.shape}, dtype: {v.dtype}, min: {v.min():.3f}, max: {v.max():.3f}")
        unique_shapes[batch["input"].shape] += 1
        min_height = min(min_height, batch["input"].shape[2])
        min_width = min(min_width, batch["input"].shape[3])
        max_height = max(max_height, batch["input"].shape[2])
        max_width = max(max_width, batch["input"].shape[3])

    print("Number of unique shapes:", len(unique_shapes))

    print("Min height:", min_height)
    print("Min width:", min_width)
    print("Max height:", max_height)
    print("Max width:", max_width)
    print("Dataset test completed successfully!")
