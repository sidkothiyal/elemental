"""DIV2K dataset."""
import pathlib
from collections import defaultdict
from typing import Dict, Optional, Tuple, Union

import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms


class DIV2KDataset(data.Dataset):
    """DIV2K dataset for image autoencoder training."""

    def __init__(
        self,
        root_dir: Union[str, pathlib.Path],
        target_shape: Optional[Tuple[int, int]] = None,
    ):
        """DIV2K dataset constructor.

        Args:
            root_dir: Path to the root directory of the dataset.
            target_shape: Optional tuple of (height, width) for random cropping.
                         If None, images are not cropped.
        """
        self.root_dir = pathlib.Path(root_dir)
        self.target_shape = target_shape

        # Find all PNG images in the directory
        self.images = list(self.root_dir.glob("*.png"))
        self.images.sort()  # Sort to ensure consistent ordering

        # Define transform pipeline
        transform_list = []

        # Add random crop if target shape is specified
        if self.target_shape is not None:
            height, width = self.target_shape
            transform_list.append(transforms.RandomCrop((height, width)))

        # Convert PIL image to tensor
        transform_list.append(transforms.ToTensor())

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
        image_path = self.images[idx]

        # Load image using PIL
        image = Image.open(image_path).convert("RGB")

        # Convert to tensor
        image_tensor = self.transform(image)

        # Return the same tensor as both input and target for autoencoder training
        return {
            "input": image_tensor,
            "target": image_tensor.clone().detach(),
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
