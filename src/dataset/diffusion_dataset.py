"""Diffusion dataset for loading reverse diffused images."""

import pathlib
import re

import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms


class DiffusionDataset(data.Dataset):
    """Dataset for loading reverse diffused images created by reverse_diffusion.py.

    This dataset loads pairs of noisy images (input) and clean base images (target)
    for training diffusion models or denoisers.
    """

    def __init__(
        self,
        root_dir: str | pathlib.Path,
        target_shape: tuple[int, int] | None = None,
        transform: transforms.Compose | None = None,
        normalize: bool = True,
        return_timestep: bool = True,
    ):
        """DiffusionDataset constructor.

        Args:
            root_dir: Path to the root directory containing subdirectories with
                     base images and their noisy versions.
            target_shape: Optional tuple of (height, width) for random cropping.
                         If None, images are not cropped.
            transform: Optional torchvision transforms to apply to images.
            normalize: Whether to normalize images to [-1, 1] range (default: True).
                      If False, images are in [0, 1] range.
            return_timestep: Whether to return the timestep value in the output dict.
        """
        self.root_dir = pathlib.Path(root_dir)
        self.normalize = normalize
        self.return_timestep = return_timestep
        self.target_shape = target_shape
        # Build index of all image pairs
        self._build_index()

        # Define transform pipeline
        transform_list = []

        # Add custom transforms if provided
        if transform is not None:
            transform_list.append(transform)

        # Add random crop if target shape is specified
        if self.target_shape is not None:
            height, width = self.target_shape
            transform_list.append(transforms.RandomCrop((height, width)))

        # Convert PIL image to tensor
        transform_list.append(transforms.ToTensor())

        # Normalize to [-1, 1] if requested (same as training diffusion models)
        if self.normalize:
            transform_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

        self.transform = transforms.Compose(transform_list)

    def _build_index(self):
        """Build index of all base images and their noisy timestep versions."""
        # Pattern to match timestep images: {name}_timestep_{t:04d}.png
        timestep_pattern = re.compile(r"(.+)_timestep_(\d{4})\.png")

        self.samples = []

        for per_image_dir in self.root_dir.iterdir():
            base_image_path = per_image_dir / f"{per_image_dir.stem}.png"
            for noisy_image_path in per_image_dir.iterdir():
                match = timestep_pattern.match(noisy_image_path.name)
                if match:
                    timestep = int(match.group(2))
                    self.samples.append((base_image_path, noisy_image_path, timestep))

        if len(self.samples) == 0:
            raise ValueError(
                f"No valid image pairs found in {self.root_dir}. "
                "Make sure the directory contains subdirectories with base images "
                "and their corresponding noisy timestep images."
            )

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.samples)

    def __getitem__(self, idx) -> dict[str, torch.Tensor | int]:
        """Get an item from the dataset.

        Args:
            idx: Index of the item to get.

        Returns:
            Dictionary with keys:
                - 'input': Noisy image tensor
                - 'target': Clean base image tensor
                - 'timestep': Timestep value (if return_timestep=True)
        """
        base_path, noisy_path, timestep = self.samples[idx]

        # Load images using PIL
        base_image = Image.open(base_path).convert("RGB")
        noisy_image = Image.open(noisy_path).convert("RGB")

        # Apply transforms
        base_tensor = self.transform(base_image)
        noisy_tensor = self.transform(noisy_image)

        result = {
            "input": noisy_tensor,
            "target": base_tensor,
        }

        if self.return_timestep:
            result["timestep"] = timestep

        return result


if __name__ == "__main__":
    from tqdm import tqdm

    # Test the dataset
    print("Testing DiffusionDataset...")

    dataset = DiffusionDataset(root_dir="data/diffused_div2k/train", target_shape=(640, 1024))
    print(f"Length of dataset: {len(dataset)}")

    # Test dataloader
    dataloader = data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for batch_idx, batch in enumerate(tqdm(dataloader)):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        if batch_idx == 0:
            print("\nFirst batch:")
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    print(
                        f"  {k}: shape={v.shape}, dtype={v.dtype}, "
                        f"min={v.min():.3f}, max={v.max():.3f}"
                    )
                else:
                    print(f"  {k}: {v}")

        if batch_idx >= 10:
            break

    print("\nDiffusionDataset test completed successfully!")

    # Test DiffusionDatasetByTimestep
    print("\n" + "=" * 50 + "\n")
