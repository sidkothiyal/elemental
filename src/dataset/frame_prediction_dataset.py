"""Dataset wrapper for next-frame prediction.

Wraps a sequence dataset and splits each sample's image sequence into
context frames (input) and a target frame to predict.
"""

import torch
from torch.utils.data import Dataset


class FramePredictionDataset(Dataset):
    """Split a sequence dataset's images into context and target frames.

    Given a base dataset that returns ``image: (T, C, H, W)``, this wrapper
    produces ``{input: (T-1, C, H, W), target: (C, H, W)}`` where the target
    is the last frame.

    Args:
        base_dataset: A dataset returning dicts with an ``image_key`` entry
            of shape ``(T, C, H, W)`` where ``T >= 2``.
        image_key: Key for the image sequence in the base dataset output.
    """

    def __init__(self, base_dataset: Dataset, image_key: str = "image"):
        self.base_dataset = base_dataset
        self.image_key = image_key

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.base_dataset[idx]
        images = sample[self.image_key]  # (T, C, H, W)

        result = {
            "input": images[:-1],  # (T-1, C, H, W) — context frames
            "target": images[-1],  # (C, H, W) — frame to predict
        }

        # Pass through any non-image keys under their original names
        for key, value in sample.items():
            if key != self.image_key:
                result[key] = value

        return result
