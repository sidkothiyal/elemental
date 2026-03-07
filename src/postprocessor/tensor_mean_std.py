"""Post-processor for computing mean and standard deviation of any tensor."""

import torch

from postprocessor.base import BasePostProcessor


class TensorMeanStdPostProcessor(BasePostProcessor):
    """Compute the mean and standard deviation of a tensor (over all elements).

    Returns a dict with "mean" and "std" as Python floats, suitable for
    logging or feeding into running-statistics logic.
    """

    def __call__(self, x: torch.Tensor) -> dict[str, float]:  # type: ignore[override]
        """Compute mean and std over all elements of x.

        Args:
            x: Any tensor.

        Returns:
            Dict with keys "mean" and "std" (Python floats). If the tensor
            has no elements or zero variance, std is 0.0.
        """
        if x.numel() == 0:
            return {"mean": 0.0, "std": 0.0}
        mean = x.detach().float().mean().item()
        std = x.detach().float().std().item()
        if torch.isnan(torch.tensor(std)) or torch.isinf(torch.tensor(std)):
            std = 0.0
        return {"mean": mean, "std": std}
