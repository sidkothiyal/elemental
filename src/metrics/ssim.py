"""Structural Similarity Index Measure (SSIM) metric using pytorch-msssim."""

import torch
from pytorch_msssim import ssim as functional_ssim
from torchmetrics import Metric


class SSIM(Metric):
    """Structural Similarity Index Measure for image quality assessment.

    This metric wraps the pytorch-msssim library for efficient, differentiable
    SSIM computation.

    SSIM measures the structural similarity between two images based on
    luminance, contrast, and structure comparisons. Values range from -1 to 1,
    where 1 indicates identical images.

    Reference:
        Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).
        Image quality assessment: from error visibility to structural similarity.
        IEEE transactions on image processing, 13(4), 600-612.
    """

    higher_is_better = True
    full_state_update = False

    def __init__(
        self,
        data_range: float = 1.0,
        win_size: int = 11,
        win_sigma: float = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
        nonnegative_ssim: bool = False,
    ):
        """Initialize the SSIM metric.

        Args:
            data_range: The data range of the input images (max - min).
                Default is 1.0 for normalized images [0, 1].
                Use 255.0 for uint8 images [0, 255].
            win_size: Size of the Gaussian kernel for local statistics.
                Must be odd. Default is 11.
            win_sigma: Standard deviation of the Gaussian kernel. Default is 1.5.
            k1: Constant for luminance comparison stability. Default is 0.01.
            k2: Constant for contrast comparison stability. Default is 0.03.
            nonnegative_ssim: If True, force SSIM values to be non-negative
                by using relu on the result. Default is False.
        """
        super().__init__()

        if win_size % 2 == 0:
            raise ValueError(f"win_size must be odd, got {win_size}")

        self.data_range = data_range
        self.win_size = win_size
        self.win_sigma = win_sigma
        self.k1 = k1
        self.k2 = k2
        self.nonnegative_ssim = nonnegative_ssim

        self.add_state("sum_ssim", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric state with predictions and targets.

        Args:
            preds: Predicted tensor of shape (N, C, H, W) or (N, H, W).
            target: Target tensor of same shape as preds.
        """
        if preds.shape != target.shape:
            raise ValueError(
                f"Predictions and targets must have the same shape. "
                f"Got preds: {preds.shape}, target: {target.shape}"
            )

        # Convert to float for computation
        preds = preds.float()
        target = target.float()

        # Add channel dimension if needed (N, H, W) -> (N, 1, H, W)
        if preds.ndim == 3:
            preds = preds.unsqueeze(1)
            target = target.unsqueeze(1)

        batch_size = preds.shape[0]

        # Compute SSIM using pytorch-msssim
        # size_average=False returns per-image SSIM values
        ssim_values = functional_ssim(
            preds,
            target,
            data_range=self.data_range,
            size_average=False,
            win_size=self.win_size,
            win_sigma=self.win_sigma,
            K=(self.k1, self.k2),
            nonnegative_ssim=self.nonnegative_ssim,
        )

        self.sum_ssim += ssim_values.sum()
        self.total_samples += batch_size

    def compute(self) -> torch.Tensor:
        """Compute the final SSIM value.

        Returns:
            SSIM value averaged over all samples. Range: [-1, 1], higher is better.
        """
        return self.sum_ssim / self.total_samples
