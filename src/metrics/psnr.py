"""Peak Signal-to-Noise Ratio (PSNR) metric."""

import torch
from torchmetrics import Metric


class PSNR(Metric):
    """Peak Signal-to-Noise Ratio metric for image reconstruction tasks.

    PSNR is computed as: 10 * log10(data_range^2 / MSE)
    Higher values indicate better reconstruction quality.
    Typical values for 8-bit images: 20-40 dB.
    """

    higher_is_better = True
    full_state_update = False

    def __init__(
        self,
        data_range: float = 1.0,
        reduction: str = "elementwise_mean",
        eps: float = 1e-8,
    ):
        """Initialize the PSNR metric.

        Args:
            data_range: The data range of the input images (max - min).
                Default is 1.0 for normalized images [0, 1].
                Use 255.0 for uint8 images [0, 255].
            reduction: How to reduce over batch dimension.
                'elementwise_mean': Return mean PSNR over batch.
                'sum': Return sum of PSNR over batch.
                'none': Return PSNR for each sample.
            eps: Small constant to avoid log(0).
        """
        super().__init__()
        self.data_range = data_range
        self.reduction = reduction
        self.eps = eps

        self.add_state("sum_psnr", default=torch.tensor(0.0), dist_reduce_fx="sum")
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

        batch_size = preds.shape[0]

        # Compute MSE per sample
        preds_flat = preds.reshape(batch_size, -1)
        target_flat = target.reshape(batch_size, -1)
        mse = torch.mean((preds_flat - target_flat) ** 2, dim=1)

        # Compute PSNR: 10 * log10(data_range^2 / MSE)
        # Add eps to avoid log(0) for perfect reconstruction
        psnr = 10 * torch.log10((self.data_range**2) / (mse + self.eps))

        self.sum_psnr += psnr.sum()
        self.total_samples += batch_size

    def compute(self) -> torch.Tensor:
        """Compute the final PSNR value.

        Returns:
            PSNR value in decibels (dB), averaged over all samples.
        """
        return self.sum_psnr / self.total_samples
