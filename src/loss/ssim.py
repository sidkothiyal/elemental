"""SSIM Loss uses implementation from pytorch-msssim."""

import pytorch_msssim
import torch
from torch import nn


class SSIMLoss(nn.Module):
    """SSIM Loss."""

    def __init__(self, multi_scale: bool = True, **ssim_kwargs):
        """Initialize the SSIMLoss.

        Args:
            multi_scale: Whether to use multi-scale SSIM.
            **ssim_kwargs: Keyword arguments forwarded to pytorch_msssim.MS_SSIM
                          or pytorch_msssim.SSIM. Defaults: data_range=1.0,
                          size_average=True, channel=3.
        """
        super().__init__()
        ssim_module = pytorch_msssim.MS_SSIM if multi_scale else pytorch_msssim.SSIM

        defaults = {"data_range": 1.0, "size_average": True, "channel": 3}
        defaults.update(ssim_kwargs)
        self.ssim = ssim_module(**defaults)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            pred: The predicted tensor.
            target: The target tensor.
        """
        return 1.0 - self.ssim(pred, target)
