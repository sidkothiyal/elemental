"""LPIPS perceptual loss wrapper.

Thin wrapper around the ``lpips`` package's ``LPIPS`` module that:
  * accepts inputs in ``[0, 1]`` (the standard image range used elsewhere in
    this project) and rescales internally to LPIPS's expected ``[-1, 1]``;
  * mean-reduces the per-sample LPIPS scores into a scalar loss.
"""

from __future__ import annotations

import lpips
import torch
from torch import nn


class LPIPSLoss(nn.Module):
    """LPIPS perceptual loss for images in ``[0, 1]``.

    Args:
        net: Backbone network for LPIPS. ``"vgg"`` (default) is the standard
            choice for image generation/compression; ``"alex"`` is faster but
            less accurate.
    """

    def __init__(self, net: str = "vgg") -> None:
        super().__init__()
        self.lpips = lpips.LPIPS(net=net)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.lpips(pred, target, normalize=True).mean()
