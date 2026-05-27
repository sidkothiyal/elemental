"""Tensor transforms for use in callback configs.

These modules plug into ``MetricsCallback.transform_pred`` /
``MetricsCallback.transform_target`` to bridge model outputs whose layout does
not match the metric's expected input shape (e.g. video predictions that need
to be flattened across the temporal axis before per-frame metrics are
computed).

They are :class:`torch.nn.Module` subclasses so Hydra ``_target_:``
instantiation works without ``hydra.utils.get_method``.
"""

import torch
from torch import nn


class FlattenTemporal(nn.Module):
    """Fold a temporal axis into the batch dimension.

    Maps ``(B, T, C, H, W) -> (B*T, C, H, W)`` so per-frame image metrics
    (PSNR, SSIM, MSE, ...) can be applied directly to video predictions and
    targets. 4-D and 3-D inputs are passed through unchanged so the same
    callback configuration is robust to image batches as well.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Collapse the leading two dimensions when input is 5-D.

        Args:
            x: Input tensor. Supported ranks: 3, 4, or 5.

        Returns:
            ``(B*T, C, H, W)`` for 5-D inputs; *x* unchanged for 3-D / 4-D.

        Raises:
            ValueError: If ``x`` has fewer than 3 dimensions.
        """
        if x.ndim == 5:
            return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
        if x.ndim in (3, 4):
            return x
        raise ValueError(
            f"FlattenTemporal expects a tensor of rank 3, 4, or 5; got rank {x.ndim} "
            f"with shape {tuple(x.shape)}"
        )
