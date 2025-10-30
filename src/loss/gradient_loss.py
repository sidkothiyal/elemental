"""Gradient Loss"""

from torch import nn


class GradientLoss(nn.Module):
    """Gradient Loss"""

    def __init__(self):
        """Initialize the GradientLoss"""
        super().__init__()
        self.loss_fn = nn.L1Loss()

    def forward(self, pred, target):
        """Forward pass"""
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

        return self.loss_fn(pred_dx, target_dx) + self.loss_fn(pred_dy, target_dy)
