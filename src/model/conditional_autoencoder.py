"""Autoencoder"""
import torch
from torch import nn


class ConditionalAutoencoder(nn.Module):
    """Conditional Autoencoder"""

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        """Initialize the Conditional Autoencoder."""
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        aux_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of the Conditional Autoencoder."""
        encoded = self.encoder(x, t, aux_features)
        decoded = self.decoder(encoded, t, aux_features)
        return decoded
