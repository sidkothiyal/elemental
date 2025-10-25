"""Vector Quantized Variational Autoencoder."""

from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn


class Quantizer(nn.Module):
    """Quantizer module."""

    def __init__(self, codebook_size: int, codebook_dim: int):
        """Initialize the Quantizer.

        Args:
            codebook_size: The size of the codebook.
            codebook_dim: The dimension of the codebook.
        """
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.codebook = nn.Parameter(torch.randn(codebook_size, codebook_dim))

        self.dictionary_loss_fn = nn.MSELoss(reduction="mean")
        self.commitment_loss_fn = nn.MSELoss(reduction="mean")
        self.dictionary_loss = None
        self.commitment_loss = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Quantizer."""
        flattened_x = x.permute(0, 2, 3, 1).reshape(-1, self.codebook_dim)
        distances = torch.cdist(flattened_x, self.codebook)
        closest_indices = torch.argmin(distances, dim=1)
        quantized = self.codebook[closest_indices].view(
            x.shape[0], x.shape[2], x.shape[3], self.codebook_dim
        )
        quantized = quantized.permute(0, 3, 1, 2)

        self.dictionary_loss = self.dictionary_loss_fn(quantized, x.detach())
        self.commitment_loss = self.commitment_loss_fn(x, quantized.detach())

        return x + (quantized - x).detach()


class VQVAE(nn.Module):
    """Vector Quantized Variational Autoencoder."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        codebook_size: int,
        codebook_dim: int,
    ):
        """Initialize the VQVAE.

        Args:
            encoder: The encoder module.
            decoder: The decoder module.
            codebook_size: The size of the codebook.
            codebook_dim: The dimension of the codebook.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = Quantizer(codebook_size, codebook_dim)

    @property
    def losses(self) -> Dict[str, torch.Tensor]:
        """Losses of the VQVAE."""
        return {
            "dictionary_loss": self.quantizer.dictionary_loss,
            "commitment_loss": self.quantizer.commitment_loss,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the VQVAE.

        Args:
            x: The input tensor.
        """
        encoded = self.encoder(x)
        quantized = self.quantizer(encoded)
        decoded = self.decoder(quantized)
        return decoded


if __name__ == "__main__":
    model = VQVAE(
        encoder=nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
        ),
        decoder=nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, padding=1, stride=2),
            nn.Sigmoid(),
        ),
        codebook_size=128,
        codebook_dim=512,
    )

    input_data = torch.randn(2, 3, 480, 640).to("cuda")
    model.to("cuda")
    output_data = model(input_data)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    print(output_data.shape)

    for _ in range(10):
        output_data = model(input_data)
        print(model.losses)

        loss = sum(list(model.losses.values())) + F.mse_loss(output_data, input_data)
        loss.backward()
        print(loss)
        optimizer.step()
        optimizer.zero_grad()
        print("=" * 100)
