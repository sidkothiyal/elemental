"""Vector Quantized Variational Autoencoder."""

from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn


class Quantizer(nn.Module):
    """Quantizer module."""

    def __init__(
        self,
        codebook_size: int,
        codebook_dim: int,
        codebook_hit_summary_frequency: int = 100,
        codebook_revival_threshold: int = 2,
        codebook_revival_enabled: bool = True,
        use_ema: bool = True,
        ema_decay: float = 0.99,
        ema_epsilon: float = 1e-5,
    ):
        """Initialize the Quantizer.

        Args:
            codebook_size: The size of the codebook.
            codebook_dim: The dimension of the codebook.
            codebook_hit_summary_frequency: The frequency to summarize the codebook hits.
            codebook_revival_threshold: Threshold below which a codebook entry is considered unused.
            codebook_revival_enabled: Whether to enable codebook revival.
            use_ema: Whether to use EMA to update codebook entries instead of gradients.
            ema_decay: Decay rate for EMA updates (only used if use_ema=True).
            ema_epsilon: Small constant for numerical stability in EMA updates.
        """
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_epsilon = ema_epsilon

        # When using EMA, codebook is updated via EMA, not gradients
        if use_ema:
            self.register_buffer("codebook", torch.randn(codebook_size, codebook_dim))
            # EMA buffers for cluster statistics
            self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
            self.register_buffer("ema_embed", torch.randn(codebook_size, codebook_dim))
        else:
            self.codebook = nn.Parameter(torch.randn(codebook_size, codebook_dim))

        self.register_buffer("codebook_hits", torch.zeros(codebook_size))
        self._codebook_hit_summary_frequency = codebook_hit_summary_frequency
        self._codebook_revival_threshold = codebook_revival_threshold
        self._codebook_revival_enabled = codebook_revival_enabled
        self._call_count = 0

        self.dictionary_loss_fn = nn.MSELoss(reduction="mean")
        self.commitment_loss_fn = nn.MSELoss(reduction="mean")
        self.dictionary_loss = None
        self.commitment_loss = None

    def update_codebook_with_ema(
        self, flattened_x: torch.Tensor, closest_indices: torch.Tensor
    ) -> None:
        """Update codebook using exponential moving average.

        Args:
            flattened_x: Flattened input tensor of shape [N, codebook_dim].
            closest_indices: Indices of closest codebook entries for each input.
        """
        if not self.use_ema or not self.training:
            return

        # Count how many times each codebook entry was used in this batch
        encodings = F.one_hot(
            closest_indices, num_classes=self.codebook_size
        ).float()  # [N, codebook_size]

        # Sum of assignments for each codebook entry in this batch
        encodings_sum = encodings.sum(0)  # [codebook_size]

        # Sum of encoder outputs for each codebook entry
        embed_sum = encodings.t() @ flattened_x  # [codebook_size, codebook_dim]

        # Update EMA statistics
        with torch.no_grad():
            # Update cluster sizes with EMA
            self.ema_cluster_size.mul_(self.ema_decay).add_(encodings_sum, alpha=1 - self.ema_decay)

            # Update embeddings with EMA
            self.ema_embed.mul_(self.ema_decay).add_(embed_sum, alpha=1 - self.ema_decay)

            # Laplace smoothing for cluster sizes
            n = self.ema_cluster_size.sum()
            cluster_size = (
                (self.ema_cluster_size + self.ema_epsilon)
                / (n + self.codebook_size * self.ema_epsilon)
                * n
            )

            # Update codebook
            self.codebook.copy_(self.ema_embed / cluster_size.unsqueeze(1))

    def revive_dead_codebook_entries(self, flattened_x: torch.Tensor) -> int:
        """Revive unused codebook entries by replacing them with random encodings from the batch.

        Args:
            flattened_x: Flattened input tensor of shape [N, codebook_dim].

        Returns:
            Number of codebook entries that were revived.
        """
        if not self._codebook_revival_enabled:
            return 0

        # Find unused codebook entries
        unused_mask = self.codebook_hits < self._codebook_revival_threshold
        unused_indices = torch.where(unused_mask)[0]
        num_unused = unused_indices.numel()

        if num_unused > 0:
            # Randomly sample encodings from the batch to replace dead codes
            batch_size = flattened_x.shape[0]
            if batch_size > 0:
                # Sample random indices from the batch
                random_batch_indices = torch.randint(
                    0, batch_size, (num_unused,), device=flattened_x.device
                )

                # Replace unused codebook entries with these random encodings
                with torch.no_grad():
                    if self.use_ema:
                        # Update both codebook and EMA statistics
                        self.codebook[unused_indices] = flattened_x[random_batch_indices].clone()
                        self.ema_embed[unused_indices] = flattened_x[random_batch_indices].clone()
                        self.ema_cluster_size[unused_indices] = 1.0
                    else:
                        self.codebook.data[unused_indices] = flattened_x[
                            random_batch_indices
                        ].clone()

                print(
                    f"Revived {num_unused} unused codebook entries "
                    f"(threshold: {self._codebook_revival_threshold})"
                )

        return num_unused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Quantizer."""
        if x.dim() == 4:
            flattened_x = x.permute(0, 2, 3, 1).reshape(-1, self.codebook_dim)
            distances = torch.cdist(flattened_x, self.codebook)
            closest_indices = torch.argmin(distances, dim=1)
            quantized = self.codebook[closest_indices].view(
                x.shape[0], x.shape[2], x.shape[3], self.codebook_dim
            )
            quantized = quantized.permute(0, 3, 1, 2)
        elif x.dim() == 3:
            flattened_x = x.reshape(-1, self.codebook_dim)
            distances = torch.cdist(flattened_x, self.codebook)
            closest_indices = torch.argmin(distances, dim=1)
            quantized = self.codebook[closest_indices].view(
                x.shape[0], x.shape[1], self.codebook_dim
            )
            quantized = quantized

        # Update codebook with EMA if enabled
        self.update_codebook_with_ema(flattened_x, closest_indices)

        self._call_count += 1
        self.codebook_hits[closest_indices] += 1
        if self._call_count % self._codebook_hit_summary_frequency == 0:
            print(f"Codebook hits: {self.codebook_hits}")
            # Revive unused codebook entries before resetting the hits
            self.revive_dead_codebook_entries(flattened_x)
            self.codebook_hits.zero_()

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
