"""Timestep embedding modules for diffusion models."""

import math

import torch
from torch import nn


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding with MLP projection.

    Encodes scalar diffusion timesteps into dense vector representations using
    sinusoidal positional encoding (Vaswani et al., "Attention Is All You Need").
    Geometrically spaced frequencies produce a smooth, continuous embedding that
    allows the model to distinguish nearby timesteps while generalizing across
    the full diffusion schedule. The raw sinusoidal features are projected through
    a 2-layer MLP so the network can learn a task-specific transformation.

    Args:
        dim: Dimensionality of the output embedding. Must be even, since half
            the channels are cosine and half are sine.
    """

    def __init__(self, dim: int):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half) / half)
        self.register_buffer("freqs", freqs)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed a batch of timesteps.

        Args:
            t: Timestep tensor of shape ``(B,)``, where values are typically
                in ``[0, 1]`` or ``[0, T]`` depending on the diffusion schedule.

        Returns:
            Embedding tensor of shape ``(B, dim)``.
        """
        args = t[:, None].float() * self.freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb)


class LearnedTimestepEmbedding(nn.Module):
    """Learned discrete timestep embedding.

    Maps discretized timesteps to learned vectors via ``nn.Embedding``. Simple
    and expressive for fixed schedules, but cannot interpolate to unseen
    timesteps. Best suited for diffusion processes with a fixed number of steps.

    Args:
        num_steps: Number of discrete timesteps (embedding table size).
        dim: Dimensionality of the output embedding.
    """

    def __init__(self, num_steps: int, dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_steps, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed a batch of integer timesteps.

        Args:
            t: Integer timestep tensor of shape ``(B,)`` with values in
                ``[0, num_steps)``.

        Returns:
            Embedding tensor of shape ``(B, dim)``.
        """
        return self.mlp(self.embedding(t.long()))


class FourierTimestepEmbedding(nn.Module):
    """Random Fourier feature timestep embedding.

    Uses a fixed random projection matrix to map timesteps into Fourier
    features (Tancik et al., "Fourier Features Let Networks Learn High
    Frequency Functions in Low Dimensional Domains"). This can capture
    higher-frequency detail than deterministic sinusoidal encodings. The
    random matrix is drawn once at init and frozen.

    Args:
        dim: Dimensionality of the output embedding. Must be even.
        scale: Standard deviation of the random frequency matrix. Higher
            values emphasize higher frequencies. Default: ``1.0``.
    """

    def __init__(self, dim: int, scale: float = 1.0):
        super().__init__()
        half = dim // 2
        freqs = torch.randn(half) * scale
        self.register_buffer("freqs", freqs)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed a batch of timesteps.

        Args:
            t: Timestep tensor of shape ``(B,)``.

        Returns:
            Embedding tensor of shape ``(B, dim)``.
        """
        args = t[:, None].float() * self.freqs[None] * 2 * math.pi
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb)


class AdaLNTimestepEmbedding(nn.Module):
    """Adaptive Layer Norm timestep embedding (DiT-style).

    Instead of producing a single embedding vector, this module outputs
    per-layer **scale** and **shift** parameters that modulate layer
    normalization (Peebles & Xie, "Scalable Diffusion Models with
    Transformers"). The timestep is first encoded with sinusoidal features,
    then an MLP predicts ``(scale, shift)`` pairs for each conditioning site.

    Args:
        dim: Model hidden dimension. The sinusoidal encoding and the MLP
            hidden layer both use this size.
        n_adaln_outputs: Number of scalar outputs. Typically ``6`` for DiT
            (scale, shift, gate for self-attn and FFN), but configurable.
            Default: ``6``.
    """

    def __init__(self, dim: int, n_adaln_outputs: int = 6):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half) / half)
        self.register_buffer("freqs", freqs)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * n_adaln_outputs),
        )
        self.n_adaln_outputs = n_adaln_outputs

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed timesteps into AdaLN modulation parameters.

        Args:
            t: Timestep tensor of shape ``(B,)``.

        Returns:
            Modulation tensor of shape ``(B, n_adaln_outputs, dim)``.
        """
        args = t[:, None].float() * self.freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        out = self.mlp(emb)
        return out.view(t.shape[0], self.n_adaln_outputs, -1)
