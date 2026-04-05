"""Next-frame prediction model using a pretrained autoencoder.

Encodes N input frames through the encoder of a pretrained autoencoder,
aggregates the latent representations via a configurable combiner, and
decodes the combined latent to predict the next frame.
"""

import logging
from pathlib import Path

import torch
from torch import nn

logger = logging.getLogger(__name__)


class FramePredictor(nn.Module):  # pylint: disable=too-few-public-methods
    """Predict the next frame from a sequence of input frames.

    Architecture::

        input (B, N, C, H, W)
          │
          ├─ frame_0 ─► encoder ─► z_0  ─┐
          ├─ frame_1 ─► encoder ─► z_1  ─┤
          ...                             ├─► combiner ─► z_combined ─► decoder ─► pred (B, C, H, W)
          └─ frame_N ─► encoder ─► z_N  ─┘

    The encoder and decoder can be loaded from a pretrained autoencoder
    checkpoint (e.g. one trained with ``compression_ae_oxe``).  Both can
    optionally be frozen.

    Args:
        encoder: Encoder module (image → latent).
        decoder: Decoder module (latent → image).
        combiner: Module that takes a sequence of latents ``(B, N, D...)``
            and returns a single latent ``(B, D...)``.
        pretrained_checkpoint: Optional path to a checkpoint containing
            an autoencoder whose weights should be loaded into the encoder
            and decoder.
        encoder_key: Key prefix for encoder weights in the checkpoint
            state dict (default ``"0"`` for ``nn.Sequential`` index 0).
        decoder_key: Key prefix for decoder weights in the checkpoint
            state dict (default ``"1"`` for ``nn.Sequential`` index 1 —
            the first decoder layer; set to ``"1,2"`` to load layers 1
            and 2 together as the decoder when the AE is a 3-module
            Sequential: encoder, upsample-block, final-conv).
        freeze_encoder: Whether to freeze encoder weights.
        freeze_decoder: Whether to freeze decoder weights.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        combiner: nn.Module,
        pretrained_checkpoint: str | None = None,
        encoder_key: str = "0",
        decoder_key: str = "1,2",
        freeze_encoder: bool = False,
        freeze_decoder: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.combiner = combiner

        if pretrained_checkpoint is not None:
            self._load_pretrained(pretrained_checkpoint, encoder_key, decoder_key)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info("Encoder weights frozen.")

        if freeze_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False
            logger.info("Decoder weights frozen.")

    def _load_pretrained(self, checkpoint_path: str, encoder_key: str, decoder_key: str):
        """Load encoder/decoder weights from a pretrained autoencoder checkpoint."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {path}")

        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))

        encoder_state = _extract_submodule_state(state_dict, encoder_key)
        decoder_keys = [k.strip() for k in decoder_key.split(",")]
        decoder_state = _extract_multi_submodule_state(state_dict, decoder_keys)

        missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
        if missing:
            logger.warning("Encoder missing keys: %s", missing)
        if unexpected:
            logger.warning("Encoder unexpected keys: %s", unexpected)

        missing, unexpected = self.decoder.load_state_dict(decoder_state, strict=False)
        if missing:
            logger.warning("Decoder missing keys: %s", missing)
        if unexpected:
            logger.warning("Decoder unexpected keys: %s", unexpected)

        logger.info("Loaded pretrained AE weights from %s", checkpoint_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input frames of shape ``(B, N, C, H, W)``.

        Returns:
            Predicted next frame of shape ``(B, C, H, W)``.
        """
        b, n, c, h, w = x.shape

        # Encode each frame independently (share weights)
        # Reshape to (B*N, C, H, W) for batch encoding
        frames_flat = x.reshape(b * n, c, h, w)
        latents_flat = self.encoder(frames_flat)

        # Reshape back: (B, N, *latent_shape)
        latent_shape = latents_flat.shape[1:]
        latents = latents_flat.reshape(b, n, *latent_shape)

        # Combine N latents → single latent
        combined = self.combiner(latents)

        # Decode to predicted frame
        return self.decoder(combined)


class LatentSequenceCombiner(nn.Module):  # pylint: disable=too-few-public-methods
    """Combine a sequence of latents along the time dimension.

    Supports multiple strategies for aggregating ``(B, N, C, H, W)``
    latent sequences into a single ``(B, C, H, W)`` latent.

    Args:
        mode: Aggregation strategy:
            - ``"mean"``: Average across time.
            - ``"last"``: Use only the last frame's latent.
            - ``"conv1d"``: Learnable 1×1 conv across the time dimension.
            - ``"attention"``: Learnable attention pooling over time.
        num_frames: Required for ``"conv1d"`` and ``"attention"`` modes.
        latent_channels: Required for ``"attention"`` mode.
    """

    def __init__(
        self,
        mode: str = "conv1d",
        num_frames: int | None = None,
        latent_channels: int | None = None,
    ):
        super().__init__()
        self.mode = mode

        if mode == "conv1d":
            if num_frames is None:
                raise ValueError("num_frames required for conv1d mode")
            # Collapse time dim: (B, N, C, H, W) → treat N as input channels
            # of a 1×1 conv: (B*C, N, H*W) → (B*C, 1, H*W)
            self.time_conv = nn.Conv1d(num_frames, 1, kernel_size=1)
        elif mode == "attention":
            if latent_channels is None:
                raise ValueError("latent_channels required for attention mode")
            # Attention over time: query = learnable, keys/values = frame latents
            self.query = nn.Parameter(torch.randn(1, 1, latent_channels))
            self.attn = nn.MultiheadAttention(
                embed_dim=latent_channels, num_heads=1, batch_first=True
            )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:  # pylint: disable=too-many-locals
        """Combine latents.

        Args:
            latents: ``(B, N, C, H, W)`` tensor of per-frame latents.

        Returns:
            ``(B, C, H, W)`` combined latent.
        """
        if self.mode == "mean":
            return latents.mean(dim=1)

        if self.mode == "last":
            return latents[:, -1]

        if self.mode == "conv1d":
            b, n, c, h, w = latents.shape
            # (B, N, C, H, W) → (B*C, N, H*W) → conv1d → (B*C, 1, H*W) → (B, C, H, W)
            x = latents.permute(0, 2, 1, 3, 4).reshape(b * c, n, h * w)
            x = self.time_conv(x)  # (B*C, 1, H*W)
            return x.reshape(b, c, h, w)

        if self.mode == "attention":
            b, _n, c, _h, _w = latents.shape
            query = self.query.expand(b, -1, -1)  # (B, 1, C)
            # Pool spatial for keys: (B, N, C) via spatial mean
            keys = latents.mean(dim=(-2, -1))  # (B, N, C)
            out, _ = self.attn(query, keys, keys)  # (B, 1, C)
            # Use attention output as channel-wise weighting
            weights = torch.softmax(out.squeeze(1), dim=-1)  # (B, C)
            # Weight the mean latent
            mean_latent = latents.mean(dim=1)  # (B, C, H, W)
            return mean_latent * weights[:, :, None, None]

        raise ValueError(f"Unknown combiner mode: {self.mode}")


def _extract_submodule_state(state_dict: dict, prefix: str) -> dict:
    """Extract and strip-prefix weights for a submodule."""
    prefix_dot = f"{prefix}."
    return {k[len(prefix_dot) :]: v for k, v in state_dict.items() if k.startswith(prefix_dot)}


def _extract_multi_submodule_state(state_dict: dict, prefixes: list[str]) -> dict:
    """Extract weights from multiple sequential submodules and re-index them.

    For a 3-module ``nn.Sequential`` AE (encoder=0, upsample=1, final_conv=2),
    passing ``prefixes=["1", "2"]`` extracts modules 1 and 2 and re-indexes
    them as 0 and 1 in the output state dict, so they can be loaded into a
    new ``nn.Sequential(upsample_block, final_conv)``.
    """
    result = {}
    for new_idx, prefix in enumerate(prefixes):
        prefix_dot = f"{prefix}."
        for k, v in state_dict.items():
            if k.startswith(prefix_dot):
                new_key = f"{new_idx}.{k[len(prefix_dot):]}"
                result[new_key] = v
    return result
