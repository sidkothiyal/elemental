"""Patcher and Unpatcher modules."""

import einops
import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import nn


class Patcher(nn.Module):
    """Patcher: convert images to patch embeddings.

    Supports overlapping patch extraction to provide each patch with more context
    from neighboring regions. When overlap > 0, larger patches are extracted
    using F.unfold with stride equal to the base patch size.
    """

    def __init__(
        self,
        image_size: tuple[int, int, int],
        patch_size: tuple[int, int],
        embed_dim: int,
        conv_encoder: nn.Module | None = None,
        overlap: int = 0,
    ):
        """Initialize the Patcher.

        Args:
            image_size: Tuple of (channels, height, width) for input images.
            patch_size: Tuple of (patch_height, patch_width).
            embed_dim: Output embedding dimension per patch.
            conv_encoder: Optional conv network applied before patching.
            overlap: Number of pixels to overlap between adjacent patches (each side).
                     When > 0, extracts larger patches with context from neighbors.
                     Typical values: 2-8 pixels (must be < patch_size // 2).
        """
        super().__init__()
        channels, height, width = image_size
        patch_height, patch_width = patch_size

        assert height % patch_height == 0, "Image height must be divisible by patch height"
        assert width % patch_width == 0, "Image width must be divisible by patch width"

        self.conv_encoder = conv_encoder
        self.overlap = overlap
        self.channels = channels
        self.height = height
        self.width = width

        # Grid dimensions (number of patches)
        self.num_patches_h = height // patch_height
        self.num_patches_w = width // patch_width
        num_patches = self.num_patches_h * self.num_patches_w

        # Stride for unfold (distance between patch centers)
        self.stride_h = patch_height
        self.stride_w = patch_width

        if overlap > 0:
            # With overlap, extract larger patches
            assert overlap < patch_height // 2, "Overlap must be less than half the patch height"
            assert overlap < patch_width // 2, "Overlap must be less than half the patch width"

            self.kernel_h = patch_height + 2 * overlap
            self.kernel_w = patch_width + 2 * overlap
            patch_dim = channels * self.kernel_h * self.kernel_w

            # Projection layers for overlapping patches
            self.patch_norm = nn.LayerNorm(patch_dim)
            self.patch_proj = nn.Linear(patch_dim, embed_dim)
            self.embed_norm = nn.LayerNorm(embed_dim)
            self.to_patch_embedding = None  # Not used in overlap mode
        else:
            self.kernel_h = patch_height
            self.kernel_w = patch_width
            patch_dim = channels * patch_height * patch_width

            self.to_patch_embedding = nn.Sequential(
                # Patchifies (h w = num_patches) and flatten the patches (pH pW C)
                # (B, C, H , W) = b c (h pH) (w pW)
                # h = H/pH, w = W/pW
                Rearrange(
                    "b c (h pH) (w pW) -> b (h w) (pH pW c)", pH=patch_height, pW=patch_width
                ),
                nn.LayerNorm(patch_dim),
                # Linear projection (to map to D dimensions) --> patch embeddings
                nn.Linear(patch_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            )
            self.patch_norm = None
            self.patch_proj = None
            self.embed_norm = None

        # 1D sequence of token embeddings
        self.pos_embeddings = nn.Parameter(torch.randn(1, num_patches, embed_dim))

    def _extract_overlapping_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Extract overlapping patches using F.unfold.

        Uses reflection padding to handle boundaries, then unfold to extract
        patches with the specified overlap.
        """
        # Pad image to handle boundary overlap (reflection padding for smooth edges)
        x_padded = F.pad(
            x, (self.overlap, self.overlap, self.overlap, self.overlap), mode="reflect"
        )

        # Unfold extracts sliding local blocks
        # Input: (B, C, H_padded, W_padded)
        # Output: (B, C * kernel_h * kernel_w, num_patches)
        patches = F.unfold(
            x_padded,
            kernel_size=(self.kernel_h, self.kernel_w),
            stride=(self.stride_h, self.stride_w),
        )

        # Reshape to (B, num_patches, C * kernel_h * kernel_w)
        patches = patches.transpose(1, 2)

        # Apply projection layers
        patches = self.patch_norm(patches)
        patches = self.patch_proj(patches)
        patches = self.embed_norm(patches)

        return patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: convert image to patch embeddings."""
        if self.conv_encoder:
            x = self.conv_encoder(x)

        if self.overlap > 0:
            patched_x = self._extract_overlapping_patches(x)
        else:
            patched_x = self.to_patch_embedding(x)

        patched_x = patched_x + self.pos_embeddings
        return patched_x


def _create_hann_window_2d(
    height: int, width: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Create a 2D Hann window for smooth blending of overlapping patches."""
    # Create 1D Hann windows
    hann_h = torch.hann_window(height, periodic=False, device=device, dtype=dtype)
    hann_w = torch.hann_window(width, periodic=False, device=device, dtype=dtype)
    # Outer product to create 2D window
    return hann_h.unsqueeze(1) * hann_w.unsqueeze(0)


class Unpatcher(nn.Module):
    """Unpatcher: convert patch embeddings back to images.

    Supports overlapping patch reconstruction to reduce grid artifacts.
    When overlap > 0, patches are projected to larger sizes and blended
    using a Hann window for smooth transitions at boundaries.
    """

    def __init__(
        self,
        image_size: tuple[int, int, int],
        patch_size: tuple[int, int],
        embed_dim: int,
        decoder_projection_dim: int | None = None,
        conv_decoder: nn.Module | None = None,
        model_downsample_factor: int = 1,
        overlap: int = 0,
    ):
        """Initialize the Unpatcher.

        Args:
            image_size: Tuple of (channels, height, width) for output image.
            patch_size: Tuple of (patch_height, patch_width).
            embed_dim: Dimension of input embeddings.
            decoder_projection_dim: Number of channels before conv_decoder (default: channels).
            conv_decoder: Optional convolutional decoder applied after unpatchifying.
            model_downsample_factor: Factor by which the model downsamples spatially.
            overlap: Number of pixels to overlap between adjacent patches (each side).
                     When > 0, enables smooth blending to reduce grid artifacts.
                     Typical values: 2-8 pixels (must be < patch_size // 2).
        """
        super().__init__()
        channels, height, width = image_size
        patch_height, patch_width = patch_size

        self.height = height
        self.width = width
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.model_downsample_factor = model_downsample_factor
        self.overlap = overlap
        self.decoder_projection_dim = decoder_projection_dim or channels

        self.conv_decoder = conv_decoder

        # Effective patch size after downsampling
        effective_patch_h = patch_height // model_downsample_factor
        effective_patch_w = patch_width // model_downsample_factor

        if overlap > 0:
            # With overlap, we project to larger patches
            # Each patch extends by 'overlap' pixels on each side
            assert (
                overlap < effective_patch_h // 2
            ), "Overlap must be less than half the patch height"
            assert (
                overlap < effective_patch_w // 2
            ), "Overlap must be less than half the patch width"

            self.output_patch_h = effective_patch_h + (2 * overlap)
            self.output_patch_w = effective_patch_w + (2 * overlap)
            decoded_patch_dim = (
                self.decoder_projection_dim * self.output_patch_h * self.output_patch_w
            )
        else:
            self.output_patch_h = effective_patch_h
            self.output_patch_w = effective_patch_w
            decoded_patch_dim = self.decoder_projection_dim * effective_patch_h * effective_patch_w

        self.embedding_to_patch_projection = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, decoded_patch_dim),
        )

        # Pre-compute grid dimensions
        self.num_patches_h = height // patch_height
        self.num_patches_w = width // patch_width
        self.effective_height = height // model_downsample_factor
        self.effective_width = width // model_downsample_factor

        # Stride for fold operation (distance between patch centers)
        self.stride_h = effective_patch_h
        self.stride_w = effective_patch_w

        # Register blend window as buffer (created on first forward)
        self.register_buffer("_blend_window", None, persistent=False)
        # Pre-computed divisor for normalization (created on first forward)
        self.register_buffer("_fold_divisor", None, persistent=False)

    def _ensure_blend_buffers(self, device: torch.device, dtype: torch.dtype) -> None:
        """Ensure blending buffers are initialized for the given device/dtype."""
        if self._blend_window is not None and self._blend_window.device == device:
            return

        # Create 2D Hann window
        blend_window = _create_hann_window_2d(
            self.output_patch_h, self.output_patch_w, device, dtype
        )
        # Store as (1, pH*pW, 1) for broadcasting with fold input
        self._blend_window = blend_window.flatten().view(1, -1, 1)

        # Pre-compute the divisor by folding ones weighted by the blend window
        # This gives us the sum of weights at each pixel location
        num_patches = self.num_patches_h * self.num_patches_w
        ones_weighted = self._blend_window.expand(1, -1, num_patches)

        # Padded output size for fold (to handle boundary overlap)
        padded_h = self.effective_height + 2 * self.overlap
        padded_w = self.effective_width + 2 * self.overlap

        divisor = F.fold(
            ones_weighted,
            output_size=(padded_h, padded_w),
            kernel_size=(self.output_patch_h, self.output_patch_w),
            stride=(self.stride_h, self.stride_w),
        )
        # Crop to actual output size and clamp to avoid division by zero
        self._fold_divisor = divisor[
            :,
            :,
            self.overlap : self.overlap + self.effective_height,
            self.overlap : self.overlap + self.effective_width,
        ].clamp(min=1e-8)

    def _reconstruct_with_overlap(self, decoded_tokens: torch.Tensor) -> torch.Tensor:
        """Reconstruct image using overlapping patches with smooth blending.

        Uses F.fold for efficient GPU-accelerated accumulation instead of loops.
        Complexity: O(B * C * num_patches * patch_size) with optimized CUDA kernels.
        """
        batch_size = decoded_tokens.shape[0]
        device = decoded_tokens.device
        dtype = decoded_tokens.dtype

        # Ensure blending buffers exist
        self._ensure_blend_buffers(device, dtype)

        # Reshape: (B, num_patches, C*pH*pW) -> (B, C, pH*pW, num_patches)
        patches = decoded_tokens.view(
            batch_size,
            self.num_patches_h * self.num_patches_w,
            self.decoder_projection_dim,
            self.output_patch_h * self.output_patch_w,
        )
        patches = patches.permute(0, 2, 3, 1)  # (B, C, pH*pW, num_patches)

        # Apply blend window weights
        blend_window = self._blend_window.to(dtype=dtype)
        patches_weighted = patches * blend_window  # (B, C, pH*pW, num_patches)

        # Reshape for fold: (B, C, pH*pW, num_patches) -> (B*C, pH*pW, num_patches)
        # Use reshape instead of view since permute makes the tensor non-contiguous
        patches_weighted = patches_weighted.reshape(
            batch_size * self.decoder_projection_dim,
            self.output_patch_h * self.output_patch_w,
            -1,
        )

        # Padded output size (patches extend beyond boundaries by 'overlap')
        padded_h = self.effective_height + 2 * self.overlap
        padded_w = self.effective_width + 2 * self.overlap

        # Fold patches back to image (accumulates overlapping regions)
        output_padded = F.fold(
            patches_weighted,
            output_size=(padded_h, padded_w),
            kernel_size=(self.output_patch_h, self.output_patch_w),
            stride=(self.stride_h, self.stride_w),
        )

        # Reshape back: (B*C, 1, padded_h, padded_w) -> (B, C, padded_h, padded_w)
        output_padded = output_padded.view(
            batch_size, self.decoder_projection_dim, padded_h, padded_w
        )

        # Crop to actual output size (remove padding from overlap at boundaries)
        output = output_padded[
            :,
            :,
            self.overlap : self.overlap + self.effective_height,
            self.overlap : self.overlap + self.effective_width,
        ]

        # Normalize by accumulated weights
        output = output / self._fold_divisor.to(dtype=dtype)

        return output

    def forward(self, decoded_tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass: convert patch embeddings to image."""
        decoded_tokens = self.embedding_to_patch_projection(decoded_tokens)

        if self.overlap > 0:
            # Use overlapping reconstruction with blending
            image = self._reconstruct_with_overlap(decoded_tokens)
        else:
            # Original non-overlapping reconstruction
            image = einops.rearrange(
                decoded_tokens,
                "b (h w) (pH pW c) -> b c (h pH) (w pW)",
                pH=self.output_patch_h,
                pW=self.output_patch_w,
                h=self.num_patches_h,
                w=self.num_patches_w,
            )

        if self.conv_decoder:
            image = self.conv_decoder(image)

        return image


class PatcherUnpatcher(nn.Module):
    """PatcherUnpatcher: combine Patcher and Unpatcher with a model."""

    def __init__(
        self,
        image_size: tuple[int, int, int],
        patch_size: tuple[int, int],
        embed_dim: int,
        model: nn.Module,
        conv_encoder: nn.Module | None = None,
        decoder_projection_dim: int | None = None,
        conv_decoder: nn.Module | None = None,
        model_downsample_factor: int = 1,
    ):
        """Initialize the PatcherUnpatcher."""
        super().__init__()
        self.patcher = Patcher(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            conv_encoder=conv_encoder,
        )
        self.model = model
        self.unpatcher = Unpatcher(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            decoder_projection_dim=decoder_projection_dim,
            conv_decoder=conv_decoder,
            model_downsample_factor=model_downsample_factor,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the PatcherUnpatcher."""
        patched_x = self.patcher(x)
        decoded_tokens = self.model(patched_x)
        image = self.unpatcher(decoded_tokens)
        return image
