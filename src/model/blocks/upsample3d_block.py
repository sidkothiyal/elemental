"""3D upsample blocks for the model.

3D analogue of :mod:`model.blocks.upsample_block`. Operates on
``(B, C, T, H, W)`` tensors. Default upsample mode is ``"trilinear"``.
"""

from typing import Callable, Literal

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from model import utils

ScaleSpec = float | tuple[float, float, float]
SizeSpec = tuple[int, int, int]
KernelSpec = int | tuple[int, int, int]


class Upsample3dNormAct(nn.Module):
    """Layer for a 3D upsample, optional 3D convolution, normalization, and activation."""

    def __init__(
        self,
        scale_factor: ScaleSpec | None = None,
        size: SizeSpec | None = None,
        mode: str = "nearest",
        align_corners: bool | None = None,
        in_channels: int | None = None,
        out_channels: int | None = None,
        kernel_size: KernelSpec | None = None,
        stride: KernelSpec = 1,
        padding: KernelSpec = 0,
        groups: int | None = 1,
        dilation: KernelSpec | None = 1,
        bias: bool | None = False,
        norm: type[nn.Module] | None = nn.BatchNorm3d,
        norm_kwargs: dict | None = None,
        act: type[nn.Module] | None = nn.ReLU,
        act_kwargs: dict | None = None,
    ):
        """Initialise the 3D upsample-norm-act layer.

        Args:
            scale_factor: Scale factor; ``float`` or 3-tuple ``(sT, sH, sW)``.
            size: Target output size ``(T, H, W)``. If provided, ``scale_factor``
                is ignored.
            mode: Upsampling mode: ``'nearest'`` or ``'trilinear'``.
            align_corners: If True, corner voxels are aligned (only used by
                ``'trilinear'``).
            in_channels: Number of input channels (for optional convolution).
            out_channels: Number of output channels (for optional convolution).
            kernel_size: Kernel size for the optional convolution; ``int`` or
                3-tuple. If ``None``, no convolution is applied.
            stride: Stride for the convolution.
            padding: Padding for the convolution.
            groups: Number of groups in the convolution.
            dilation: Dilation for the convolution.
            bias: Whether to use a bias in the convolution.
            norm: Normalization class (e.g., nn.BatchNorm3d, nn.GroupNorm, None).
            norm_kwargs: Keyword arguments for the normalization layer.
            act: Activation class (e.g., nn.ReLU, nn.SiLU, None).
            act_kwargs: Keyword arguments for the activation layer.
        """
        super().__init__()

        if scale_factor is None and size is None:
            raise ValueError("Either scale_factor or size must be provided for upsampling")

        # Coerce ListConfig / list / tuple to tuple so torch.nn.Upsample accepts it.
        # A bare int/float is passed through unchanged.
        if scale_factor is not None and not isinstance(scale_factor, (int, float)):
            vals = tuple(float(v) for v in scale_factor)
            scale_factor = (vals[0], vals[1], vals[2])

        self.upsample = nn.Upsample(
            scale_factor=scale_factor,
            size=tuple(size) if size is not None else None,
            mode=mode,
            align_corners=align_corners,
        )

        self.conv: nn.Conv3d | None = None
        norm_channels: int | None = None

        if kernel_size is not None:
            if in_channels is None or out_channels is None:
                raise ValueError(
                    "in_channels and out_channels must be provided when using convolution"
                )
            self.conv = nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation if dilation is not None else 1,
                groups=groups if groups is not None else 1,
                bias=bool(bias) if bias is not None else False,
            )

        if out_channels is not None:
            norm_channels = out_channels
        elif in_channels is not None:
            norm_channels = in_channels

        self.norm = utils.create_normalization(norm, norm_channels, norm_kwargs)
        self.act = utils.create_activation(act, act_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.upsample(x)
        if self.conv is not None:
            x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class Upsample3dBlock(nn.Module):
    """Block of :class:`Upsample3dNormAct` layers."""

    def __init__(
        self,
        scale_factors: list[ScaleSpec] | None = None,
        sizes: list[SizeSpec] | None = None,
        modes: list[str] | str = "nearest",
        align_corners: list[bool | None] | bool | None = None,
        in_channels: int | None = None,
        channels: list[int] | None = None,
        kernel_sizes: list[KernelSpec] | None = None,
        strides: list[KernelSpec] | None = None,
        paddings: list[KernelSpec] | None = None,
        groups: list[int] | None = None,
        dilations: list[KernelSpec] | None = None,
        bias: bool | None = False,
        norm: type[nn.Module] | None = nn.BatchNorm3d,
        norm_kwargs: dict | None = None,
        act: type[nn.Module] | None = nn.ReLU,
        act_kwargs: dict | None = None,
        checkpoint_layers: list[int] | Literal["all"] | None = None,
        checkpoint_use_reentrant: bool = False,
    ):
        """Initialise the block.

        Args mirror :class:`model.blocks.upsample_block.UpsampleBlock` but with
        3-tuple specs and default ``norm=nn.BatchNorm3d``.
        """
        super().__init__()

        if scale_factors is None and sizes is None:
            raise ValueError("Either scale_factors or sizes must be provided for upsampling")
        if scale_factors is not None and sizes is not None:
            raise ValueError("Cannot provide both scale_factors and sizes")

        if scale_factors is not None:
            self.num_layers = len(scale_factors)
        else:
            assert sizes is not None
            self.num_layers = len(sizes)

        self.scale_factors = scale_factors
        self.sizes = sizes
        self.modes = modes
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.paddings = paddings
        self.groups = groups
        self.dilations = dilations
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.act = act
        self.act_kwargs = act_kwargs

        if isinstance(self.modes, str):
            self.modes = [self.modes] * self.num_layers
        if isinstance(self.align_corners, bool):
            self.align_corners = [self.align_corners] * self.num_layers
        elif self.align_corners is None:
            self.align_corners = [None] * self.num_layers

        self._setup_checkpointing(checkpoint_layers, checkpoint_use_reentrant)

        self.assert_same_lengths()
        self.upsample_blocks = nn.ModuleList()

        for idx in range(self.num_layers):
            scale_factor = self.scale_factors[idx] if self.scale_factors is not None else None
            size = self.sizes[idx] if self.sizes is not None else None
            mode = self.modes[idx]
            align_corner = self.align_corners[idx]

            layer_in_channels: int | None = None
            layer_out_channels: int | None = None
            layer_kernel_size: KernelSpec | None = None

            if self.channels is not None:
                layer_in_channels = self.in_channels if idx == 0 else self.channels[idx - 1]
                layer_out_channels = self.channels[idx]
                if self.kernel_sizes is not None:
                    layer_kernel_size = self.kernel_sizes[idx]

            layer_stride = self.strides[idx] if self.strides else 1
            layer_padding = self.paddings[idx] if self.paddings else 0
            layer_group = self.groups[idx] if self.groups else 1
            layer_dilation = self.dilations[idx] if self.dilations else 1

            self.upsample_blocks.append(
                Upsample3dNormAct(
                    scale_factor=scale_factor,
                    size=size,
                    mode=mode,
                    align_corners=align_corner,
                    in_channels=layer_in_channels,
                    out_channels=layer_out_channels,
                    kernel_size=layer_kernel_size,
                    stride=layer_stride,
                    padding=layer_padding,
                    groups=layer_group,
                    dilation=layer_dilation,
                    bias=bias,
                    norm=norm,
                    norm_kwargs=norm_kwargs,
                    act=act,
                    act_kwargs=act_kwargs,
                )
            )

    def _setup_checkpointing(
        self,
        checkpoint_layers: list[int] | Literal["all"] | None,
        checkpoint_use_reentrant: bool,
    ) -> None:
        """Setup gradient checkpointing configuration."""
        self.checkpoint_use_reentrant = checkpoint_use_reentrant

        if checkpoint_layers is None:
            self.checkpoint_layers: set[int] = set()
        elif checkpoint_layers == "all":
            self.checkpoint_layers = set(range(self.num_layers))
        else:
            for layer_idx in checkpoint_layers:
                if not 0 <= layer_idx < self.num_layers:
                    raise ValueError(
                        f"Invalid layer index {layer_idx}. Must be in range [0, {self.num_layers})"
                    )
            self.checkpoint_layers = set(checkpoint_layers)

    def _should_checkpoint(self, layer_idx: int) -> bool:
        """Check if a layer should use gradient checkpointing."""
        return self.training and layer_idx in self.checkpoint_layers

    def _run_module(
        self,
        fn: nn.Module | Callable[[torch.Tensor], torch.Tensor],
        x: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Run a module with optional gradient checkpointing."""
        if self._should_checkpoint(layer_idx):
            return checkpoint(fn, x, use_reentrant=self.checkpoint_use_reentrant)
        return fn(x)

    def assert_same_lengths(self):
        """Assert that all per-layer lists have matching length."""
        num_layers = self.num_layers

        utils.assert_list_length(self.scale_factors, num_layers, "scale_factors")
        utils.assert_list_length(self.sizes, num_layers, "sizes")
        utils.assert_list_length(self.modes, num_layers, "modes")
        utils.assert_list_length(self.align_corners, num_layers, "align_corners")
        utils.assert_list_length(self.channels, num_layers, "channels")
        utils.assert_list_length(self.kernel_sizes, num_layers, "kernel_sizes")
        utils.assert_list_length(self.strides, num_layers, "strides")
        utils.assert_list_length(self.paddings, num_layers, "paddings")
        utils.assert_list_length(self.groups, num_layers, "groups")
        utils.assert_list_length(self.dilations, num_layers, "dilations")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(B, C, T, H, W)`` tensor.

        Returns:
            ``(B, C', T', H', W')`` tensor.
        """
        for idx, upsample_block in enumerate(self.upsample_blocks):
            x = self._run_module(upsample_block, x, idx)
        return x


if __name__ == "__main__":
    model = Upsample3dBlock(
        scale_factors=[(1, 2, 2), (2, 2, 2), (1, 2, 2)],
        modes="trilinear",
        align_corners=False,
        in_channels=256,
        channels=[128, 64, 32],
        kernel_sizes=[(3, 3, 3), (3, 3, 3), (3, 3, 3)],
        paddings=[(1, 1, 1), (1, 1, 1), (1, 1, 1)],
    )

    input_data = torch.randn(2, 256, 4, 8, 8)
    output_data = model(input_data)
    print(output_data.shape)
