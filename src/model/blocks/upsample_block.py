"""Upsample blocks for the model."""

from typing import Callable, Literal

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from model import utils


class UpsampleNormAct(nn.Module):
    """Layer for an Upsample, optional convolution, normalization, and activation."""

    def __init__(
        self,
        scale_factor: float | tuple[float, float] | None = None,
        size: tuple[int, int] | None = None,
        mode: str = "nearest",
        align_corners: bool | None = None,
        in_channels: int | None = None,
        out_channels: int | None = None,
        kernel_size: int | None = None,
        stride: int = 1,
        padding: int = 0,
        groups: int | None = 1,
        dilation: int | None = 1,
        bias: bool | None = False,
        norm: type[nn.Module] | None = nn.BatchNorm2d,
        norm_kwargs: dict | None = None,
        act: type[nn.Module] | None = nn.ReLU,
        act_kwargs: dict | None = None,
    ):
        """Layer for an Upsample, optional convolution, normalization, and activation.

        Args:
            scale_factor: Scale factor for upsampling. Can be a single float or tuple (H, W).
            size: Target output size (H, W). If provided, scale_factor is ignored.
            mode: Upsampling mode: 'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'.
            align_corners: If True, the corner pixels of the input and output tensors are aligned.
            in_channels: Number of input channels (for optional convolution after upsampling).
            out_channels: Number of output channels (for optional convolution after upsampling).
            kernel_size: Size of the convolutional kernel (if using convolution).
            stride: Stride of the convolution.
            padding: Padding of the convolution.
            groups: Number of groups in the convolution.
            dilation: Dilation of the convolution.
            bias: Whether to use a bias in the convolution.
            norm: Normalization class (e.g., nn.BatchNorm2d, nn.GroupNorm, None).
            norm_kwargs: Keyword arguments for initializing the normalization layer.
            act: Activation class (e.g., nn.ReLU, nn.SiLU, None).
            act_kwargs: Keyword arguments for initializing the activation layer.
        """
        super().__init__()

        if scale_factor is None and size is None:
            raise ValueError("Either scale_factor or size must be provided for upsampling")

        self.upsample = nn.Upsample(
            scale_factor=scale_factor,
            size=tuple(size) if size is not None else None,
            mode=mode,
            align_corners=align_corners,
        )

        # Optional convolution after upsampling
        self.conv = None
        norm_channels = None

        if kernel_size is not None:
            # Create convolution if kernel_size is provided
            if in_channels is None or out_channels is None:
                raise ValueError(
                    "in_channels and out_channels must be provided when using convolution"
                )
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )

        if out_channels is not None:
            norm_channels = out_channels
        elif in_channels is not None:
            norm_channels = in_channels

        self.norm = utils.create_normalization(norm, norm_channels, norm_kwargs)
        self.act = utils.create_activation(act, act_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the UpsampleNormAct.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        # The NHWC bilinear kernel has an INT_MAX element count limit; convert to
        # contiguous (NCHW) format to use the standard kernel when needed.
        if self.upsample.mode in ("bilinear", "bicubic") and x.is_contiguous(
            memory_format=torch.channels_last
        ):
            x = x.to(memory_format=torch.contiguous_format)
        x = self.upsample(x)
        if self.conv is not None:
            x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class UpsampleBlock(nn.Module):
    """Block of UpsampleNormAct layers."""

    def __init__(
        self,
        scale_factors: list[float | tuple[float, float]] | None = None,
        sizes: list[tuple[int, int]] | None = None,
        modes: list[str] | str = "nearest",
        align_corners: list[bool | None] | bool | None = None,
        in_channels: int | None = None,
        channels: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
        strides: list[int] | None = None,
        paddings: list[int] | None = None,
        groups: list[int] | None = None,
        dilations: list[int] | None = None,
        bias: bool | None = False,
        norm: type[nn.Module] | None = nn.BatchNorm2d,
        norm_kwargs: dict | None = None,
        act: type[nn.Module] | None = nn.ReLU,
        act_kwargs: dict | None = None,
        checkpoint_layers: list[int] | Literal["all"] | None = None,
        checkpoint_use_reentrant: bool = False,
    ):
        """Block of UpsampleNormAct layers.

        Args:
            scale_factors: List of scale factors for upsampling. Can be floats or tuples (H, W).
            sizes: List of target output sizes (H, W). If provided, scale_factors is ignored.
            modes: Upsampling mode(s): 'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'.
                Can be a single string or list of strings.
            align_corners: If True, the corner pixels are aligned. Can be a single bool or list.
            in_channels: Number of input channels (for optional convolution after upsampling).
            channels: List of output channels (for optional convolution after upsampling).
            kernel_sizes: List of kernel sizes (if using convolution).
            strides: List of strides.
            paddings: List of paddings.
            groups: List of groups.
            dilations: List of dilations.
            bias: Whether to use a bias in the convolution.
            norm: Normalization class (e.g., nn.BatchNorm2d, nn.GroupNorm, None).
            norm_kwargs: Keyword arguments for initializing the normalization layer.
            act: Activation class (e.g., nn.ReLU, nn.SiLU, None).
            act_kwargs: Keyword arguments for initializing the activation layer.
            checkpoint_layers: List of layer indices to checkpoint, "all" for all layers,
                or None to disable checkpointing.
            checkpoint_use_reentrant: Whether to use reentrant checkpointing.
        """
        super().__init__()

        if scale_factors is None and sizes is None:
            raise ValueError("Either scale_factors or sizes must be provided for upsampling")

        if scale_factors is not None and sizes is not None:
            raise ValueError("Cannot provide both scale_factors and sizes")

        # Determine number of layers
        if scale_factors is not None:
            self.num_layers = len(scale_factors)
        else:
            assert sizes is not None  # guarded by the ValueError checks above
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

        # Normalize modes and align_corners to lists
        if isinstance(self.modes, str):
            self.modes = [self.modes] * self.num_layers
        if isinstance(self.align_corners, bool):
            self.align_corners = [self.align_corners] * self.num_layers
        elif self.align_corners is None:
            self.align_corners = [None] * self.num_layers

        # Setup checkpointing
        self._setup_checkpointing(checkpoint_layers, checkpoint_use_reentrant)

        self.assert_same_lengths()
        self.upsample_blocks = nn.ModuleList()

        for idx in range(self.num_layers):
            scale_factor = self.scale_factors[idx] if self.scale_factors is not None else None
            size = self.sizes[idx] if self.sizes is not None else None
            mode = self.modes[idx]
            align_corner = self.align_corners[idx]

            # Determine channels for this layer
            layer_in_channels = None
            layer_out_channels = None
            layer_kernel_size = None

            if self.channels is not None:
                if idx == 0:
                    layer_in_channels = self.in_channels
                else:
                    layer_in_channels = self.channels[idx - 1]
                layer_out_channels = self.channels[idx]

                if self.kernel_sizes is not None:
                    layer_kernel_size = self.kernel_sizes[idx]

            layer_stride = self.strides[idx] if self.strides else 1
            layer_padding = self.paddings[idx] if self.paddings else 0
            layer_group = self.groups[idx] if self.groups else 1
            layer_dilation = self.dilations[idx] if self.dilations else 1

            self.upsample_blocks.append(
                UpsampleNormAct(
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
        """Setup gradient checkpointing configuration.

        Args:
            checkpoint_layers: Layer indices to checkpoint, "all", or None.
            checkpoint_use_reentrant: Whether to use reentrant checkpointing.
        """
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
        """Assert that all lists have the same length."""
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
        """Forward pass of the UpsampleBlock."""
        for idx, upsample_block in enumerate(self.upsample_blocks):
            x = self._run_module(upsample_block, x, idx)
        return x


if __name__ == "__main__":
    # Test simple upsampling without convolution
    model = UpsampleBlock(
        scale_factors=[2.0, 2.0],
        modes=["bilinear", "nearest"],
    )
    input_data = torch.randn(1, 3, 32, 32)
    output_data = model(input_data)
    print(f"Simple upsampling: {input_data.shape} -> {output_data.shape}")

    # Test upsampling with convolution
    model = UpsampleBlock(
        scale_factors=[2.0, 2.0],
        modes="bilinear",
        in_channels=3,
        channels=[64, 128],
        kernel_sizes=[3, 3],
        paddings=[1, 1],
    )
    input_data = torch.randn(1, 3, 32, 32)
    output_data = model(input_data)
    print(f"Upsampling with conv: {input_data.shape} -> {output_data.shape}")
