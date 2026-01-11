"""Convolutional transpose blocks for the model."""

from typing import Callable, Literal

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class ConvTransposeBNReLU(nn.Module):
    """Layer for a ConvTranspose2d, batch norm, and ReLU activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        output_padding: int | None = 0,
        groups: int | None = 1,
        dilation: int | None = 1,
        bias: bool | None = False,
        use_bn: bool | None = True,
        use_relu: bool | None = True,
    ):
        """Layer for a ConvTranspose2d, batch norm, and ReLU activation.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the transposed convolutional kernel.
            stride: Stride of the transposed convolution.
            padding: Padding of the transposed convolution.
            output_padding: Output padding of the transposed convolution.
            groups: Number of groups in the transposed convolution.
            dilation: Dilation of the transposed convolution.
            bias: Whether to use a bias in the transposed convolution.
            use_bn: Whether to use batch normalization.
            use_relu: Whether to use a ReLU activation.
        """
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            dilation=dilation,
            bias=bias,
        )
        self.use_bn = use_bn
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        self.use_relu = use_relu
        if self.use_relu:
            self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ConvTransposeBNReLU.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        x = self.conv_transpose(x)
        if self.use_bn:
            x = self.bn(x)
        if self.use_relu:
            x = self.relu(x)
        return x


class ConvTransposeBlock(nn.Module):
    """Block of ConvTransposeBNReLU layers."""

    def __init__(
        self,
        in_channels: int,
        channels: list[int],
        kernel_sizes: list[int],
        strides: list[int],
        paddings: list[int],
        output_paddings: list[int],
        groups: list[int] | None = None,
        dilations: list[int] | None = None,
        bias: bool | None = False,
        use_bn: bool | None = True,
        use_relu: bool | None = True,
        checkpoint_layers: list[int] | Literal["all"] | None = None,
        checkpoint_use_reentrant: bool = False,
    ):
        """Block of ConvTransposeBNReLU layers.

        Args:
            in_channels: Number of input channels.
            channels: list of output channels.
            kernel_sizes: list of kernel sizes.
            strides: list of strides.
            paddings: list of paddings.
            output_paddings: list of output paddings.
            groups: list of groups.
            dilations: list of dilations.
            bias: Whether to use a bias in the transposed convolution.
            use_bn: Whether to use batch normalization.
            use_relu: Whether to use a ReLU activation.
            checkpoint_layers: List of layer indices to checkpoint, "all" for all layers,
                or None to disable checkpointing.
            checkpoint_use_reentrant: Whether to use reentrant checkpointing.
        """
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.paddings = paddings
        self.output_paddings = output_paddings
        self.groups = groups
        self.dilations = dilations
        self.use_bn = use_bn
        self.use_relu = use_relu
        self.num_layers = len(channels)

        # Setup checkpointing
        self._setup_checkpointing(checkpoint_layers, checkpoint_use_reentrant)

        self.assert_same_lengths()
        self.conv_transpose_blocks = nn.ModuleList()
        for idx, (
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
        ) in enumerate(
            zip(
                self.channels,
                self.kernel_sizes,
                self.strides,
                self.paddings,
                self.output_paddings,
            )
        ):
            in_channels = self.in_channels if idx == 0 else self.channels[idx - 1]
            group = groups[idx] if groups else 1
            dilation = dilations[idx] if dilations else 1
            self.conv_transpose_blocks.append(
                ConvTransposeBNReLU(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    output_padding,
                    group,
                    dilation,
                    bias,
                    use_bn,
                    use_relu,
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
        num_layers = len(self.channels)
        if (
            len(self.kernel_sizes) != num_layers
            or len(self.strides) != num_layers
            or len(self.paddings) != num_layers
            or len(self.output_paddings) != num_layers
            or (self.groups and len(self.groups) != num_layers)
            or (self.dilations and len(self.dilations) != num_layers)
        ):
            error_msg = (
                "All lists must have the same length. Got: "
                f"Kernel sizes: {len(self.kernel_sizes)}, "
                f"Strides: {len(self.strides)}, "
                f"Paddings: {len(self.paddings)}, "
                f"Output paddings: {len(self.output_paddings)}, "
                f"Groups: {len(self.groups) if self.groups else None}, "
                f"Dilations: {len(self.dilations) if self.dilations else None}"
            )
            raise ValueError(error_msg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ConvTransposeBlock."""
        for idx, conv_transpose_block in enumerate(self.conv_transpose_blocks):
            x = self._run_module(conv_transpose_block, x, idx)
        return x
