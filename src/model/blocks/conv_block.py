"""Convolutional blocks for the model."""
from typing import List
from typing import Optional

from torch import nn


class ConvBNReLU(nn.Module):
    """Layer with a convolution, batch normalization, and ReLU activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        groups: Optional[int] = 1,
        dilation: Optional[int] = 1,
        bias: Optional[bool] = False,
        use_bn: Optional[bool] = True,
        use_relu: Optional[bool] = True,
    ):
        """Layer with a convolution, batch normalization, and ReLU activation.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolutional kernel.
            stride: Stride of the convolution.
            padding: Padding of the convolution.
            groups: Number of groups in the convolution.
            dilation: Dilation of the convolution.
            bias: Whether to use a bias in the convolution.
            use_bn: Whether to use batch normalization.
            use_relu: Whether to use a ReLU activation.
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.use_bn = use_bn
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        self.use_relu = use_relu
        if self.use_relu:
            self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass."""
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.use_relu:
            x = self.relu(x)
        return x


class ConvBlock(nn.Module):
    """Block of ConvBNReLU layers."""

    def __init__(
        self,
        in_channels: int,
        channels: List[int],
        kernel_sizes: List[int],
        strides: List[int],
        paddings: List[int],
        groups: Optional[List[int]] = None,
        dilations: Optional[List[int]] = None,
        use_bn: Optional[bool] = True,
        use_relu: Optional[bool] = True,
    ):
        """Block of ConvBNReLU layers.

        Args:
            in_channels: Number of input channels.
            channels: List of output channels.
            kernel_sizes: List of kernel sizes.
            strides: List of strides.
            paddings: List of paddings.
            groups: List of groups.
            dilations: List of dilations.
            bias: Whether to use a bias in the convolution.
            use_bn: Whether to use batch normalization.
            use_relu: Whether to use a ReLU activation.
        """
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.paddings = paddings
        self.groups = groups
        self.dilations = dilations
        self.use_bn = use_bn
        self.use_relu = use_relu

        self.assert_same_lengths()
        self.conv_blocks = nn.ModuleList()
        for idx, (
            out_channels,
            kernel_size,
            stride,
            padding,
        ) in enumerate(
            zip(
                self.channels,
                self.kernel_sizes,
                self.strides,
                self.paddings,
            )
        ):
            in_channels = (
                self.in_channels if idx == 0 else self.channels[idx - 1]
            )
            group = groups[idx] if groups else 1
            dilation = dilations[idx] if dilations else 1
            self.conv_blocks.append(
                ConvBNReLU(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    group,
                    dilation,
                    use_bn,
                    use_relu,
                )
            )

    def assert_same_lengths(self):
        """Assert that all lists have the same length."""
        num_layers = len(self.channels)
        if (
            len(self.kernel_sizes) != num_layers
            or len(self.strides) != num_layers
            or len(self.paddings) != num_layers
            or (self.groups and len(self.groups) != num_layers)
            or (self.dilations and len(self.dilations) != num_layers)
        ):
            error_msg = (
                "All lists must have the same length. Got: "
                f"Kernel sizes: {len(self.kernel_sizes)}, "
                f"Strides: {len(self.strides)}, "
                f"Paddings: {len(self.paddings)}, "
                f"Groups: {len(self.groups) if self.groups else None}, "
                f"Dilations: {len(self.dilations) if self.dilations else None}"
            )
            raise ValueError(error_msg)

    def forward(self, x):
        """Forward pass."""
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        return x
