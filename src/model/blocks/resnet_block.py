"""ResNet blocks for the model."""

import torch
from torch import nn

from model.blocks.conv_block import ConvBNReLU


class BasicResNetBlock(nn.Module):
    """Basic ResNet block for ResNet-18 and ResNet-34.

    This block consists of two 3x3 convolutions with batch normalization
    and ReLU activation. It includes a skip connection that adds the input
    to the output.
    """

    expansion = 1  # Output channels multiplier

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        dilation: int = 1,
    ):
        """Basic ResNet block initialization.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            stride: Stride for the first convolution (used for downsampling).
            downsample: Optional downsampling module for the skip connection.
            groups: Number of groups for grouped convolution.
            dilation: Dilation rate for convolutions.
        """
        super().__init__()

        # First 3x3 convolution with stride (for potential downsampling)
        self.conv1 = ConvBNReLU(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            groups=groups,
            dilation=dilation,
            bias=False,
            use_bn=True,
            use_relu=True,
        )

        # Second 3x3 convolution with stride 1
        self.conv2 = ConvBNReLU(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            groups=groups,
            dilation=dilation,
            bias=False,
            use_bn=True,
            use_relu=False,  # No ReLU here, applied after addition
        )

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the BasicResNetBlock.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, C', H', W').
        """
        identity = x

        # Apply convolutions
        out = self.conv1(x)
        out = self.conv2(out)

        # Apply downsampling to identity if needed
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add skip connection
        out += identity
        out = self.relu(out)

        return out


class BottleneckResNetBlock(nn.Module):
    """Bottleneck ResNet block for ResNet-50, ResNet-101, and ResNet-152.

    This block consists of three convolutions:
    1. 1x1 convolution to reduce dimensions
    2. 3x3 convolution to process features
    3. 1x1 convolution to expand dimensions

    It includes a skip connection that adds the input to the output.
    """

    expansion = 4  # Output channels multiplier

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
    ):
        """Bottleneck ResNet block initialization.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of intermediate channels (final will be out_channels * expansion).
            stride: Stride for the 3x3 convolution (used for downsampling).
            downsample: Optional downsampling module for the skip connection.
            groups: Number of groups for grouped convolution (for ResNeXt).
            base_width: Base width for computing the bottleneck width (for ResNeXt).
            dilation: Dilation rate for the 3x3 convolution.
        """
        super().__init__()

        # Calculate width for grouped convolutions (ResNeXt)
        width = int(out_channels * (base_width / 64.0)) * groups

        # 1x1 convolution to reduce dimensions
        self.conv1 = ConvBNReLU(
            in_channels=in_channels,
            out_channels=width,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            dilation=1,
            bias=False,
            use_bn=True,
            use_relu=True,
        )

        # 3x3 convolution with stride (for potential downsampling)
        self.conv2 = ConvBNReLU(
            in_channels=width,
            out_channels=width,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            groups=groups,
            dilation=dilation,
            bias=False,
            use_bn=True,
            use_relu=True,
        )

        # 1x1 convolution to expand dimensions
        self.conv3 = ConvBNReLU(
            in_channels=width,
            out_channels=out_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            dilation=1,
            bias=False,
            use_bn=True,
            use_relu=False,  # No ReLU here, applied after addition
        )

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the BottleneckResNetBlock.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, C', H', W').
        """
        identity = x

        # Apply convolutions
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        # Apply downsampling to identity if needed
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add skip connection
        out += identity
        out = self.relu(out)

        return out


class ResNetStage(nn.Module):
    """A stage of ResNet consisting of multiple residual blocks.

    This module creates a sequence of residual blocks with optional
    downsampling at the first block.
    """

    def __init__(
        self,
        block_type: type[BasicResNetBlock | BottleneckResNetBlock],
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
    ):
        """ResNet stage initialization.

        Args:
            block_type: Type of residual block (BasicResNetBlock or BottleneckResNetBlock).
            in_channels: Number of input channels.
            out_channels: Number of output channels for each block.
            num_blocks: Number of residual blocks in this stage.
            stride: Stride for the first block (used for downsampling).
            groups: Number of groups for grouped convolution.
            base_width: Base width for computing the bottleneck width.
            dilation: Dilation rate for convolutions.
        """
        super().__init__()

        self.blocks = nn.ModuleList()

        # Create downsampling module if needed
        downsample = None
        if stride != 1 or in_channels != out_channels * block_type.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * block_type.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * block_type.expansion),
            )

        # First block with potential downsampling
        if block_type == BottleneckResNetBlock:
            self.blocks.append(
                block_type(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    downsample=downsample,
                    groups=groups,
                    base_width=base_width,
                    dilation=dilation,
                )
            )
        else:  # BasicResNetBlock or PreActBasicBlock
            self.blocks.append(
                block_type(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    downsample=downsample,
                    groups=groups,
                    dilation=dilation,
                )
            )

        # Remaining blocks
        for _ in range(1, num_blocks):
            if block_type == BottleneckResNetBlock:
                self.blocks.append(
                    block_type(
                        in_channels=out_channels * block_type.expansion,
                        out_channels=out_channels,
                        stride=1,
                        downsample=None,
                        groups=groups,
                        base_width=base_width,
                        dilation=dilation,
                    )
                )
            else:  # BasicResNetBlock or PreActBasicBlock
                self.blocks.append(
                    block_type(
                        in_channels=out_channels * block_type.expansion,
                        out_channels=out_channels,
                        stride=1,
                        downsample=None,
                        groups=groups,
                        dilation=dilation,
                    )
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ResNet stage.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, C', H', W').
        """
        for block in self.blocks:
            x = block(x)
        return x


class PreActBasicBlock(nn.Module):
    """Pre-activation Basic ResNet block.

    This variant applies batch normalization and ReLU before convolutions,
    which can improve gradient flow. Used in some ResNet variations.
    """

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        dilation: int = 1,
    ):
        """Pre-activation Basic ResNet block initialization.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            stride: Stride for the first convolution.
            downsample: Optional downsampling module for the skip connection.
            groups: Number of groups for grouped convolution.
            dilation: Dilation rate for convolutions.
        """
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            groups=groups,
            dilation=dilation,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            groups=groups,
            dilation=dilation,
            bias=False,
        )

        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the PreActBasicBlock.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, C', H', W').
        """
        identity = x

        # Pre-activation
        out = self.bn1(x)
        out = self.relu1(out)

        # Apply downsampling to identity if needed
        if self.downsample is not None:
            identity = self.downsample(out)

        # First convolution
        out = self.conv1(out)

        # Second pre-activation and convolution
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        # Add skip connection
        out += identity

        return out


if __name__ == "__main__":
    # Test BasicResNetBlock
    print("Testing BasicResNetBlock...")
    basic_block = BasicResNetBlock(in_channels=64, out_channels=64)
    x = torch.randn(2, 64, 56, 56)
    out = basic_block(x)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")

    # Test BasicResNetBlock with downsampling
    print("\nTesting BasicResNetBlock with downsampling...")
    downsample = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
        nn.BatchNorm2d(128),
    )
    basic_block_downsample = BasicResNetBlock(
        in_channels=64, out_channels=128, stride=2, downsample=downsample
    )
    out = basic_block_downsample(x)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")

    # Test BottleneckResNetBlock
    print("\nTesting BottleneckResNetBlock...")
    bottleneck_block = BottleneckResNetBlock(in_channels=256, out_channels=64)
    x = torch.randn(2, 256, 56, 56)
    out = bottleneck_block(x)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")

    # Test BottleneckResNetBlock with downsampling
    print("\nTesting BottleneckResNetBlock with downsampling...")
    downsample = nn.Sequential(
        nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
        nn.BatchNorm2d(512),
    )
    bottleneck_block_downsample = BottleneckResNetBlock(
        in_channels=256, out_channels=128, stride=2, downsample=downsample
    )
    out = bottleneck_block_downsample(x)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")

    # Test ResNetStage
    print("\nTesting ResNetStage...")
    stage = ResNetStage(
        block_type=BasicResNetBlock,
        in_channels=64,
        out_channels=128,
        num_blocks=3,
        stride=2,
    )
    x = torch.randn(2, 64, 56, 56)
    out = stage(x)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")

    # Test PreActBasicBlock
    print("\nTesting PreActBasicBlock...")
    preact_block = PreActBasicBlock(in_channels=64, out_channels=64)
    x = torch.randn(2, 64, 56, 56)
    out = preact_block(x)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")

    print("\nAll tests passed!")
