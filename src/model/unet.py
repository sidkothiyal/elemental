"""U-Net Architecture."""
# Reference: https://paperswithcode.com/method/u-net
from typing import List

import torch
from torch import nn

from model import blocks


class UNet(nn.Module):
    """
    U-Net Architecture
    5 Convolutional Blocks (down / encoder)
    4 Convolutional Blocks (up / decoder)
    skip connections

    Args:
        in_channels: int
        out_channels: int

    Returns:
        final_x: torch.Tensor

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: List[int] | None = None,
        skip_first_connection: bool = False,
    ):
        """Initialize the U-Net model."""
        super().__init__()
        if channels is None or len(channels) == 0:
            channels = [64, 128, 256, 512, 1024]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.skip_first_connection = skip_first_connection

        self.encoder = nn.ModuleList()
        self.skip_connections = nn.ModuleDict()
        self.decoder = nn.ModuleList()

        for channel_idx, channel in enumerate(channels[:-1]):
            block_input_channels = in_channels
            if channel_idx > 0:
                block_input_channels = channels[channel_idx - 1]
            self.encoder.append(self._create_down_block(block_input_channels, channel))
            if skip_first_connection and channel_idx == 0:
                continue
            self.skip_connections[str(channel_idx)] = nn.Conv2d(
                channel, channel, kernel_size=1, stride=1, padding=0
            )

        self.bottleneck = blocks.ConvBlock(
            in_channels=channels[-2],
            channels=[channels[-1], channels[-1]],
            kernel_sizes=[3, 3],
            strides=[1, 1],
            paddings=[1, 1],
        )

        reversed_channels = channels[::-1]
        for channel_idx, channel in enumerate(reversed_channels[:-1]):
            self.decoder.append(self._create_up_block(channel, reversed_channels[channel_idx + 1]))

        self.output_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1, stride=1, padding=0)

    def _create_down_block(self, in_channels, out_channels):
        # TODO: Downsample and 2x channels on the same layer instead of two separate layers.
        return nn.Sequential(
            blocks.ConvBlock(
                in_channels=in_channels,
                channels=[out_channels, out_channels],
                kernel_sizes=[3, 3],
                strides=[1, 1],
                paddings=[1, 1],
            ),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
        )

    def _create_up_block(self, in_channels, out_channels):
        return nn.Sequential(
            blocks.ConvTransposeBNReLU(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
                padding=0,
            ),
            blocks.ConvBlock(
                in_channels=in_channels,
                channels=[out_channels, out_channels],
                kernel_sizes=[3, 3],
                strides=[1, 1],
                paddings=[1, 1],
            ),
        )

    def forward(self, x):
        """Forward pass of the U-Net model."""
        skip_x = {}
        for idx, down in enumerate(self.encoder):
            x = down[0](x)
            if str(idx) in self.skip_connections:
                skip_x[idx] = self.skip_connections[str(idx)](x)
            x = down[1](x)

        x = self.bottleneck(x)
        for idx, up in enumerate(self.decoder):
            x = up[0](x)
            skip_idx = len(self.decoder) - idx - 1
            skip = x
            if str(skip_idx) in self.skip_connections:
                skip = skip_x[skip_idx]
            x = torch.cat((x, skip), dim=1)
            x = up[1](x)

        return self.output_conv(x)


if __name__ == "__main__":
    # No skip first connection
    model = UNet(in_channels=3, out_channels=5, skip_first_connection=False)
    model.to("cuda")
    input_x = torch.randn((2, 3, 256, 256)).to("cuda")
    output = model(input_x)
    print(output.shape)

    # Skip first connection
    model = UNet(in_channels=3, out_channels=5, skip_first_connection=True)
    model.to("cuda")
    input_x = torch.randn((2, 3, 256, 256)).to("cuda")
    output = model(input_x)
    print(output.shape)
