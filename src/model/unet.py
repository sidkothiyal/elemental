"""U-Net Architecture."""
# Reference: https://paperswithcode.com/method/u-net
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

    def __init__(self, in_channels, out_channels):
        """Initialize the U-Net model."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder = nn.ModuleList(
            [
                self._create_down_block(in_channels, 64),
                self._create_down_block(64, 128),
                self._create_down_block(128, 256),
                self._create_down_block(256, 512),
            ]
        )

        self.skips = nn.ModuleList(
            [
                nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            ]
        )

        self.bottleneck = blocks.ConvBlock(
            in_channels=512,
            channels=[1024, 1024],
            kernel_sizes=[3, 3],
            strides=[1, 1],
            paddings=[1, 1],
        )

        self.decoder = nn.ModuleList(
            [
                self._create_up_block(1024, 512),
                self._create_up_block(512, 256),
                self._create_up_block(256, 128),
                self._create_up_block(128, 64),
            ]
        )

        self.output_conv = nn.Conv2d(64, out_channels, kernel_size=1, stride=1, padding=0)

    def _create_down_block(self, in_channels, out_channels):
        return nn.Sequential(
            blocks.ConvBlock(
                in_channels=in_channels,
                channels=[out_channels, out_channels],
                kernel_sizes=[3, 3],
                strides=[1, 1],
                paddings=[1, 1],
            ),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
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
        skips = []
        for down in self.encoder:
            x = down[0](x)
            skips.append(down[1](x))
            x = down[2](x)

        x = self.bottleneck(x)

        for itr, (up, skip) in enumerate(zip(self.decoder, reversed(skips))):
            x = up[0](x)
            skip = self.skips[itr](skip)
            x = torch.cat((x, skip), dim=1)
            x = up[1](x)

        return self.output_conv(x)


if __name__ == "__main__":
    model = UNet(in_channels=3, out_channels=5)
    input_x = torch.randn((2, 3, 512, 512))
    output = model(input_x)
    print(output.shape)
