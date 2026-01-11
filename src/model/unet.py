"""U-Net Architecture."""
# Reference: https://paperswithcode.com/method/u-net
from typing import Callable, List, Literal

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

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

    # Valid module types for gradient checkpointing
    CHECKPOINT_MODULE_TYPES = ("encoder", "bottleneck", "decoder")

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: List[int] | None = None,
        skip_first_connection: bool = False,
        checkpoint_blocks: list[int] | Literal["all"] | None = None,
        checkpoint_modules: list[str] | Literal["all"] | None = None,
        checkpoint_use_reentrant: bool = False,
    ):
        """Initialize the U-Net model.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            channels: List of channel sizes for each level. Defaults to [64, 128, 256, 512, 1024].
            skip_first_connection: Whether to skip the first skip connection.
            checkpoint_blocks: List of block indices to checkpoint, "all" for all blocks,
                or None to disable. Block indices correspond to encoder/decoder block positions.
            checkpoint_modules: List of module types to checkpoint. Valid types are
                "encoder", "bottleneck", "decoder". Use "all" or None to checkpoint all modules
                when checkpoint_blocks is set.
            checkpoint_use_reentrant: Whether to use reentrant checkpointing.
        """
        super().__init__()
        if channels is None or len(channels) == 0:
            channels = [64, 128, 256, 512, 1024]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.skip_first_connection = skip_first_connection
        self.num_encoder_blocks = len(channels) - 1
        self.num_decoder_blocks = len(channels) - 1

        # Setup checkpointing configuration
        self._setup_checkpointing(checkpoint_blocks, checkpoint_modules, checkpoint_use_reentrant)

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

    def _setup_checkpointing(
        self,
        checkpoint_blocks: list[int] | Literal["all"] | None,
        checkpoint_modules: list[str] | Literal["all"] | None,
        checkpoint_use_reentrant: bool,
    ) -> None:
        """Setup gradient checkpointing configuration.

        Args:
            checkpoint_blocks: Block indices to checkpoint, "all", or None.
            checkpoint_modules: Module types to checkpoint, "all", or None.
            checkpoint_use_reentrant: Whether to use reentrant checkpointing.
        """
        self.checkpoint_use_reentrant = checkpoint_use_reentrant

        # Determine which blocks to checkpoint
        max_blocks = max(self.num_encoder_blocks, self.num_decoder_blocks)
        if checkpoint_blocks is None:
            self.checkpoint_blocks: set[int] = set()
        elif checkpoint_blocks == "all":
            self.checkpoint_blocks = set(range(max_blocks))
        else:
            # Validate block indices
            for block_idx in checkpoint_blocks:
                if not 0 <= block_idx < max_blocks:
                    raise ValueError(
                        f"Invalid block index {block_idx}. Must be in range [0, {max_blocks})"
                    )
            self.checkpoint_blocks = set(checkpoint_blocks)

        # Determine which modules to checkpoint
        if checkpoint_modules is None or checkpoint_modules == "all":
            self.checkpoint_modules: set[str] = set(self.CHECKPOINT_MODULE_TYPES)
        else:
            for module_type in checkpoint_modules:
                if module_type not in self.CHECKPOINT_MODULE_TYPES:
                    raise ValueError(
                        f"Invalid module type '{module_type}'. "
                        f"Must be one of {self.CHECKPOINT_MODULE_TYPES}"
                    )
            self.checkpoint_modules = set(checkpoint_modules)

    def _should_checkpoint(self, block_idx: int, module_type: str) -> bool:
        """Check if a specific block should use gradient checkpointing.

        Args:
            block_idx: The index of the block.
            module_type: The type of module ("encoder", "bottleneck", "decoder").

        Returns:
            True if the block should be checkpointed, False otherwise.
        """
        if not self.training:
            return False
        if module_type == "bottleneck":
            # Bottleneck uses block index 0 for checkpointing decision
            return 0 in self.checkpoint_blocks and "bottleneck" in self.checkpoint_modules
        return block_idx in self.checkpoint_blocks and module_type in self.checkpoint_modules

    def _run_module(
        self,
        fn: nn.Module | Callable[[torch.Tensor], torch.Tensor],
        x: torch.Tensor,
        block_idx: int,
        module_type: str,
    ) -> torch.Tensor:
        """Run a module with optional gradient checkpointing.

        Args:
            fn: The module or callable to run.
            x: Input tensor.
            block_idx: The index of the block.
            module_type: The type of module ("encoder", "bottleneck", "decoder").

        Returns:
            Output tensor from the module.
        """
        if self._should_checkpoint(block_idx, module_type):
            return checkpoint(fn, x, use_reentrant=self.checkpoint_use_reentrant)
        return fn(x)

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
            x = self._run_module(down[0], x, idx, "encoder")
            if str(idx) in self.skip_connections:
                skip_x[idx] = self.skip_connections[str(idx)](x)
            x = down[1](x)  # Downsampling conv - not checkpointed (small operation)

        x = self._run_module(self.bottleneck, x, 0, "bottleneck")

        for idx, up in enumerate(self.decoder):
            x = up[0](x)  # Upsample - not checkpointed (small operation)
            skip_idx = len(self.decoder) - idx - 1
            skip = x
            if str(skip_idx) in self.skip_connections:
                skip = skip_x[skip_idx]
            x = torch.cat((x, skip), dim=1)
            x = self._run_module(up[1], x, idx, "decoder")

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
