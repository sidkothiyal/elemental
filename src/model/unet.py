"""U-Net Architecture."""
# Reference: https://paperswithcode.com/method/u-net
import logging
from typing import Callable, List, Literal

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from model import blocks

logger = logging.getLogger(__name__)


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
        # Conditional block arguments
        use_conditional_blocks: bool = False,
        time_embed_dim: int | None = None,
        cond_embed_dim: int | None = None,
        num_groups: int | list[int] = 32,
        dropout: float = 0.1,
        use_cross_attn: bool | list[bool] = True,
        cross_attn_num_heads: int | list[int] = 8,
        cond_level_mapping: dict[int, str] | None = None,
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
            use_conditional_blocks: Whether to use ConditionalConvBlock instead of regular blocks.
            time_embed_dim: Dimension of the time embedding. Required if use_conditional_blocks.
            cond_embed_dim: Dimension of the condition embedding. Required if
                use_conditional_blocks.
            num_groups: Number of groups for GroupNorm. Can be a single int or a list with
                length equal to len(channels)-1 for per-level customization.
            dropout: Dropout rate for conditional blocks.
            use_cross_attn: Whether to use cross-attention in conditional blocks. Can be a
                single bool or a list with length equal to len(channels)-1 for per-level
                customization.
            cross_attn_num_heads: Number of heads for cross-attention. Can be a single int or
                a list with length equal to len(channels)-1 for per-level customization.
            cond_level_mapping: Optional mapping from block index to condition key.
                If provided, aux_features should be a dict with keys matching the mapping values.
                Block indices: 0 to num_encoder_blocks-1 for encoder, num_encoder_blocks for
                bottleneck. Example: {0: "low_level", 1: "low_level", 2: "high_level"} means
                encoder blocks 0 and 1 use aux_features["low_level"], bottleneck uses
                aux_features["high_level"].
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
        self.use_conditional_blocks = use_conditional_blocks

        # Validate conditional block arguments
        if use_conditional_blocks:
            if time_embed_dim is None or cond_embed_dim is None:
                raise ValueError(
                    "time_embed_dim and cond_embed_dim are required when "
                    "use_conditional_blocks=True"
                )
        self.time_embed_dim = time_embed_dim
        self.cond_embed_dim = cond_embed_dim
        self.dropout = dropout
        self.cond_level_mapping = cond_level_mapping or {}

        # Setup per-level conditional block parameters
        self._setup_conditional_params(num_groups, use_cross_attn, cross_attn_num_heads)

        # Setup checkpointing configuration
        self._setup_checkpointing(checkpoint_blocks, checkpoint_modules, checkpoint_use_reentrant)

        self.encoder = nn.ModuleList()
        self.skip_connections = nn.ModuleDict()
        self.decoder = nn.ModuleList()

        for channel_idx, channel in enumerate(channels[:-1]):
            block_input_channels = in_channels
            if channel_idx > 0:
                block_input_channels = channels[channel_idx - 1]
            self.encoder.append(self._create_down_block(block_input_channels, channel, channel_idx))
            if skip_first_connection and channel_idx == 0:
                continue
            # Skip connection handling differs between standard and conditional modes:
            # - Standard: skip saved BEFORE downsample (higher res), use block_input_channels
            # - Conditional: skip saved AFTER block (lower res), use channel (output channels)
            skip_connection_input_channels = (
                channel if use_conditional_blocks else block_input_channels
            )
            self.skip_connections[str(channel_idx)] = nn.Conv2d(
                skip_connection_input_channels, channel, kernel_size=1, stride=1, padding=0
            )

        if use_conditional_blocks:
            # Use last encoder level's parameters for bottleneck
            bottleneck_num_groups = self.num_groups_list[-1]
            bottleneck_use_cross_attn = self.use_cross_attn_list[-1]
            bottleneck_cross_attn_num_heads = self.cross_attn_num_heads_list[-1]
            self.bottleneck = blocks.ConditionalConvBlock(
                in_channels=channels[-2],
                out_channels=channels[-1],
                time_embed_dim=time_embed_dim,
                cond_embed_dim=cond_embed_dim,
                num_groups=bottleneck_num_groups,
                dropout=dropout,
                use_cross_attn=bottleneck_use_cross_attn,
                cross_attn_num_heads=bottleneck_cross_attn_num_heads,
            )
        else:
            self.bottleneck = blocks.ConvBlock(
                in_channels=channels[-2],
                channels=[channels[-1], channels[-1]],
                kernel_sizes=[3, 3],
                strides=[1, 1],
                paddings=[1, 1],
            )

        reversed_channels = channels[::-1]
        for channel_idx, channel in enumerate(reversed_channels[:-1]):
            # Map decoder index to encoder level (reversed)
            encoder_level_idx = len(reversed_channels) - 2 - channel_idx
            out_ch = reversed_channels[channel_idx + 1]
            # For conditional blocks, account for skip concatenation in input channels
            skip_ch = (
                channels[encoder_level_idx]
                if str(encoder_level_idx) in self.skip_connections
                else 0
            )
            self.decoder.append(
                self._create_up_block(channel, out_ch, encoder_level_idx, skip_channels=skip_ch)
            )

        self.output_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1, stride=1, padding=0)

    def _setup_conditional_params(
        self,
        num_groups: int | list[int],
        use_cross_attn: bool | list[bool],
        cross_attn_num_heads: int | list[int],
    ) -> None:
        """Setup per-level conditional block parameters.

        Args:
            num_groups: Number of groups for GroupNorm.
            use_cross_attn: Whether to use cross-attention.
            cross_attn_num_heads: Number of heads for cross-attention.
        """
        if not self.use_conditional_blocks:
            return
        num_levels = self.num_encoder_blocks
        self.num_groups_list = self._create_block_params(num_groups, num_levels, "num_groups")
        if self.in_channels % self.num_groups_list[0] != 0:
            logger.warning(
                "In_channels %s is not divisible by num_groups %s, setting num_groups to 1",
                self.in_channels,
                self.num_groups_list[0],
            )
            self.num_groups_list[0] = 1
        self.use_cross_attn_list = self._create_block_params(
            use_cross_attn, num_levels, "use_cross_attn"
        )
        self.cross_attn_num_heads_list = self._create_block_params(
            cross_attn_num_heads, num_levels, "cross_attn_num_heads"
        )

    def _create_block_params(
        self, param: list[bool | int] | bool | int, num_blocks: int, param_name: str
    ) -> list[bool | int]:
        """Create a list of parameters for each block.

        Args:
            param: Parameter to create a list of for each block.
                Can be a single number, a list of the same length as the number of blocks,
                or a list of booleans.
            num_blocks: Number of blocks.
            param_name: Name of the parameter.

        Returns:
            List of parameters for each block.

        Raises:
            ValueError: If the parameter is not a single number or a list of the same length
                as the number of blocks.
        """
        if isinstance(param, (bool, int)):
            return [param] * num_blocks

        param = list(param)
        if len(param) != num_blocks:
            raise ValueError(
                f"{param_name} must be a single number, or a list of the same length "
                f"as the number of blocks ({num_blocks}). Got {len(param)} for {param_name}."
            )
        return param

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

    def _create_down_block(self, in_channels, out_channels, level_idx: int):
        """Create a downsampling block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            level_idx: Index of the encoder level for per-level parameters.
        """
        if self.use_conditional_blocks:
            return blocks.ConditionalConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                time_embed_dim=self.time_embed_dim,
                cond_embed_dim=self.cond_embed_dim,
                num_groups=self.num_groups_list[level_idx],
                dropout=self.dropout,
                use_cross_attn=self.use_cross_attn_list[level_idx],
                cross_attn_num_heads=self.cross_attn_num_heads_list[level_idx],
                downsample=True,
            )
        return nn.Sequential(
            blocks.ConvNormAct(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm=nn.BatchNorm2d,
                act=nn.ReLU,
            ),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def _create_up_block(
        self,
        in_channels,
        out_channels,
        level_idx: int,
        skip_channels: int = 0,
        mode: str = "bilinear",
    ):
        """Create an upsampling block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            level_idx: Index of the encoder level for per-level parameters (decoder uses
                symmetric/reversed parameters from encoder).
            skip_channels: Number of channels from skip connection (used for conditional blocks).
            mode: Mode of the upsampling.
                Can be 'nearest' or 'bilinear'.
                Bilinear is more expensive but better for image reconstruction.
                Nearest is faster but less accurate.
        """
        if self.use_conditional_blocks:
            # For conditional blocks, input includes concatenated skip connection
            block_in_channels = in_channels + skip_channels
            return blocks.ConditionalConvBlock(
                in_channels=block_in_channels,
                out_channels=out_channels,
                time_embed_dim=self.time_embed_dim,
                cond_embed_dim=self.cond_embed_dim,
                num_groups=self.num_groups_list[level_idx],
                dropout=self.dropout,
                use_cross_attn=self.use_cross_attn_list[level_idx],
                cross_attn_num_heads=self.cross_attn_num_heads_list[level_idx],
                upsample=True,
            )
        return nn.Sequential(
            blocks.UpsampleNormAct(
                scale_factor=2.0,
                mode=mode,
                align_corners=True,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                norm=nn.BatchNorm2d,
                act=nn.ReLU,
            ),
            blocks.ConvBlock(
                in_channels=in_channels,
                channels=[out_channels, out_channels],
                kernel_sizes=[3, 3],
                strides=[1, 1],
                paddings=[1, 1],
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        aux_features: torch.Tensor | dict[str, torch.Tensor] | None = None,
    ):
        """Forward pass of the U-Net model.

        Args:
            x: Input tensor of shape (batch, channels, height, width).
            t: Time embedding tensor of shape (batch, time_embed_dim).
                Required if use_conditional_blocks=True.
            aux_features: Condition embedding tensor(s). Can be:
                - torch.Tensor of shape (batch, seq_len, cond_embed_dim) for single condition
                - dict[str, torch.Tensor] with condition keys when cond_level_mapping is provided
                Required if use_conditional_blocks=True.
        """
        if self.use_conditional_blocks:
            if t is None or aux_features is None:
                raise ValueError("t and aux_features are required when use_conditional_blocks=True")
            return self._forward_conditional(x, t, aux_features)
        return self._forward_standard(x)

    def _forward_standard(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for standard (non-conditional) U-Net."""
        skip_x = {}
        for idx, down in enumerate(self.encoder):
            # Save skip BEFORE downsampling to match decoder's upsampled resolution
            if str(idx) in self.skip_connections:
                skip_x[idx] = self.skip_connections[str(idx)](x)
            x = self._run_module(down[0], x, idx, "encoder")
            x = down[1](x)  # Conv - not checkpointed (small operation)

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

    def _forward_conditional(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        aux_features: torch.Tensor | dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass for conditional U-Net with ConditionalConvBlocks.

        For conditional mode, skip connections are saved AFTER encoder blocks
        (at lower resolution) because ConditionalConvBlock does downsampling
        at the end. Decoder concatenates with skip before processing.

        Args:
            x: Input tensor.
            t: Time embedding tensor.
            aux_features: Condition embedding(s). Can be a single tensor or a dict
                mapping condition keys to tensors (when cond_level_mapping is used).
        """

        # Helper function to get the right condition for a block
        def get_cond_for_block(block_idx: int) -> torch.Tensor:
            if isinstance(aux_features, dict):
                if block_idx in self.cond_level_mapping:
                    cond_key = self.cond_level_mapping[block_idx]
                    if cond_key not in aux_features:
                        raise KeyError(
                            f"Condition key '{cond_key}' not found in aux_features. "
                            f"Available keys: {list(aux_features.keys())}"
                        )
                    return aux_features[cond_key]
                # If no mapping for this block, use the first available condition
                # or raise an error if dict is provided but mapping is incomplete
                if self.cond_level_mapping:
                    raise KeyError(
                        f"No condition mapping found for block {block_idx}. "
                        f"Available mappings: {self.cond_level_mapping}"
                    )
                # Fallback: use first value in dict
                return next(iter(aux_features.values()))
            # Single tensor case
            return aux_features

        skip_x = {}
        for idx, down in enumerate(self.encoder):
            cond = get_cond_for_block(idx)
            if self._should_checkpoint(idx, "encoder"):
                x = checkpoint(down, x, t, cond, use_reentrant=self.checkpoint_use_reentrant)
            else:
                x = down(x, t, cond)
            # Save skip AFTER the block (at lower resolution) for conditional mode
            if str(idx) in self.skip_connections:
                skip_x[idx] = self.skip_connections[str(idx)](x)

        # Bottleneck uses the last encoder block's condition (or highest level)
        bottleneck_idx = self.num_encoder_blocks
        cond = get_cond_for_block(bottleneck_idx)
        if self._should_checkpoint(0, "bottleneck"):
            x = checkpoint(
                self.bottleneck,
                x,
                t,
                cond,
                use_reentrant=self.checkpoint_use_reentrant,
            )
        else:
            x = self.bottleneck(x, t, cond)

        for idx, up in enumerate(self.decoder):
            skip_idx = len(self.decoder) - idx - 1
            # Concat with skip at same resolution, then up block upsamples at end
            if str(skip_idx) in self.skip_connections:
                x = torch.cat((x, skip_x[skip_idx]), dim=1)
            # Decoder uses the same condition as the corresponding encoder block
            cond = get_cond_for_block(skip_idx)
            if self._should_checkpoint(idx, "decoder"):
                x = checkpoint(up, x, t, cond, use_reentrant=self.checkpoint_use_reentrant)
            else:
                x = up(x, t, cond)

        return self.output_conv(x)


if __name__ == "__main__":
    # No skip first connection
    model = UNet(in_channels=3, out_channels=5, skip_first_connection=False)
    model.to("cuda")
    input_x = torch.randn((2, 3, 256, 256)).to("cuda")
    output = model(input_x)
    print(f"Standard UNet output shape: {output.shape}")

    # Skip first connection
    model = UNet(in_channels=3, out_channels=5, skip_first_connection=True)
    model.to("cuda")
    input_x = torch.randn((2, 3, 256, 256)).to("cuda")
    output = model(input_x)
    print(f"Standard UNet (skip first) output shape: {output.shape}")

    # Conditional UNet with single values for params
    model = UNet(
        in_channels=3,
        out_channels=5,
        use_conditional_blocks=True,
        time_embed_dim=128,
        cond_embed_dim=128,
        num_groups=32,
        use_cross_attn=True,
        cross_attn_num_heads=8,
    )
    model.to("cuda")
    input_x = torch.randn((2, 3, 256, 256)).to("cuda")
    input_t = torch.randn((2, 128)).to("cuda")
    input_aux_features = torch.randn((2, 256, 128)).to("cuda")
    output = model(input_x, input_t, input_aux_features)
    print(f"Conditional UNet output shape: {output.shape}")

    # Conditional UNet with per-level customization
    model = UNet(
        in_channels=3,
        out_channels=5,
        channels=[64, 128, 256, 512, 1024],
        use_conditional_blocks=True,
        time_embed_dim=128,
        cond_embed_dim=128,
        num_groups=[1, 32, 32, 32],  # Different num_groups per level
        use_cross_attn=[False, True, True, True],  # No cross-attn in first level
        cross_attn_num_heads=[4, 8, 8, 8],  # Different heads per level
    )
    model.to("cuda")
    input_x = torch.randn((2, 3, 256, 256)).to("cuda")
    input_t = torch.randn((2, 128)).to("cuda")
    input_aux_features = torch.randn((2, 256, 128)).to("cuda")
    output = model(input_x, input_t, input_aux_features)
    print(f"Conditional UNet (per-level config) output shape: {output.shape}")
