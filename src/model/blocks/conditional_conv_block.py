"""Conditional Convolutional Block used for the conditional diffusion model."""

import torch
from torch import nn

from model import blocks


class ConditionalConvBlock(nn.Module):
    """Conditional Convolutional Block used for the conditional diffusion model."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        cond_embed_dim: int,
        num_groups: int = 32,
        dropout: float = 0.1,
        use_cross_attn: bool = True,
        cross_attn_num_heads: int = 8,
        downsample: bool = False,
        upsample: bool = False,
    ):
        """Conditional Convolutional Block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            time_embed_dim: Dimension of the time embedding.
            cond_embed_dim: Dimension of the condition embedding.
            num_groups: Number of groups in the convolution.
            dropout: Dropout rate.
            use_cross_attn: Whether to use cross-attention.
            cross_attn_num_heads: Number of heads in the cross-attention.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_embed_dim = time_embed_dim
        self.cond_embed_dim = cond_embed_dim
        self.num_groups = num_groups
        self.dropout = dropout
        self.use_cross_attn = use_cross_attn
        self.cross_attn_num_heads = cross_attn_num_heads

        # First convolutional block
        self.conv_block = nn.Sequential(
            nn.GroupNorm(num_groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

        # Time embedding projection -> scale and shift
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels * 2),
        )

        # Normalization
        self.norm = nn.GroupNorm(num_groups, out_channels)

        # Second convolutional block
        self.conv_block2 = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )
        else:
            self.residual_conv = nn.Identity()

        if use_cross_attn:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=out_channels,
                num_heads=cross_attn_num_heads,
                dropout=dropout,
                kdim=cond_embed_dim,
                vdim=cond_embed_dim,
                batch_first=True,
            )
            self.cross_attn_norm = nn.GroupNorm(num_groups, out_channels)

        self.sample_up_or_down = nn.Identity()
        if downsample and upsample:
            raise ValueError("Downsample and upsample cannot be True at the same time.")
        if downsample:
            self.sample_up_or_down = self._create_down_block(out_channels, out_channels)
        if upsample:
            self.sample_up_or_down = self._create_up_block(out_channels, out_channels)

    def _create_down_block(self, in_channels, out_channels):
        """Create a downsampling block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
        """
        return nn.Sequential(
            blocks.ConvNormAct(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm=nn.GroupNorm,
                norm_kwargs={"num_groups": self.num_groups},
                act=nn.SiLU,
            ),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def _create_up_block(self, in_channels, out_channels, mode: str = "bilinear"):
        """Create an upsampling block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            mode: Mode of the upsampling.
                Can be 'nearest' or 'bilinear'.
                Bilinear is more expensive but better for image reconstruction.
                Nearest is faster but less accurate.
        """
        return nn.Sequential(
            blocks.UpsampleNormAct(
                scale_factor=2.0,
                mode=mode,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                norm=nn.GroupNorm,
                norm_kwargs={"num_groups": self.num_groups},
                act=nn.SiLU,
            ),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        time_embed: torch.Tensor,
        cond_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the Conditional Convolutional Block.

        Args:
            x: Input tensor of shape (batch, channels, height, width).
            time_embed: Time embedding tensor of shape (batch, time_embed_dim).
            cond_embed: Condition embedding tensor of shape (batch, cond_embed_dim).
        """
        h = self.conv_block(x)

        t_emb = self.time_proj(time_embed)
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)
        t_scale, t_shift = t_emb.chunk(2, dim=1)
        h = self.norm(h)
        h = h * (1 + t_scale) + t_shift

        h = self.conv_block2(h)
        h = self.residual_conv(x) + h

        if self.use_cross_attn:
            h_flat = h.flatten(2).transpose(1, 2)

            h_norm = self.cross_attn_norm(h).flatten(2).transpose(1, 2)
            attn_output, _ = self.cross_attn(
                h_norm,
                cond_embed,
                cond_embed,
                attn_mask=None,
                key_padding_mask=None,
                is_causal=False,
            )

            h_flat = h_flat + attn_output
            h = h_flat.transpose(1, 2).reshape_as(h)

        h = self.sample_up_or_down(h)
        return h


class ConditionalConvStage(nn.Module):
    """Conditional Convolutional Stage used for the conditional diffusion model."""

    def __init__(
        self,
        in_channels: int,
        channels: list[int],
        time_embed_dim: int,
        cond_embed_dim: int,
        num_groups: int | list[int] = 32,
        dropout: float = 0.1,
        use_cross_attn: list[bool] | bool = True,
        cross_attn_num_heads: list[int] | int = 8,
        downsample: list[bool] | bool = False,
        upsample: list[bool] | bool = False,
        cond_level_mapping: dict[int, str] | None = None,
    ):
        """Conditional Convolutional Stage.

        Args:
            in_channels: Number of input channels.
            channels: List of output channels.
            time_embed_dim: Dimension of the time embedding.
            cond_embed_dim: Dimension of the condition embedding.
            num_groups: Number of groups in the convolution or list of numbers of groups
                for each block.
            dropout: Dropout rate.
            use_cross_attn: Whether to use cross-attention.
            cross_attn_num_heads: Number of heads in the cross-attention.
            downsample: List of booleans indicating whether to downsample the input.
            upsample: List of booleans indicating whether to upsample the input.
            cond_level_mapping: Optional mapping from block index to condition key.
                If provided, cond_embed should be a dict with keys matching the mapping values.
                Block indices: 0 to len(channels)-1 for each block in the stage.
                Example: {0: "low_level", 1: "high_level"} means
                block 0 uses cond_embed["low_level"], block 1 uses cond_embed["high_level"].
        """
        super().__init__()
        self.block = nn.ModuleList()
        self.cond_level_mapping = cond_level_mapping or {}
        block_in_channels = in_channels
        num_blocks = len(channels)

        use_cross_attn = self._create_block_params(use_cross_attn, num_blocks, "use_cross_attn")
        block_num_groups = self._create_block_params(num_groups, num_blocks, "num_groups")
        block_cross_attn_num_heads = self._create_block_params(
            cross_attn_num_heads, num_blocks, "cross_attn_num_heads"
        )
        block_downsample = self._create_block_params(downsample, num_blocks, "downsample")
        block_upsample = self._create_block_params(upsample, num_blocks, "upsample")

        for idx, channel in enumerate(channels):
            self.block.append(
                ConditionalConvBlock(
                    in_channels=block_in_channels,
                    out_channels=channel,
                    time_embed_dim=time_embed_dim,
                    cond_embed_dim=cond_embed_dim,
                    num_groups=block_num_groups[idx],
                    dropout=dropout,
                    use_cross_attn=use_cross_attn[idx],
                    cross_attn_num_heads=block_cross_attn_num_heads[idx],
                    downsample=block_downsample[idx],
                    upsample=block_upsample[idx],
                )
            )
            block_in_channels = channel

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
                f"as the number of blocks. Got {len(param)} for {param_name}, while "
                f"num_blocks is {num_blocks}."
            )
        return param

    def forward(
        self,
        x: torch.Tensor,
        time_embed: torch.Tensor,
        cond_embed: torch.Tensor | dict[str, torch.Tensor],
    ):
        """Forward pass of the Conditional Convolutional Stage.

        Args:
            x: Input tensor of shape (batch, channels, height, width).
            time_embed: Time embedding tensor of shape (batch, time_embed_dim).
            cond_embed: Condition embedding(s). Can be:
                - torch.Tensor of shape (batch, seq_len, cond_embed_dim) for single condition
                - dict[str, torch.Tensor] with condition keys when cond_level_mapping is provided
        """

        # Helper function to get the right condition for a block
        def get_cond_for_block(block_idx: int) -> torch.Tensor:
            if isinstance(cond_embed, dict):
                if block_idx in self.cond_level_mapping:
                    cond_key = self.cond_level_mapping[block_idx]
                    if cond_key not in cond_embed:
                        raise KeyError(
                            f"Condition key '{cond_key}' not found in cond_embed. "
                            f"Available keys: {list(cond_embed.keys())}"
                        )
                    return cond_embed[cond_key]
                # If no mapping for this block, use the first available condition
                # or raise an error if dict is provided but mapping is incomplete
                if self.cond_level_mapping:
                    raise KeyError(
                        f"No condition mapping found for block {block_idx}. "
                        f"Available mappings: {self.cond_level_mapping}"
                    )
                # Fallback: use first value in dict
                return next(iter(cond_embed.values()))
            # Single tensor case
            return cond_embed

        for idx, block in enumerate(self.block):
            cond = get_cond_for_block(idx)
            x = block(x, time_embed, cond)

        return x


if __name__ == "__main__":
    cond_conv_block = ConditionalConvBlock(
        in_channels=32,
        out_channels=64,
        time_embed_dim=128,
        cond_embed_dim=128,
        downsample=True,
    )
    example_x = torch.randn(1, 32, 256, 256)
    example_time_embed = torch.randn(1, 128)
    example_cond_embed = torch.randn(1, 256, 128)
    print(cond_conv_block(example_x, example_time_embed, example_cond_embed).shape)

    stage = ConditionalConvStage(
        in_channels=32,
        channels=[64, 128, 256],
        time_embed_dim=128,
        cond_embed_dim=128,
        downsample=[True, True, True],
        upsample=[False, False, False],
    )
    example_x = torch.randn(1, 32, 256, 256)
    example_time_embed = torch.randn(1, 128)
    example_cond_embed = torch.randn(1, 256, 128)
    print(stage(example_x, example_time_embed, example_cond_embed).shape)
