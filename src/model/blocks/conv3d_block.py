"""3D convolutional blocks for the model.

3D analogue of :mod:`model.blocks.conv_block`. Operates on
``(B, C, T, H, W)`` tensors (matching :class:`torch.nn.Conv3d`'s input
layout). Use a :class:`model.nn_modules.Permute` around these blocks when
upstream code emits ``(B, T, C, H, W)``.
"""

from typing import Callable, Literal

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from model import utils

KernelSpec = int | tuple[int, int, int]


class Conv3dNormAct(nn.Module):
    """Layer with a 3D convolution, normalization, and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: KernelSpec,
        stride: KernelSpec,
        padding: KernelSpec,
        groups: int | None = 1,
        dilation: KernelSpec | None = 1,
        bias: bool | None = False,
        norm: type[nn.Module] | None = nn.BatchNorm3d,
        norm_kwargs: dict | None = None,
        act: type[nn.Module] | None = nn.ReLU,
        act_kwargs: dict | None = None,
    ):
        """Layer with a 3D convolution, normalization, and activation.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size; ``int`` or 3-tuple ``(kT, kH, kW)``.
            stride: Stride; ``int`` or 3-tuple.
            padding: Padding; ``int`` or 3-tuple.
            groups: Number of groups in the convolution.
            dilation: Dilation; ``int`` or 3-tuple.
            bias: Whether to use a bias in the convolution.
            norm: Normalization class (e.g., nn.BatchNorm3d, nn.GroupNorm, None).
            norm_kwargs: Keyword arguments for initializing the normalization layer.
            act: Activation class (e.g., nn.ReLU, nn.SiLU, None).
            act_kwargs: Keyword arguments for initializing the activation layer.
        """
        super().__init__()
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
        self.norm = utils.create_normalization(norm, out_channels, norm_kwargs)
        self.act = utils.create_activation(act, act_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Conv3dNormAct."""
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class ConvBlock3d(nn.Module):
    """Block of :class:`Conv3dNormAct` layers."""

    def __init__(
        self,
        in_channels: int,
        channels: list[int],
        kernel_sizes: list[KernelSpec],
        strides: list[KernelSpec],
        paddings: list[KernelSpec],
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
        """Block of Conv3dNormAct layers.

        Args:
            in_channels: Number of input channels.
            channels: List of output channels per layer.
            kernel_sizes: List of kernel sizes per layer (``int`` or 3-tuple).
            strides: List of strides per layer.
            paddings: List of paddings per layer.
            groups: Optional list of group counts per layer.
            dilations: Optional list of dilations per layer.
            bias: Whether to use bias in the convolutions.
            norm: Normalization class (e.g., nn.BatchNorm3d, nn.GroupNorm, None).
            norm_kwargs: Keyword arguments for initializing the normalization layer.
            act: Activation class (e.g., nn.ReLU, nn.SiLU, None).
            act_kwargs: Keyword arguments for initializing the activation layer.
            checkpoint_layers: List of layer indices to checkpoint, "all" for all
                layers, or None to disable checkpointing.
            checkpoint_use_reentrant: Whether to use reentrant checkpointing.
        """
        super().__init__()
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
        self.num_layers = len(channels)

        self._setup_checkpointing(checkpoint_layers, checkpoint_use_reentrant)

        self.assert_same_lengths()
        self.conv_blocks = nn.ModuleList()
        for idx, (out_channels, kernel_size, stride, padding) in enumerate(
            zip(self.channels, self.kernel_sizes, self.strides, self.paddings)
        ):
            layer_in = self.in_channels if idx == 0 else self.channels[idx - 1]
            group = groups[idx] if groups else 1
            dilation = dilations[idx] if dilations else 1
            self.conv_blocks.append(
                Conv3dNormAct(
                    layer_in,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    group,
                    dilation,
                    bias,
                    norm,
                    norm_kwargs,
                    act,
                    act_kwargs,
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
        num_layers = len(self.channels)
        utils.assert_list_length(self.channels, num_layers, "channels")
        utils.assert_list_length(self.kernel_sizes, num_layers, "kernel_sizes")
        utils.assert_list_length(self.strides, num_layers, "strides")
        utils.assert_list_length(self.paddings, num_layers, "paddings")
        utils.assert_list_length(self.groups, num_layers, "groups")
        utils.assert_list_length(self.dilations, num_layers, "dilations")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ConvBlock3d.

        Args:
            x: ``(B, C, T, H, W)`` tensor.

        Returns:
            ``(B, C', T', H', W')`` tensor.
        """
        for idx, conv_block in enumerate(self.conv_blocks):
            x = self._run_module(conv_block, x, idx)
        return x


if __name__ == "__main__":
    model = ConvBlock3d(
        in_channels=3,
        channels=[64, 128, 256],
        kernel_sizes=[(3, 3, 3), (3, 3, 3), (3, 3, 3)],
        strides=[(1, 2, 2), (2, 2, 2), (1, 2, 2)],
        paddings=[(1, 1, 1), (1, 1, 1), (1, 1, 1)],
    )

    input_data = torch.randn(2, 3, 8, 64, 64)
    output_data = model(input_data)
    print(output_data.shape)
