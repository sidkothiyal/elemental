"""Linear blocks for the model."""

from typing import Callable, Literal

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class LinearBNReLU(nn.Module):
    """Linear block with batch normalization and ReLU activation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        use_bn: bool = True,
        use_relu: bool = True,
    ):
        """Linear block with batch normalization and ReLU activation.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            bias: Whether to use a bias.
            use_bn: Whether to use batch normalization.
            use_relu: Whether to use ReLU activation.
        """
        super().__init__()
        self.use_bn = use_bn
        self.use_relu = use_relu
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if self.use_bn:
            self.bn = nn.BatchNorm1d(out_features)
        if self.use_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        x = self.linear(x)
        if self.use_bn:
            x = self.bn(x)
        if self.use_relu:
            x = self.relu(x)
        return x


class LinearBlock(nn.Module):
    """Linear block with batch normalization and ReLU activation."""

    def __init__(
        self,
        dims: list[int],
        out_dim: int,
        bias: bool = False,
        use_bn: bool = True,
        use_relu: bool = True,
        checkpoint_layers: list[int] | Literal["all"] | None = None,
        checkpoint_use_reentrant: bool = False,
    ):
        """Linear block with batch normalization and ReLU activation.

        Args:
            dims: list of dimensions.
            out_dim: Number of output dimensions.
            bias: Whether to use a bias.
            use_bn: Whether to use batch normalization.
            use_relu: Whether to use ReLU activation.
            checkpoint_layers: List of layer indices to checkpoint, "all" for all layers,
                or None to disable checkpointing.
            checkpoint_use_reentrant: Whether to use reentrant checkpointing.
        """
        super().__init__()
        self.num_layers = len(dims)  # includes the final output layer

        # Setup checkpointing
        self._setup_checkpointing(checkpoint_layers, checkpoint_use_reentrant)

        self.linear_blocks = nn.ModuleList(
            [
                LinearBNReLU(dims[i], dims[i + 1], bias=bias, use_bn=use_bn, use_relu=use_relu)
                for i in range(len(dims) - 1)
            ]
        )
        self.linear_blocks.append(
            LinearBNReLU(dims[-1], out_dim, bias=bias, use_bn=False, use_relu=False)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        for idx, linear_block in enumerate(self.linear_blocks):
            x = self._run_module(linear_block, x, idx)
        return x
