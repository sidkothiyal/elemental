"""Linear blocks for the model."""
import torch
from torch import nn


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
    ):
        """Linear block with batch normalization and ReLU activation.

        Args:
            dims: list of dimensions.
            out_dim: Number of output dimensions.
            bias: Whether to use a bias.
            use_bn: Whether to use batch normalization.
            use_relu: Whether to use ReLU activation.
        """
        super().__init__()

        self.linear_blocks = nn.ModuleList(
            [
                LinearBNReLU(dims[i], dims[i + 1], bias=bias, use_bn=use_bn, use_relu=use_relu)
                for i in range(len(dims) - 1)
            ]
        )
        self.linear_blocks.append(
            LinearBNReLU(dims[-1], out_dim, bias=bias, use_bn=False, use_relu=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        for linear_block in self.linear_blocks:
            x = linear_block(x)
        return x
