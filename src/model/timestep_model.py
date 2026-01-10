"""Timestep model."""

from torch import nn


class TimestepModel(nn.Module):
    """Timestep model that projects the timestep to a tensor of the same shape as the input."""

    def __init__(self, model: nn.Module, timestep_model: nn.Module, combiner: nn.Module) -> None:
        """Initialize the TimestepModel.

        Args:
            model: The model to apply the timestep to.
            timestep_model: The model to project the timestep to a tensor of the same shape
                    as the input (e.g. a linear layer, embedding layer, etc.).
            combiner: The model to combine the input and the projected timestep.
        """
        super().__init__()
        self.model = model
        self.timestep_model = timestep_model
        self.combiner = combiner

    def forward(self, x, t):
        """Forward pass of the TimestepModel.

        Args:
            x: The input tensor.
            t: The timestep.

        Returns:
            The output tensor.
        """
        proj_t = self.timestep_model(t)
        x = self.combiner(x, proj_t)
        return self.model(x)
