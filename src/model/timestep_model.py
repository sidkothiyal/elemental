from torch import nn


class TimestepModel(nn.Module):
    def __init__(self, model: nn.Module, timestep_model: nn.Module, combiner: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.timestep_model = timestep_model
        self.combiner = combiner

    def forward(self, x, t):
        proj_t = self.timestep_model(t)
        x = self.combiner(x, proj_t)
        x = self.model(x, t)
        return x
