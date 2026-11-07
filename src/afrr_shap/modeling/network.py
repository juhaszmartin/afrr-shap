"""Neural-network architectures used by the forecasting pipeline."""

import torch.nn as nn


class MultiOutputNN(nn.Module):
    """Two-hidden-layer MLP that predicts all horizons and targets together."""

    def __init__(self, input_dim: int, output_dim: int, lag_groups=None):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        return self.model(x)
