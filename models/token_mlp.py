import torch
from torch import nn


class MotionTokenMLP(nn.Module):
    def __init__(self, input_dim, hidden_size, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, latent_dim),
        )

    def forward(self, x):
        return self.net(x)
