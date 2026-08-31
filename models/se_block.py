import torch
import torch.nn as nn


class SEBlock(nn.Module):

    def __init__(self, channels, reduction=16):

        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(

            nn.Linear(channels, channels // reduction),

            nn.ReLU(inplace=True),

            nn.Linear(channels // reduction, channels),

            nn.Sigmoid()

        )

    def forward(self, x):

        batch, channels, _, _ = x.size()

        weights = self.pool(x).view(batch, channels)

        weights = self.fc(weights)

        weights = weights.view(batch, channels, 1, 1)

        return x * weights