import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import CalibrationOutput


class TemperatureScaling(nn.Module):

    def __init__(self):

        super().__init__()

        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):

        # Prevent division by zero
        temperature = torch.clamp(
            self.temperature,
            min=1e-6
        )

        # Calibrated logits
        calibrated_logits = logits / temperature

        # Calibrated probabilities
        probabilities = F.softmax(
            calibrated_logits,
            dim=1
        )

        # Confidence = maximum probability
        confidence, prediction = torch.max(
            probabilities,
            dim=1,
            keepdim=True
        )

        return CalibrationOutput(

            logits=calibrated_logits,

            probabilities=probabilities,

            confidence=confidence,

            prediction=prediction

        )