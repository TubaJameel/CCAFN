import torch
import torch.nn as nn


class WeightedAverageFusion(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(
        self,
        probabilities,
        weights
    ):

        """
        probabilities
        Shape:
            (Batch, Num_Models, Num_Classes)

        weights
        Shape:
            (Batch, Num_Models)
        """

        # ---------------------------------------
        # Expand weights
        # (B,3) -> (B,3,1)
        # ---------------------------------------

        weights = weights.unsqueeze(2)

        # ---------------------------------------
        # Weighted Average Fusion
        # ---------------------------------------

        fused_probabilities = torch.sum(

            probabilities * weights,

            dim=1

        )

        # ---------------------------------------
        # Numerical Stability
        # ---------------------------------------

        fused_probabilities = torch.clamp(

            fused_probabilities,

            min=1e-8,

            max=1.0

        )

        # ---------------------------------------
        # Final Prediction
        # ---------------------------------------

        prediction = torch.argmax(

            fused_probabilities,

            dim=1

        )

        # ---------------------------------------
        # Confidence
        # ---------------------------------------

        confidence = torch.max(

            fused_probabilities,

            dim=1

        ).values

        # ---------------------------------------
        # Return
        # ---------------------------------------

        return {

            "probabilities": fused_probabilities,

            "prediction": prediction,

            "confidence": confidence,

            "weights": weights.squeeze(2)

        }