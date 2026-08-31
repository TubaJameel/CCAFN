import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidenceAwareFusionMLP(nn.Module):

    def __init__(
        self,
        num_models=3,
        num_classes=2
    ):

        super().__init__()

        self.num_models = num_models
        self.num_classes = num_classes

        # --------------------------------------------------
        # Input:
        # probabilities + confidence
        #
        # 3 models × 2 probabilities = 6
        # 3 confidence scores        = 3
        #
        # Total = 9
        # --------------------------------------------------

        input_dim = (num_models * num_classes) + num_models

        self.network = nn.Sequential(

            nn.Linear(input_dim, 32),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(32, 16),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(16, num_models)

        )

    def forward(self, fusion_input):

        """
        fusion_input shape

        Batch × 9

        [P_vgg,
         P_wrn,
         P_inc,
         C_vgg,
         C_wrn,
         C_inc]
        """

        weights = self.network(fusion_input)

        weights = F.softmax(

            weights,

            dim=1

        )

        return weights