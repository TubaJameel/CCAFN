import torch
import torch.nn as nn

from models.vgg11 import VGG11Model
from models.wideresnet_se import WideResNetSE
from models.inceptionv3 import InceptionV3Model

from models.calibration import TemperatureScaling
from models.fusion_mlp import ConfidenceAwareFusionMLP
from models.weighted_fusion import WeightedAverageFusion


class CCAFN(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------
        # Backbone Networks
        # ----------------------------

        self.vgg = VGG11Model()

        self.wrn = WideResNetSE()

        self.inception = InceptionV3Model()

        # ----------------------------
        # Temperature Scaling
        # ----------------------------

        self.vgg_temp = TemperatureScaling()

        self.wrn_temp = TemperatureScaling()

        self.inc_temp = TemperatureScaling()

        # ----------------------------
        # Fusion Network
        # ----------------------------

        self.fusion_mlp = ConfidenceAwareFusionMLP(

            num_models=3,

            num_classes=2

        )

        # ----------------------------
        # Weighted Fusion
        # ----------------------------

        self.weighted_fusion = WeightedAverageFusion()

    def forward(self, x):

        # ------------------------------------
        # Backbone Outputs
        # ------------------------------------

        vgg = self.vgg(x)

        wrn = self.wrn(x)

        inc = self.inception(x)

        # ------------------------------------
        # Temperature Scaling
        # ------------------------------------

        vgg = self.vgg_temp(vgg.logits)

        wrn = self.wrn_temp(wrn.logits)

        inc = self.inc_temp(inc.logits)

        # ------------------------------------
        # Fusion Input
        #
        # [Pvgg Pwgg]
        # [Pwrn]
        # [Pinc]
        # [Cvgg]
        # [Cwrn]
        # [Cinc]
        #
        # Total = 9 values
        # ------------------------------------

        fusion_input = torch.cat(

            [

                vgg.probabilities,

                wrn.probabilities,

                inc.probabilities,

                vgg.confidence,

                wrn.confidence,

                inc.confidence

            ],

            dim=1

        )

        # ------------------------------------
        # Adaptive Fusion Weights
        # ------------------------------------

        weights = self.fusion_mlp(

            fusion_input

        )

        # ------------------------------------
        # Stack Calibrated Probabilities
        #
        # Shape:
        # Batch × 3 × 2
        # ------------------------------------

        probabilities = torch.stack(

            [

                vgg.probabilities,

                wrn.probabilities,

                inc.probabilities

            ],

            dim=1

        )

        # ------------------------------------
        # Weighted Fusion
        # ------------------------------------

        output = self.weighted_fusion(

            probabilities,

            weights

        )

        # ------------------------------------
        # Store Fusion Weights
        # ------------------------------------

        output["weights"] = weights

        return output