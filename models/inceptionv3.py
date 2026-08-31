

import torch
import torch.nn as nn

from torchvision.models import (
    inception_v3,
    Inception_V3_Weights
)

from models.common import BackboneOutput

import config


class InceptionV3Model(nn.Module):

    def __init__(self):

        super().__init__()

        backbone = inception_v3(
            weights=Inception_V3_Weights.DEFAULT
        )

        # Disable auxiliary classifier
        backbone.aux_logits = False
        backbone.AuxLogits = None

        # Freeze all layers
        for param in backbone.parameters():
            param.requires_grad = False

        # Fine-tune the last Inception block
        for param in backbone.Mixed_7c.parameters():
            param.requires_grad = True

        self.features = nn.Sequential(

            backbone.Conv2d_1a_3x3,
            backbone.Conv2d_2a_3x3,
            backbone.Conv2d_2b_3x3,
            backbone.maxpool1,

            backbone.Conv2d_3b_1x1,
            backbone.Conv2d_4a_3x3,
            backbone.maxpool2,

            backbone.Mixed_5b,
            backbone.Mixed_5c,
            backbone.Mixed_5d,

            backbone.Mixed_6a,
            backbone.Mixed_6b,
            backbone.Mixed_6c,
            backbone.Mixed_6d,
            backbone.Mixed_6e,

            backbone.Mixed_7a,
            backbone.Mixed_7b,
            backbone.Mixed_7c
        )

        self.avgpool = backbone.avgpool

        self.dropout = nn.Dropout(0.5)

        self.feature_dim = 2048

        self.classifier = nn.Linear(
            self.feature_dim,
            config.NUM_CLASSES
        )

    def forward(self, x):

    # Last convolution feature maps
        feature_maps = self.features(x)

        x = self.avgpool(feature_maps)

        x = torch.flatten(x, 1)

        features = self.dropout(x)

        logits = self.classifier(features)  

        return BackboneOutput(

        features=features,

        logits=logits,

        feature_maps=feature_maps

    )