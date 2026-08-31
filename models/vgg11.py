import torch
import torch.nn as nn

from torchvision.models import vgg11
from torchvision.models import VGG11_Weights
from models.common import BackboneOutput

import config


class VGG11Model(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = vgg11(
            weights=VGG11_Weights.DEFAULT
        )

        # Freeze convolution layers
        for param in self.model.features.parameters():
            param.requires_grad = False

        # Save feature dimension
        self.feature_dim = 4096

        # Replace classifier
        self.model.classifier[6] = nn.Linear(
            self.feature_dim,
            config.NUM_CLASSES
        )

    def forward(self, x):

    # Last convolution feature maps

        feature_maps = self.model.features(x)

        x = self.model.avgpool(feature_maps)

        x = torch.flatten(x, 1)

        x = self.model.classifier[:6](x)

        features = x

        logits = self.model.classifier[6](features)

        return BackboneOutput(

        features=features,

        logits=logits,

        feature_maps=feature_maps

    )