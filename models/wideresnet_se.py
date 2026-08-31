

import torch
import torch.nn as nn
from models.common import BackboneOutput

from torchvision.models import (
    wide_resnet50_2,
    Wide_ResNet50_2_Weights
)

from models.se_block import SEBlock

import config


class WideResNetSE(nn.Module):

    def __init__(self):

        super().__init__()

        backbone = wide_resnet50_2(
            weights=Wide_ResNet50_2_Weights.DEFAULT
        )

        # Freeze backbone
        for param in backbone.parameters():
            param.requires_grad = False

        # Fine tune only layer4
        for param in backbone.layer4.parameters():
            param.requires_grad = True

        self.stem = nn.Sequential(

            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool

        )

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.se = SEBlock(2048)

        self.avgpool = backbone.avgpool

        self.feature_dim = 2048

        self.classifier = nn.Linear(
            self.feature_dim,
            config.NUM_CLASSES
        )

    def forward(self, x):

        x = self.stem(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x = self.se(x)

        x = self.avgpool(x)

        features = torch.flatten(x, 1)

        logits = self.classifier(features)

        import torch
import torch.nn as nn
from models.common import BackboneOutput

from torchvision.models import (
    wide_resnet50_2,
    Wide_ResNet50_2_Weights
)

from models.se_block import SEBlock

import config


class WideResNetSE(nn.Module):

    def __init__(self):

        super().__init__()

        backbone = wide_resnet50_2(
            weights=Wide_ResNet50_2_Weights.DEFAULT
        )

        # Freeze backbone
        for param in backbone.parameters():
            param.requires_grad = False

        # Fine tune only layer4
        for param in backbone.layer4.parameters():
            param.requires_grad = True

        self.stem = nn.Sequential(

            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool

        )

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.se = SEBlock(2048)

        self.avgpool = backbone.avgpool

        self.feature_dim = 2048

        self.classifier = nn.Linear(
            self.feature_dim,
            config.NUM_CLASSES
        )

    def forward(self, x):

        x = self.stem(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

    # Last convolution feature maps
        feature_maps = self.se(x)

        x = self.avgpool(feature_maps)

        features = torch.flatten(x, 1)

        logits = self.classifier(features)

        return BackboneOutput(

        features=features,

        logits=logits,

        feature_maps=feature_maps

    )