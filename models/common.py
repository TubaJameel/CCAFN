from dataclasses import dataclass
import torch


@dataclass
class BackboneOutput:

    features: torch.Tensor

    logits: torch.Tensor

    feature_maps: torch.Tensor


@dataclass
class CalibrationOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    confidence: torch.Tensor
    prediction: torch.Tensor