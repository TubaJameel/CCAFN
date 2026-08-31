CCAFN — COVID-19 CT Classification

A Calibrated Confidence-Aware Fusion Network (CCAFN) for COVID-19 CT image classification using three CNN backbones:

VGG11 — 4096-dimensional features

WideResNet50-2 + SE — 2048-dimensional features

InceptionV3 — 2048-dimensional features

Temperature Scaling for calibrated probabilities

Confidence-Aware Fusion MLP for adaptive model weighting

NLL Loss for fusion training

Grad-CAM for visualization

Datasets

Harvard Dataverse: 4,173 images originally across 3 classes; the Other class is excluded, leaving COVID-19 and Non-COVID/Healthy images.

SARS-CoV-2 CT Scan: 2,482 images with COVID-19 and Non-COVID/Healthy classes.

Both datasets use 70% / 15% / 15% stratified train/validation/test splitting.

Training images are augmented.

Training

Backbones are trained independently using Cross-Entropy Loss, followed by frozen-backbone CCAFN fusion training using NLL Loss.

Hardware

GPU: NVIDIA GeForce RTX 4050

Framework: PyTorch

Input Size: 224 × 224


## License

The source code in this repository is licensed under the MIT License.

The datasets used in this project are not covered by this license and remain
subject to their respective licenses and terms of use.

See the `LICENSE` file for the full license text.
