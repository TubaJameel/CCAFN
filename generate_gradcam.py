
import os
import cv2
import numpy as np
import torch

from PIL import Image
from torchvision import transforms

import config

from visualization.gradcam import GradCAM

from models.vgg11 import VGG11Model
from models.wideresnet_se import WideResNetSE
from models.inceptionv3 import InceptionV3Model


device = torch.device(config.DEVICE)


# --------------------------------------------------
# Transform
# --------------------------------------------------

transform = transforms.Compose([

    transforms.Resize(
        (
            config.IMAGE_SIZE,
            config.IMAGE_SIZE
        )
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[0.485, 0.456, 0.406],

        std=[0.229, 0.224, 0.225]

    )

])


# --------------------------------------------------
# Image Path
# --------------------------------------------------

image_path = input(
    "Enter image path : "
).strip().strip('"')

if not os.path.exists(image_path):

    raise FileNotFoundError(
        f"Image not found:\n{image_path}"
    )


# --------------------------------------------------
# Load Image
# --------------------------------------------------

pil_image = Image.open(
    image_path
).convert("RGB")

image_tensor = transform(
    pil_image
).unsqueeze(0).to(device)

original = cv2.imread(
    image_path,
    cv2.IMREAD_COLOR
)

if original is None:

    raise RuntimeError(
        "OpenCV could not read image."
    )

original = cv2.resize(

    original,

    (
        config.IMAGE_SIZE,
        config.IMAGE_SIZE
    )

)


# --------------------------------------------------
# Output Directory
# --------------------------------------------------

save_dir = "results/gradcam"

os.makedirs(
    save_dir,
    exist_ok=True
)

cv2.imwrite(

    os.path.join(
        save_dir,
        "original.png"
    ),

    original

)


# ==================================================
# VGG11
# ==================================================

print("\nLoading VGG11...")

vgg = VGG11Model().to(device)

checkpoint = torch.load(
    os.path.join(
        config.CHECKPOINT_DIR,
        "best_vgg11.pth"
    ),
    map_location=device
)

vgg.load_state_dict(
    checkpoint["model_state_dict"]
)

vgg.eval()

vgg_cam_generator = GradCAM(
    vgg,
    vgg.model.features[-1]
)

vgg_cam = vgg_cam_generator.generate(
    image_tensor
)

# Resize and colorize
vgg_cam_vis = cv2.resize(
    vgg_cam,
    (
        config.IMAGE_SIZE,
        config.IMAGE_SIZE
    )
)

vgg_cam_vis = np.uint8(
    255 * vgg_cam_vis
)

vgg_cam_vis = cv2.applyColorMap(
    vgg_cam_vis,
    cv2.COLORMAP_JET
)

cv2.imwrite(
    os.path.join(
        save_dir,
        "vgg11_gradcam.png"
    ),
    vgg_cam_vis
)

vgg_cam_generator.remove_hooks()
# ==================================================
# WideResNet-SE
# ==================================================

print("\nLoading WideResNet-SE...")

wrn = WideResNetSE().to(device)

checkpoint = torch.load(
    os.path.join(
        config.CHECKPOINT_DIR,
        "best_wideresnet.pth"
    ),
    map_location=device
)

wrn.load_state_dict(
    checkpoint["model_state_dict"]
)

wrn.eval()

wrn_cam_generator = GradCAM(
    wrn,
    wrn.se
)

wrn_cam = wrn_cam_generator.generate(
    image_tensor
)

# Resize and colorize
wrn_cam_vis = cv2.resize(
    wrn_cam,
    (
        config.IMAGE_SIZE,
        config.IMAGE_SIZE
    )
)

wrn_cam_vis = np.uint8(
    255 * wrn_cam_vis
)

wrn_cam_vis = cv2.applyColorMap(
    wrn_cam_vis,
    cv2.COLORMAP_JET
)

cv2.imwrite(
    os.path.join(
        save_dir,
        "wideresnet_gradcam.png"
    ),
    wrn_cam_vis
)

wrn_cam_generator.remove_hooks()
# ==================================================
# InceptionV3
# ==================================================

print("\nLoading InceptionV3...")

inc = InceptionV3Model().to(device)

checkpoint = torch.load(
    os.path.join(
        config.CHECKPOINT_DIR,
        "best_inception.pth"
    ),
    map_location=device
)

inc.load_state_dict(
    checkpoint["model_state_dict"]
)

inc.eval()

inc_cam_generator = GradCAM(
    inc,
    inc.features[-1]
)

inc_cam = inc_cam_generator.generate(
    image_tensor
)

# Resize and colorize
inc_cam_vis = cv2.resize(
    inc_cam,
    (
        config.IMAGE_SIZE,
        config.IMAGE_SIZE
    )
)

inc_cam_vis = np.uint8(
    255 * inc_cam_vis
)

inc_cam_vis = cv2.applyColorMap(
    inc_cam_vis,
    cv2.COLORMAP_JET
)

cv2.imwrite(
    os.path.join(
        save_dir,
        "inception_gradcam.png"
    ),
    inc_cam_vis
)

inc_cam_generator.remove_hooks()

# ==================================================
# Aggregated GradCAM
# ==================================================

print("\nGenerating Aggregated GradCAM...")

# Resize all CAMs to image size

vgg_cam = cv2.resize(
    vgg_cam,
    (config.IMAGE_SIZE, config.IMAGE_SIZE)
)

wrn_cam = cv2.resize(
    wrn_cam,
    (config.IMAGE_SIZE, config.IMAGE_SIZE)
)

inc_cam = cv2.resize(
    inc_cam,
    (config.IMAGE_SIZE, config.IMAGE_SIZE)
)

ensemble_cam = (
    vgg_cam +
    wrn_cam +
    inc_cam
) / 3.0

ensemble_cam -= ensemble_cam.min()
ensemble_cam /= (ensemble_cam.max() + 1e-8)

ensemble_heatmap, ensemble_overlay = GradCAM.overlay(

    original,

    ensemble_cam

)

cv2.imwrite(

    os.path.join(
        save_dir,
        "aggregated_heatmap.png"
    ),

    ensemble_heatmap

)

cv2.imwrite(

    os.path.join(
        save_dir,
        "aggregated_overlay.png"
    ),

    ensemble_overlay

)


# ==================================================
# Finished
# ==================================================

print()

print("=" * 70)

print("GradCAM generation completed successfully.")

print("=" * 70)

print()

print("Saved files:")

print("original.png")

print("vgg11_gradcam.png")
print("wideresnet_gradcam.png")
print("inception_gradcam.png")

print("aggregated_heatmap.png")
print("aggregated_overlay.png")

print()
print("Results folder :", save_dir)
