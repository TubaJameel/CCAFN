import cv2
import numpy as np

import torch
import torch.nn.functional as F


class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = None

        self.register_hooks()

    # -------------------------------------------------
    # Save gradients
    # -------------------------------------------------

    def save_gradient(self, grad):

        self.gradients = grad

    # -------------------------------------------------
    # Register forward hook
    # -------------------------------------------------

    def register_hooks(self):

        def forward_hook(module, input, output):

            self.activations = output

            output.register_hook(self.save_gradient)

        self.forward_handle = self.target_layer.register_forward_hook(
            forward_hook
        )

    # -------------------------------------------------
    # Remove hooks
    # -------------------------------------------------

    def remove_hooks(self):

        if self.forward_handle is not None:

            self.forward_handle.remove()

    # -------------------------------------------------
    # Generate GradCAM
    # -------------------------------------------------

    def generate(self, image_tensor, class_index=None):

        self.model.eval()

        image_tensor = image_tensor.clone().detach().requires_grad_(True)

        output = self.model(image_tensor)

        logits = output.logits

        if class_index is None:

            class_index = torch.argmax(
                logits,
                dim=1
            ).item()

        score = logits[:, class_index]

        self.model.zero_grad()

        score.backward(retain_graph=True)

        gradients = self.gradients
        activations = self.activations

        if gradients is None:

            raise RuntimeError(
                "Gradients were not captured.\n"
                "Check that the selected target layer is correct."
            )

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = torch.sum(
            weights * activations,
            dim=1
        )

        cam = F.relu(cam)

        cam = cam.squeeze()

        cam = cam.detach().cpu().numpy()

        cam -= cam.min()

        cam /= (cam.max() + 1e-8)

        return cam

    # -------------------------------------------------
    # Overlay Heatmap
    # -------------------------------------------------

    @staticmethod
    def overlay(image, cam, alpha=0.4):

        h, w = image.shape[:2]

        cam = cv2.resize(
            cam,
            (w, h)
        )

        heatmap = np.uint8(
            255 * cam
        )

        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        overlay = cv2.addWeighted(
            image,
            1 - alpha,
            heatmap,
            alpha,
            0
        )

        return heatmap, overlay