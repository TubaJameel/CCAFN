import os

import torch
import numpy as np

import config

from models.ccafn import CCAFN

from utils.dataset import get_dataloaders
from utils.metrics import calculate_metrics


def main():

    device = torch.device(config.DEVICE)

    print("=" * 70)
    print("CCAFN TESTING")
    print("=" * 70)

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    _, _, test_loader = get_dataloaders()

    print()
    print("Test Images :", len(test_loader.dataset))

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = CCAFN().to(device)

    checkpoint_path = os.path.join(

        config.CHECKPOINT_DIR,

        "best_ccafn.pth"

    )

    checkpoint = torch.load(

        checkpoint_path,

        map_location=device

    )

    model.load_state_dict(

        checkpoint["model_state_dict"]

    )

    print()
    print("Checkpoint Loaded Successfully")
    print(f"Epoch      : {checkpoint['epoch'] + 1}")
    print(f"Accuracy   : {checkpoint['accuracy']:.2f}%")

    model.eval()

    predictions = []
    labels_list = []
    probabilities = []
    confidences = []

    # --------------------------------------------------
    # Testing Loop
    # --------------------------------------------------

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            prediction = outputs["prediction"]

            probability = outputs["probabilities"][:, 0]

            confidence = outputs["confidence"]

            predictions.extend(
                prediction.cpu().numpy()
            )

            labels_list.extend(
                labels.cpu().numpy()
            )

            probabilities.extend(
                probability.cpu().numpy()
            )

            confidences.extend(
                confidence.cpu().numpy()
            )

    predictions = np.array(predictions)

    labels_list = np.array(labels_list)

    probabilities = np.array(probabilities)

    confidences = np.array(confidences)

    # --------------------------------------------------
    # Confidence Statistics
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("Confidence Statistics")
    print("=" * 70)

    print(f"Average Confidence : {confidences.mean():.4f}")
    print(f"Maximum Confidence : {confidences.max():.4f}")
    print(f"Minimum Confidence : {confidences.min():.4f}")

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    calculate_metrics(

        labels=labels_list,

        predictions=predictions,

        probabilities=probabilities,

        class_names=config.CLASS_NAMES,

        save_dir=os.path.join(

            "results",

            "ccafn"

        )

    )

    print()
    print("=" * 70)
    print("CCAFN TESTING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":

    main()