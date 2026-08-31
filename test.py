import os
import torch
import numpy as np

import config

from utils.dataset import get_dataloaders
from utils.metrics import calculate_metrics


def main(
    model,
    model_name
):

    device = torch.device(config.DEVICE)

    print("=" * 70)
    print(f"{model_name.upper()} TESTING")
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

    model = model.to(device)

    checkpoint_path = os.path.join(

        config.CHECKPOINT_DIR,

        f"best_{model_name}.pth"

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

    # --------------------------------------------------
    # Testing Loop
    # --------------------------------------------------

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # Support both tensor output and .logits output
            if hasattr(outputs, "logits"):

                logits = outputs.logits

            else:

                logits = outputs

            probs = torch.softmax(

                logits,

                dim=1

            )

            preds = torch.argmax(

                logits,

                dim=1

            )

            predictions.extend(

                preds.cpu().numpy()

            )

            labels_list.extend(

                labels.cpu().numpy()

            )

            # Probability of positive class
            probabilities.extend(

                probs[:, 1].cpu().numpy()

            )

    predictions = np.array(predictions)

    labels_list = np.array(labels_list)

    probabilities = np.array(probabilities)

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

            model_name

        )

    )

    print()

    print("=" * 70)
    print(f"{model_name.upper()} TESTING COMPLETED")
    print("=" * 70)