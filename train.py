import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler

import config

from utils.dataset import get_dataloaders

from engine.train_one_epoch import train_one_epoch
from engine.validate import validate
from engine.utils import save_checkpoint
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

def seed_everything(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------
# Generic Training Function
# ---------------------------------------------------------

def main(
    model,
    model_name
):

    # -----------------------------------------------------
    # Random Seed
    # -----------------------------------------------------

    seed_everything(config.SEED)

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

    device = torch.device(config.DEVICE)

    print("=" * 70)
    print(f"Training {model_name.upper()}")
    print("=" * 70)

    print("Device :", device)
    print("CUDA :", torch.cuda.is_available())

    # -----------------------------------------------------
    # Create Folders
    # -----------------------------------------------------

    os.makedirs(
        config.CHECKPOINT_DIR,
        exist_ok=True
    )

    os.makedirs(
        config.LOG_DIR,
        exist_ok=True
    )

    # -----------------------------------------------------
    # TensorBoard
    # -----------------------------------------------------

    writer = SummaryWriter(

        os.path.join(

            config.LOG_DIR,

            model_name

        )

    )

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    print()

    print("Loading Dataset...")

    train_loader, val_loader, test_loader = get_dataloaders()

    print()

    print("Train Images      :", len(train_loader.dataset))

    print("Validation Images :", len(val_loader.dataset))

    print("Test Images       :", len(test_loader.dataset))

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = model.to(device)

    print()

    print(model)

    # -----------------------------------------------------
    # Loss Function
    # -----------------------------------------------------

    class_weights = torch.tensor(
        [0.69, 1.84],
        device=device
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------

    optimizer = optim.AdamW(

        model.parameters(),

        lr=config.LEARNING_RATE,

        weight_decay=config.WEIGHT_DECAY

    )

    # -----------------------------------------------------
    # Scheduler
    # -----------------------------------------------------

    scheduler = optim.lr_scheduler.CosineAnnealingLR(

        optimizer,

        T_max=config.EPOCHS

    )

    # -----------------------------------------------------
    # Mixed Precision
    # -----------------------------------------------------

    scaler = GradScaler(

        "cuda",

        enabled=config.USE_AMP

    )

    # -----------------------------------------------------
    # Best Validation Accuracy
    # -----------------------------------------------------
    # -----------------------------------------------------
    # Store Metrics History
    # -----------------------------------------------------

    train_losses = []
    validation_losses = []

    train_accuracies = []
    validation_accuracies = []
    best_accuracy = 0.0
    print()

    print("=" * 70)
    print("Training Started")
    print("=" * 70)

    # -----------------------------------------------------
    # Training Loop
    # -----------------------------------------------------

    for epoch in range(config.EPOCHS):

        print()

        print("=" * 70)
        print(f"Epoch {epoch + 1}/{config.EPOCHS}")
        print("=" * 70)

        # -------------------------------------------------
        # Train One Epoch
        # -------------------------------------------------

        train_loss, train_accuracy = train_one_epoch(

            model=model,

            loader=train_loader,

            optimizer=optimizer,

            criterion=criterion,

            device=device,

            scaler=scaler

        )

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        validation_loss, validation_accuracy = validate(

            model=model,

            loader=val_loader,

            criterion=criterion,

            device=device

        )

        # -------------------------------------------------
        # Learning Rate Scheduler
        # -------------------------------------------------

        scheduler.step()

        # -------------------------------------------------
        # Display Results
        # -------------------------------------------------

        print()

        print(f"Train Loss           : {train_loss:.4f}")

        print(f"Train Accuracy       : {train_accuracy:.2f}%")

        print()

        print(f"Validation Loss      : {validation_loss:.4f}")

        print(f"Validation Accuracy  : {validation_accuracy:.2f}%")

        print()

        print("Learning Rate :", scheduler.get_last_lr()[0])

        # -------------------------------------------------
        # TensorBoard Logging
        # -------------------------------------------------

        writer.add_scalar(

            "Loss/Train",

            train_loss,

            epoch

        )

        writer.add_scalar(

            "Loss/Validation",

            validation_loss,

            epoch

        )

        writer.add_scalar(

            "Accuracy/Train",

            train_accuracy,

            epoch

        )

        writer.add_scalar(

            "Accuracy/Validation",

            validation_accuracy,

            epoch

        )

        writer.add_scalar(

            "Learning Rate",

            scheduler.get_last_lr()[0],

            epoch

        )

        # -----------------------------------------------------
# Store Metrics History
# -----------------------------------------------------

        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        train_accuracies.append(train_accuracy)
        validation_accuracies.append(validation_accuracy)

        # -------------------------------------------------
        # Save Best Model
        # -------------------------------------------------

        if validation_accuracy > best_accuracy:

            best_accuracy = validation_accuracy

            save_checkpoint(

                model=model,

                optimizer=optimizer,

                epoch=epoch,

                accuracy=validation_accuracy,

                path=os.path.join(

                    config.CHECKPOINT_DIR,

                    f"best_{model_name}.pth"

                )

            )

            print()

            print("Best Model Updated!")

            print(f"Best Validation Accuracy : {best_accuracy:.2f}%")
                # -----------------------------------------------------
    # Training Finished
    # -----------------------------------------------------

    print()

    print("=" * 70)
    print("Training Completed")
    print("=" * 70)

    print()

    print(f"Best Validation Accuracy : {best_accuracy:.2f}%")

  

# -----------------------------------------------------
# Training and Validation Loss Curve
# -----------------------------------------------------
    plt.figure(figsize=(8, 6))

    plt.plot(
        range(1, len(train_losses) + 1),
        train_losses,
        label="Training Loss"
    )

    plt.plot(
        range(1, len(validation_losses) + 1),
        validation_losses,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            config.LOG_DIR,
            f"{model_name}_loss_curve.png"
        )
    )


    plt.close()


# -----------------------------------------------------
# Training and Validation Accuracy Curve
# -----------------------------------------------------

    plt.figure(figsize=(8, 6))

    plt.plot(
        range(1, len(train_accuracies) + 1),
        train_accuracies,
        label="Training Accuracy"
    )

    plt.plot(
        range(1, len(validation_accuracies) + 1),
        validation_accuracies,
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            config.LOG_DIR,
            f"{model_name}_accuracy_curve.png"
        )
    )

    plt.close()

    # -----------------------------------------------------
    # Close TensorBoard
    # -----------------------------------------------------

    writer.close()

    # -----------------------------------------------------
    # Load Best Model
    # -----------------------------------------------------

    best_model_path = os.path.join(

        config.CHECKPOINT_DIR,

        f"best_{model_name}.pth"

    )

    if os.path.exists(best_model_path):

        checkpoint = torch.load(

            best_model_path,

            map_location=device

        )

        model.load_state_dict(

            checkpoint["model_state_dict"]

        )

        print()

        print("=" * 70)

        print("Best Model Loaded Successfully")

        print("=" * 70)

        print(f"Checkpoint Epoch       : {checkpoint['epoch'] + 1}")

        print(f"Validation Accuracy    : {checkpoint['accuracy']:.2f}%")

    else:

        print()

        print("Best model checkpoint was not found!")

    print()

    print("=" * 70)
    print(f"{model_name.upper()} Training Pipeline Finished Successfully")
    print("=" * 70)

    return model


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    print()

    print("=" * 70)

    print("This file is a generic trainer.")

    print()

    print("Use one of the following scripts instead:")

    print()

    print("python train_vgg11.py")

    print("python train_wideresnet.py")

    print("python train_inceptionv3.py")

    print("=" * 70)