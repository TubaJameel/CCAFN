import os
from pyexpat import model
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler

import config

from models.ccafn import CCAFN

from utils.dataset import get_dataloaders

from engine.train_one_epoch import train_one_epoch
from engine.validate import validate
from engine.utils import save_checkpoint


# -------------------------------------------------------
# Seed
# -------------------------------------------------------

def seed_everything(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# -------------------------------------------------------
# Train CCAFN
# -------------------------------------------------------

def main():

    seed_everything(config.SEED)

    device = torch.device(config.DEVICE)

    print("=" * 70)
    print("Confidence Aware Calibration Fusion Network")
    print("=" * 70)

    print("Device :", device)
    print("CUDA :", torch.cuda.is_available())

    # -------------------------------------------------------
    # Create folders
    # -------------------------------------------------------

    os.makedirs(

        config.CHECKPOINT_DIR,

        exist_ok=True

    )

    os.makedirs(

        config.LOG_DIR,

        exist_ok=True

    )

    # -------------------------------------------------------
    # TensorBoard
    # -------------------------------------------------------

    writer = SummaryWriter(

        os.path.join(

            config.LOG_DIR,

            "ccafn"

        )

    )

    # -------------------------------------------------------
    # Dataset
    # -------------------------------------------------------

    print()

    print("Loading Dataset...")

    train_loader, val_loader, test_loader = get_dataloaders()

    print()

    print("Train Images      :", len(train_loader.dataset))

    print("Validation Images :", len(val_loader.dataset))

    print("Test Images       :", len(test_loader.dataset))

    # -------------------------------------------------------
    # Model
    # -------------------------------------------------------

    
# -------------------------------------------------------
# Model
# -------------------------------------------------------

    model = CCAFN().to(device)

    print()
    print(model)

# -------------------------------------------------------
# Load Pretrained Backbone Checkpoints
# -------------------------------------------------------

    print()
    print("Loading pretrained backbone checkpoints...")

# VGG11
    vgg_checkpoint = torch.load(
        os.path.join(
            config.CHECKPOINT_DIR,
            "best_vgg11.pth"
        ),
        map_location=device
    )

    model.vgg.load_state_dict(
    vgg_checkpoint["model_state_dict"]
    )

    print("VGG11 weights loaded.")

# WideResNet-SE
    wrn_checkpoint = torch.load(
        os.path.join(
            config.CHECKPOINT_DIR,
            "best_wideresnet.pth"
        ),
        map_location=device
    )

    model.wrn.load_state_dict(
        wrn_checkpoint["model_state_dict"]
    )

    print("WideResNet-SE weights loaded.")

# InceptionV3
    inc_checkpoint = torch.load(
        os.path.join(
            config.CHECKPOINT_DIR,
            "best_inception.pth"
        ),
        map_location=device
    )

    model.inception.load_state_dict(
        inc_checkpoint["model_state_dict"]
    )

    print("InceptionV3 weights loaded.")

# -------------------------------------------------------
# Freeze Backbone Networks
# -------------------------------------------------------

    print()
    print("Freezing Backbone Networks...")

    for param in model.vgg.parameters():
        param.requires_grad = False

    for param in model.wrn.parameters():
        param.requires_grad = False

    for param in model.inception.parameters():
        param.requires_grad = False

    print("Done.")

# -------------------------------------------------------
# Trainable Parameters
# -------------------------------------------------------

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
            if p.requires_grad
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print()
    print(f"Trainable Parameters : {trainable_params:,}")
    print(f"Total Parameters     : {total_params:,}")


    # -------------------------------------------------------
    # Loss Function
    # -------------------------------------------------------

    criterion = nn.NLLLoss()

    # -------------------------------------------------------
    # Optimizer
    # Train ONLY Fusion Components
    # -------------------------------------------------------

    optimizer = optim.AdamW(

    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),

    lr=config.CCAFN_LEARNING_RATE,

    weight_decay=config.WEIGHT_DECAY

)
    # -------------------------------------------------------
    # Learning Rate Scheduler
    # -------------------------------------------------------

    scheduler = optim.lr_scheduler.CosineAnnealingLR(

        optimizer,

        T_max=config.EPOCHS

    )

    # -------------------------------------------------------
    # Mixed Precision
    # -------------------------------------------------------

    scaler = GradScaler(

        "cuda",

        enabled=config.USE_AMP

    )

    # -------------------------------------------------------
    # Best Accuracy
    # -------------------------------------------------------

    best_accuracy = 0.0

    print()

    print("=" * 70)

    print("Training Fusion Network")

    print("=" * 70)

    # -------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------

    for epoch in range(config.EPOCHS):

        print()

        print("=" * 70)

        print(f"Epoch {epoch + 1}/{config.EPOCHS}")

        print("=" * 70)

        # ---------------------------------------------------
        # Train
        # ---------------------------------------------------

        train_result = train_one_epoch(

            model=model,

            loader=train_loader,

            optimizer=optimizer,

            criterion=criterion,

            device=device,

            scaler=scaler

        )

        if len(train_result) == 3:

            train_loss, train_accuracy, train_confidence = train_result

        else:

            train_loss, train_accuracy = train_result

            train_confidence = None

        # ---------------------------------------------------
        # Validation
        # ---------------------------------------------------

        validation_result = validate(

            model=model,

            loader=val_loader,

            criterion=criterion,

            device=device

        )

        if len(validation_result) == 3:

            validation_loss, validation_accuracy, validation_confidence = validation_result

        else:

            validation_loss, validation_accuracy = validation_result

            validation_confidence = None

        # ---------------------------------------------------
        # Scheduler
        # ---------------------------------------------------

        scheduler.step()

        # ---------------------------------------------------
        # Print Results
        # ---------------------------------------------------

        print()

        print(f"Train Loss          : {train_loss:.4f}")

        print(f"Train Accuracy      : {train_accuracy:.2f}%")

        if train_confidence is not None:

            print(f"Train Confidence    : {train_confidence:.2f}%")

        print()

        print(f"Validation Loss     : {validation_loss:.4f}")

        print(f"Validation Accuracy : {validation_accuracy:.2f}%")

        if validation_confidence is not None:

            print(f"Validation Confidence : {validation_confidence:.2f}%")

        print()

        print(f"Learning Rate       : {scheduler.get_last_lr()[0]:.8f}")

                # -------------------------------------------------------
        # TensorBoard Logging
        # -------------------------------------------------------

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

        if train_confidence is not None:

            writer.add_scalar(

                "Confidence/Train",

                train_confidence,

                epoch

            )

        writer.add_scalar(

            "Accuracy/Validation",

            validation_accuracy,

            epoch

        )

        if validation_confidence is not None:

            writer.add_scalar(

                "Confidence/Validation",

                validation_confidence,

                epoch

            )

        writer.add_scalar(

            "Learning Rate",

            scheduler.get_last_lr()[0],

            epoch

        )

        # -------------------------------------------------------
        # Save Best Model
        # -------------------------------------------------------

        if validation_accuracy > best_accuracy:

            best_accuracy = validation_accuracy

            save_checkpoint(

                model=model,

                optimizer=optimizer,

                epoch=epoch,

                accuracy=validation_accuracy,

                path=os.path.join(

                    config.CHECKPOINT_DIR,

                    "best_ccafn.pth"

                )

            )

            print()

            print("=" * 60)

            print("Best CCAFN Model Saved")

            print(f"Validation Accuracy : {best_accuracy:.2f}%")

            print("=" * 60)

    # -------------------------------------------------------
    # Training Finished
    # -------------------------------------------------------

    print()

    print("=" * 70)

    print("Fusion Training Completed")

    print("=" * 70)

    print()

    print(f"Best Validation Accuracy : {best_accuracy:.2f}%")

    # -------------------------------------------------------
    # Close TensorBoard
    # -------------------------------------------------------

    writer.close()

    # -------------------------------------------------------
    # Load Best Model
    # -------------------------------------------------------

    checkpoint_path = os.path.join(

        config.CHECKPOINT_DIR,

        "best_ccafn.pth"

    )

    if os.path.exists(checkpoint_path):

        checkpoint = torch.load(

            checkpoint_path,

            map_location=device

        )

        model.load_state_dict(

            checkpoint["model_state_dict"]

        )

        print()

        print("=" * 60)

        print("Best CCAFN Loaded Successfully")

        print("=" * 60)

        print(f"Epoch      : {checkpoint['epoch'] + 1}")

        print(f"Accuracy   : {checkpoint['accuracy']:.2f}%")

    else:

        print()

        print("No best checkpoint found.")

    print()

    print("=" * 70)

    print("CCAFN Training Pipeline Finished Successfully")

    print("=" * 70)

    return model


# -------------------------------------------------------
# Main
# -------------------------------------------------------

if __name__ == "__main__":

    main()