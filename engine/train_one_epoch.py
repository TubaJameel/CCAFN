import torch
import torch.nn.functional as F
from tqdm import tqdm


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler=None
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    # For CCAFN confidence
    running_confidence = 0.0

    progress = tqdm(
        loader,
        desc="Training",
        leave=False
    )

    for images, labels in progress:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # -----------------------------------------
        # Mixed Precision Forward Pass
        # -----------------------------------------

        with torch.amp.autocast(
            device_type="cuda",
            enabled=(scaler is not None)
        ):

            output = model(images)

            # -----------------------------------------
            # Backbone Models
            # -----------------------------------------

            if hasattr(output, "logits"):

                logits = output.logits

                loss = criterion(
                    logits,
                    labels
                )

                prediction = torch.argmax(
                    logits,
                    dim=1
                )

                is_ccafn = False

            # -----------------------------------------
            # CCAFN
            # -----------------------------------------

            elif isinstance(output, dict):

                probabilities = output["probabilities"]

                prediction = output["prediction"]

                confidence = output["confidence"]

                loss = F.nll_loss(
                    torch.log(
                        torch.clamp(
                            probabilities,
                            min=1e-8
                        )
                    ),
                    labels
                )

                is_ccafn = True

            else:

                raise TypeError(
                    f"Unsupported output type: {type(output)}"
                )

        # -----------------------------------------
        # Backpropagation
        # -----------------------------------------

        if scaler is not None:

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

        else:

            loss.backward()

            optimizer.step()

        # -----------------------------------------
        # Statistics
        # -----------------------------------------

        running_loss += loss.item()

        correct += (prediction == labels).sum().item()

        total += labels.size(0)

        accuracy = 100.0 * correct / total

        # -----------------------------------------
        # Confidence (Only for CCAFN)
        # -----------------------------------------

        if is_ccafn:

            running_confidence += confidence.mean().item()

            avg_confidence = (
                running_confidence / (progress.n + 1)
            ) * 100

            progress.set_postfix(

                Loss=f"{loss.item():.4f}",

                Accuracy=f"{accuracy:.2f}%",

                Confidence=f"{avg_confidence:.2f}%"

            )

        else:

            progress.set_postfix(

                Loss=f"{loss.item():.4f}",

                Accuracy=f"{accuracy:.2f}%"

            )

    # -----------------------------------------
    # Epoch Statistics
    # -----------------------------------------

    epoch_loss = running_loss / len(loader)

    epoch_accuracy = 100.0 * correct / total

    # -----------------------------------------
    # Return
    # -----------------------------------------

    if running_confidence > 0:

        epoch_confidence = (
            running_confidence / len(loader)
        ) * 100

        return (
            epoch_loss,
            epoch_accuracy,
            epoch_confidence
        )

    return (
        epoch_loss,
        epoch_accuracy
    )