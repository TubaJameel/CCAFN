import torch
import torch.nn.functional as F


def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    # For CCAFN
    running_confidence = 0.0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # -----------------------------------------
            # Forward
            # -----------------------------------------

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

                running_confidence += confidence.mean().item()

                is_ccafn = True

            else:

                raise TypeError(
                    f"Unsupported output type: {type(output)}"
                )

            running_loss += loss.item()

            correct += (prediction == labels).sum().item()

            total += labels.size(0)

    epoch_loss = running_loss / len(loader)

    epoch_accuracy = 100.0 * correct / total

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