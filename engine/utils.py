import torch


def save_checkpoint(

    model,

    optimizer,

    epoch,

    accuracy,

    path

):

    torch.save(

        {

            "epoch": epoch,

            "model_state_dict": model.state_dict(),

            "optimizer_state_dict": optimizer.state_dict(),

            "accuracy": accuracy

        },

        path

    )

    print(f"\nBest model saved to {path}")