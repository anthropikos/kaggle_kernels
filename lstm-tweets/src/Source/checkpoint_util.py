## Various checkpoint utility
## Anthony Lee 2024-12-20

import torch
from pathlib import Path
import os

def checkpoint_save(model:torch.nn.Module, optimizer:torch.optim.Optimizer, epoch, training_loss, validation_loss, dir_path=None) -> Path:

    filename = f"checkpoint_epoch_{epoch}.checkpoint"

    if dir_path is None:
        dir_path = Path.cwd()
    else:
        dir_path = Path(dir_path)
        os.makedirs(dir_path, exist_ok=True)

    file_path = dir_path / Path(filename)

    dict_to_save = {
        "epoch": epoch,
        "training_loss": training_loss,
        "validation_loss": validation_loss,
        "model_class_name": model.__class__.__name__,
        "model_state_dict": model.state_dict(),
        "optimizer_class_name": optimizer.__class__.__name__,
        "optimizer_state_dict": optimizer.state_dict(),
    }

    torch.save(dict_to_save, file_path)

    return file_path


# def checkpoint_load(file_path) -> tuple:
#     file_path = Path(file_path)
#     checkpoint = torch.load(file_path, weights_only=False)

#     return checkpoint


# def checkpoint_load_into_objects(checkpoint) -> tuple:
#     """Use the provided checkpoint to return stateful model and optimizer instances."""

#     # Checkpoint structure check
#     for key_name in [
#         "model_class",
#         "optimizer_class",
#         "training_loss",
#         "validation_loss",
#     ]:
#         if key_name not in checkpoint.keys():
#             raise KeyError(f"The checkpoint is incorrect, {key_name} is missing")

#     # Create and load model state dict
#     model = checkpoint["model_class"]()  # Instantiate using the class name
#     model.load_state_dict(checkpoint["model_state_dict"])

#     # Create and load optimizer state dict
#     optimizer = checkpoint["optimizer_class"](model.parameters())  # Instantiate using the class name
#     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

#     # Load loss information
#     training_loss = checkpoint["training_loss"]
#     validation_loss = checkpoint["validation_loss"]

#     return CheckpointResult(
#         model=model,
#         optimizer=optimizer,
#         training_loss=training_loss,
#         validation_loss=validation_loss,
#     )