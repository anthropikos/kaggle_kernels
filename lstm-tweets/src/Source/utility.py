## Various utility functions
## Anthony Lee 2024-12-17

from typing import List, Union, Iterable, Tuple, Dict
import os
from pathlib import Path
from collections import namedtuple
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from .data_code import text_to_vector

# TODO: IGNORE - Implement batch training by logging loss every x tweets.
# TODO: DONE - Re-implement the SimpleLSTM interface to define things like layers and hidden outputs
# TODO: DONE - Re-implement the SimpleLSTM interface to remove the first argument of input size and delay the input size into the call dunder.
# TODO: IGNORED (validation process not too expensive so whatever) - Remove validation from training loop and checkpoint
# TODO: DONE - Encapsulate the entire data loading into one pipeline


TrainingLoopResult = namedtuple("training_loop_results", ["avg_train_loss", "avg_validation_loss"])
BatchedValidationLoss = namedtuple("batched_validation_loss", ["last_loss", "running_loss"])
CheckpointResult = namedtuple("checkpoint", ["model", "optimizer", "training_loss", "validation_loss"])
BatchedTrainLoss = namedtuple("batched_train_loss", ["last_loss", "running_loss"])


def evaluate_validation_loss(
    model: torch.nn.Module,
    criterion: torch.nn.modules.loss._Loss,
    validation_dataset: torch.utils.data.Dataset,
) -> BatchedValidationLoss:
    """Calculate validation loss and return last loss and running loss of the model."""

    last_validation_loss = 0
    running_validation_loss = 0

    model.train(False)  # Eval mode

    with torch.no_grad():
        for idx_data, (validation_target, validation_data) in enumerate(validation_dataset):

            # Inference
            prediction = model(torch.tensor(validation_data))

            # Calculate loss
            validation_target = torch.tensor(validation_target).double().reshape(-1)
            loss = criterion(prediction, validation_target)

            # Keep track of the loss
            last_validation_loss = loss.detach()  # Solves memory leak - Or use loss.item()
            running_validation_loss += last_validation_loss

    return BatchedValidationLoss(last_loss=last_validation_loss, running_loss=running_validation_loss)


def train_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.modules.loss._Loss,
    train_dataset: torch.utils.data.Dataset,
) -> BatchedTrainLoss:
    """Model training function, and returns the last loss and running loss as a tuple."""

    last_train_loss = 0
    running_train_loss = 0

    model.train(True)  # Training mode

    # Training loop through each data point
    # for idx_data, (train_target, train_data) in enumerate(tqdm(train_dataset, desc="    Training...", unit="Tweet")):
    for idx_data, (train_target, train_data) in tqdm(enumerate(train_dataset), desc="    Training...", unit="Tweet"):
        # if idx_batch == 5: break  # DEBUG

        # Forward prop
        model.zero_grad()  # Zero out the graident
        optimizer.zero_grad()
        prediction = model(torch.tensor(train_data))

        # Calculate loss
        train_target = torch.tensor(train_target).double().reshape(-1)
        loss = criterion(prediction, train_target)

        # Backward prop
        loss.backward()  # Calculate gradients after the loss is aggregated with the reduction strategy
        optimizer.step()  # Update parameter with gradients

        # Keep track of loss
        last_train_loss = loss.detach()  # Solves memory leak - Or loss.item()
        running_train_loss += last_train_loss

    return BatchedTrainLoss(last_loss=last_train_loss, running_loss=running_train_loss)


def checkpoint_save(model:torch.nn.Module, optimizer:torch.optim.Optimizer, epoch, training_loss, validation_loss, dir_path=None) -> Path:

    filename = f"checkpoint_epoch_{epoch}.checkpoint"

    if dir_path is None:
        dir_path = Path.cwd()
    else:
        dir_path = Path(dir_path)
        os.makedirs(dir_path, exists_ok=True)

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


def plot_train_validation_loss(avg_training_loss: np.ndarray, avg_validation_loss: np.ndarray) -> mpl.axes.Axes:
    assert len(avg_training_loss) == len(
        avg_validation_loss
    ), f"Training loss and validation loss arrays should have the same length, got {len(avg_training_loss)} and {len(avg_validation_loss)}"

    fig, ax = plt.subplots()
    marker_size = 10

    ax.set_title("Avg training and validation loss for each epoch")
    ax.set_ylabel("Avg [training | validation] loss")
    ax.set_xlabel("Epoch")

    num_of_epochs = len(avg_training_loss)

    ax.scatter(
        range(num_of_epochs),
        avg_training_loss,
        color="C1",
        s=marker_size,
        label="avg training loss",
    )
    ax.scatter(
        range(num_of_epochs),
        avg_validation_loss,
        color="C2",
        s=marker_size,
        label="avg validation loss",
    )

    ax.plot(avg_training_loss, "--", alpha=0.3, color="C1")
    ax.plot(avg_validation_loss, "--", alpha=0.3, color="C2")

    ax.legend(loc="upper right")
    ax.grid(visible=True, which="both", axis="both", alpha=0.2)
    return ax


def predict_test_data_for_submission(model: torch.nn.Module, df_test: pd.DataFrame, save: bool = None) -> List:
    """Convenience function to predict for submission."""
    if save is None:
        save = False

    test_datas_vectorized = text_to_vector(df_test.text.to_list())

    model.train(False)
    holder = []

    for item in test_datas_vectorized:
        prediction = model(torch.tensor(item)).detach()

        # Convert probability to categorical label
        if prediction > 0.5:
            prediction = 1
        else:
            prediction = 0

        holder.append(prediction)

    if save is True:
        submission = pd.DataFrame({"id": df_test.id, "target": holder})
        submission.to_csv("/kaggle/working/submission.csv", index=False)

    return holder
