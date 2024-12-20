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
from .data_processing_util import text_to_vector

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


