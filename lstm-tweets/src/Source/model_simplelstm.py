## A Simple LSTM model
## Anthony Lee 2024-12-17

from typing import List, Union, Iterable, Tuple, Dict
from multiprocessing import set_start_method, cpu_count
from collections import namedtuple

import torch
from tqdm import tqdm
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler


class SimpleLSTM(torch.nn.Module):

    def __init__(
        self,
        config: dict,
    ) -> None:
        """Simple LSTM model.

        Structure:
            LSTM-Layer(s) > Dense-Layer > Sigmoid activation function

        - Unable to to train in batches as each tweet (or document) has varied length.
        - Batch normalization is not really needed because outputs have a sigmoid activation function.
        """
        super().__init__()

        # Variables
        self.config = config
        self.target_output_size = 1
        self.dtype = torch.float64

        # Fully connected layer
        self.layer_linear = torch.nn.Linear(
            in_features=self.config["lstm_hiddenSize"],
            out_features=self.target_output_size,
            bias=config["linear_bias"],
            dtype=self.dtype,
        )

        # Final sigmoid layer
        self.layer_sigmoid = torch.nn.Sigmoid()  # transforms to probability space
        self.to(dtype=self.dtype)  # self.double() also works

        return

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Forward calculation, call the module instance instead.

        Even though this method is defined, one should call the module instance instead to make
        sure that all the registered hooks are taken care of.
        Source: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.forward
        """
        # Instantiate input layer
        if input_data.dim() != 2:
            raise ValueError(
                f"Input data expected to have 2-dimension, got {input_data.dim()}-dimension(s)."
            )
        self.input_size = input_data.shape[1]  # Vectorized encoding size

        self.layer_lstm = torch.nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.config["lstm_hiddenSize"],
            num_layers=self.config["lstm_numLayers"],
            bias=self.config["lstm_bias"],
            batch_first=True,  # Batch first is more nature, but hidden and cell state outputs are not batch first (see PyTorch documentation)
            dropout=self.config["lstm_dropoutProbability"],
            bidirectional=self.config["lstm_bidirectional"],
            dtype=self.dtype,
        )

        # LSTM layer
        # - last_layer_output: the last layer output for each sequence of the input sequence (tokens of a sentence)
        # - Output vs h_n: The former only has output for the last LSTM layer, whereas the latter has output for all LSTM layers.
        last_layer_output, (h_n, c_n) = self.layer_lstm(input_data)
        if last_layer_output.dim() == 3:
            last_layer_output = last_layer_output[
                :, -1, :
            ]  # All batches; Last sequence output; hidden size
        elif last_layer_output.dim() == 2:
            last_layer_output = last_layer_output[
                -1, :
            ]  # When no batches - Last seq output; hidden size
        else:
            raise ValueError(
                f"Output of LSTM layer expected to be either 3 or 2 dimensions (got {last_layer_output.dim()} dimensions.)"
            )

        # Dense fully connected layer
        output = torch.squeeze(last_layer_output)
        output = self.layer_linear(output)

        # Sigmoid activation function
        output = self.layer_sigmoid(output)

        return output

################################################################################
##### <<<<< ARCHIVE >>>>>
################################################################################

## Some ancient RayTune code that doesn't work well in my environment because my machine only has 16GB and thus constantly face out of memory error.
################################################################################
# config = {
#     "lstm_hiddenSize": tune.choice([10, 20, 30, 40, 50]),
#     "linear_bias": tune.choice([True, False]),
#     "lstm_numLayers": tune.choice([1, 3, 5, 7, 9]),
#     "lstm_bias": tune.choice([True, False]),
#     "lstm_dropoutProbability": tune.choice([0, 0.2, 0.4, 0.6]),
#     "lstm_bidirectional": False,
#     "adam_learningRate": tune.choice([0.001 ,0.01, 0.1]),
#     "n_epochs": tune.choice(list(range(10))),
#     "data_directory": Path("../input/").resolve().absolute()
# }

# scheduler = ASHAScheduler(
#     max_t=10,
#     grace_period=1,
#     reduction_factor=2
# )

# tuner = tune.Tuner(
#     tune.with_resources(
#         tune.with_parameters(train_SimpleLSTM_ray),
#         resources={"cpu": cpu_count()}
#     ),
#     tune_config=tune.TuneConfig(
#         metric="loss",
#         mode="min",
#         scheduler=scheduler,
#         num_samples=2,
#         # num_samples=10,
#     ),
#     param_space=config
# )
# result = tuner.fit()

# def train_SimpleLSTM_ray(
#     config: dict,
# ):
# 
#     model = SimpleLSTM(config)
#     optimizer = torch.optim.Adam(model.parameters(), lr=config["adam_learningRate"])
#     criterion = torch.nn.MSELoss(reduction="none")
# 
#     train_dataset, validation_dataset = load_dataset(data_directory=config["data_directory"])
# 
#     model.train(True)  # Training mode
# 
#     # Training Loop
#     for idx_epoch in range(config["n_epochs"]):
#         # Reset for each epoch
#         last_train_loss = 0
#         running_train_loss = 0
# 
#         for idx_data, (train_target, train_data) in enumerate(
#             tqdm(
#                 train_dataset,
#                 desc=f"    Training...; running_train_loss={round(running_train_loss, 3)}",
#                 unit="Tweet(s)",
#                 miniters=100,
#             )
#         ):
#             # Forward prop
#             model.zero_grad()  # Zero out the graident
#             optimizer.zero_grad()
#             prediction = model(torch.tensor(train_data))
# 
#             # Calculate loss
#             loss = criterion(prediction, torch.tensor(train_target).double())
# 
#             # Backward prop
#             loss.backward()  # Calculate gradients after the loss is aggregated with the reduction strategy
#             optimizer.step()  # Update parameter with gradients
# 
#             # Keep track of loss
#             last_train_loss = loss.detach()  # Solves memory leak - Or loss.item()
#             running_train_loss += last_train_loss
# 
#         # Calculate validation loss
#         validation_loss_tuple = validation_loss(
#             model=model,
#             criterion=torch.nn.MSELoss,
#             validation_dataset=validation_dataset,
#         )
#         running_validation_loss = validation_loss_tuple.running_loss
# 
#         # Save checkpoint
#         checkpoint_path = checkpoint_save(
#             model=model,
#             optimizer=optimizer,
#             epoch=idx_epoch,
#             training_loss=running_train_loss,
#             validation_loss=running_validation_loss,
#         )
# 
#         # Send results to Ray
#         checkpoint = Checkpoint.from_directory(checkpoint_path)
#         train.report({"loss": running_train_loss}, checkpoint=checkpoint)  # Send report to Ray
# 
#     return
# 