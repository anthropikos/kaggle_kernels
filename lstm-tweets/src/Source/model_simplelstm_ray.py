## Archival code for implementing Ray Tune
## Anthony Lee 2024-12-18

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