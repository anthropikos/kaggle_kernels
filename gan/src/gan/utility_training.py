# Anthony Lee 2025-01-05

from typing import Union
from pathlib import Path
import torch
from torch import nn
from .gan import CycleGAN
from .data import ImageDataset, ImageDataLoader
from tqdm import tqdm
from .utility_data import map_rgb_to_tanh
from .utility_checkpoint import checkpoint_save

def check_batch_size_eq(real_monet_batch:torch.Tensor, real_photo_batch:torch.Tensor) -> bool:
    return real_monet_batch.size()[0] == real_photo_batch.size()[0]

def train_one_epoch(
        monet_dataloader: ImageDataLoader,
        photo_dataloader: ImageDataLoader,
        model: CycleGAN,
        monet_gen_optim:torch.optim.Optimizer,
        photo_gen_optim:torch.optim.Optimizer,
        monet_dis_optim:torch.optim.Optimizer,
        photo_dis_optim:torch.optim.Optimizer,
        device:torch.cuda.device = torch.device("cpu"),
    ):
    """Train one epoch"""
    model = model.to(device=device)  # Make sure to update what model references to
    model.train()  # Set model to training mode
    loss_tracker = {
        "monet_gen_loss_epoch_sum": 0,
        "photo_gen_loss_epoch_sum": 0,
        "monet_dis_loss_epoch_sum": 0,
        "photo_dis_loss_epoch_sum": 0, 
    }
    num_batches = 0

    # Train the model one batch at a time
    for idx_batch, (real_monet_batch, real_photo_batch) in tqdm(enumerate(zip(monet_dataloader, photo_dataloader)), desc="Training...", unit="batch"):
        
        # print(f"Training {idx_batch}-th batch...")

        # Forward pass and get loss dict
        loss_dict = train_one_batch(
            real_monet_batch=real_monet_batch, 
            real_photo_batch=real_photo_batch,
            model = model,
            monet_gen_optim=monet_gen_optim,
            photo_gen_optim=photo_gen_optim,
            monet_dis_optim=monet_dis_optim,
            photo_dis_optim=photo_dis_optim,
            device=device)

        # Update loss tracker - Make sure to not accumulate history
        # https://pytorch.org/docs/stable/notes/faq.html
        for key in loss_dict:
            loss_tracker[f"{key}_epoch_sum"] += loss_dict[key].item()  # Python number thus detached from comp graph
        num_batches += 1
    
    loss_tracker = {
        "loss_tracker": loss_tracker, 
        "num_batches": num_batches
        }
    
    return loss_tracker

def train_one_batch(
        real_monet_batch: torch.Tensor, 
        real_photo_batch: torch.Tensor,
        model: nn.Module,
        monet_gen_optim:torch.optim.Optimizer,
        photo_gen_optim:torch.optim.Optimizer,
        monet_dis_optim:torch.optim.Optimizer,
        photo_dis_optim:torch.optim.Optimizer,
        device:torch.cuda.device
        ):

    if not check_batch_size_eq(real_monet_batch, real_photo_batch):
        raise ValueError(f"Inputs are expected to have the same batch size, got real_monet_batch.size(): {real_monet_batch.size()}, real_photo_batch.size(): {real_photo_batch.size()}")

    real_monet_batch = map_rgb_to_tanh(real_monet_batch).to(device=device)
    real_photo_batch = map_rgb_to_tanh(real_photo_batch).to(device=device)

    # Forward pass to get the loss
    loss_dict = model(real_monet_batch, real_photo_batch)

    # Collect all optimizers
    all_optim = [
        monet_gen_optim,
        photo_gen_optim,
        monet_dis_optim,
        photo_dis_optim,
    ]

    # Propagating multiple optimizers: https://discuss.pytorch.org/t/struggling-with-a-runtime-error-related-to-in-place-operations/102222/6

    # Zero out all optimizers
    for optim in all_optim:
        optim.zero_grad()

    # Loss backpropagate
    loss_dict["monet_gen_loss"].backward(retain_graph=True)
    loss_dict["photo_gen_loss"].backward(retain_graph=True)
    loss_dict["monet_dis_loss"].backward(retain_graph=True)
    loss_dict["photo_dis_loss"].backward(retain_graph=False)

    # Step the gradient
    for optim in all_optim:
        optim.step()

    return loss_dict


def training_loop(monet_data_dir:Union[str, Path], photo_data_dir:Union[str,Path], epochs:int=10, save_checkpoint:bool=False, checkpoint_data_dir: Union[str, Path]=None) -> CycleGAN:
    if (save_checkpoint) & (checkpoint_data_dir is None):
        raise ValueError(f"Checkpoint data directory has to be provided if save_checkpoint is True.")

    # Determine the device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print( torch.cuda.get_device_properties(device) )

    # Create the model and optimizers
    model = CycleGAN()
    monet_gen_optim = torch.optim.Adam(model.monet_gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
    photo_gen_optim = torch.optim.Adam(model.photo_gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
    monet_dis_optim = torch.optim.Adam(model.monet_dis.parameters(), lr=2e-4, betas=(0.5, 0.999))
    photo_dis_optim = torch.optim.Adam(model.photo_dis.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # Get the data

    monet_dataset = ImageDataset(data_dir=monet_data_dir)
    photo_dataset = ImageDataset(data_dir=photo_data_dir)

    monet_dataloader = ImageDataLoader(monet_dataset)  # Auto shuffle at each epoch
    photo_dataloader = ImageDataLoader(photo_dataset)  # Auto shuffle at each epoch

    ## Train through the epochs
    for idx_epoch in range(epochs):
        loss_tracker = train_one_epoch(
            monet_dataloader=monet_dataloader, 
            photo_dataloader=photo_dataloader,
            model=model,
            monet_gen_optim=monet_gen_optim, 
            photo_gen_optim=photo_gen_optim,
            monet_dis_optim=monet_dis_optim,
            photo_dis_optim=photo_dis_optim,
            device=device
        )

        if (save_checkpoint) & (idx_epoch%2==0):
            _ = checkpoint_save(
                epoch=idx_epoch, 
                save_path=checkpoint_data_dir, 
                model = model,
                monet_gen_optim=monet_gen_optim, 
                photo_gen_optim=photo_gen_optim, 
                monet_dis_optim=monet_dis_optim, 
                photo_dis_optim=photo_dis_optim,
                loss_tracker=loss_tracker,
                )
    return model