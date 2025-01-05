## Anthony Lee 2024-12-29

import torch
from torch import nn
from .generator import Generator
from .discriminator import Discriminator
from .loss import GeneratorLoss, DiscriminatorLoss, CycleLoss, IdentityLoss
from .data import ImageDataLoader
from .data_processing import normalize_image
from pathlib import Path
from typing import Union
from sys import getsizeof
from tqdm import tqdm


class CycleGAN(nn.Module):
    def __init__(
        self,
        monet_generator: Generator = None,
        photo_generator: Generator = None,
        monet_discriminator: Discriminator = None,
        photo_discriminator: Discriminator = None,
        cycle: int = 10,
    ):

        super().__init__()

        if monet_generator is None:
            monet_generator = Generator()
        if photo_generator is None:
            photo_generator = Generator()
        if monet_discriminator is None:
            monet_discriminator = Discriminator()
        if photo_discriminator is None:
            photo_discriminator = Discriminator()

        self.monet_gen = monet_generator
        self.photo_gen = photo_generator
        self.monet_dis = monet_discriminator
        self.photo_dis = photo_discriminator
        self.cycle = cycle

    def forward(self, real_monet: torch.Tensor, real_photo: torch.Tensor):

        # Generator working - 6x outputs to work with
        fake_monet = self.monet_gen(real_photo)
        fake_photo = self.photo_gen(real_monet)
        cycled_monet = self.monet_gen(fake_photo)
        cycled_photo = self.photo_gen(fake_monet)
        regenerated_monet = self.monet_gen(real_monet)
        regenerated_photo = self.photo_gen(real_photo)

        # Discriminator working
        disc_real_monet = self.monet_dis(real_monet)
        disc_fake_monet = self.monet_dis(
            fake_monet
        )  # Generator trying to make monet_dis output all 1s indicating a real Monet
        disc_real_photo = self.photo_dis(real_photo)
        disc_fake_photo = self.photo_dis(
            fake_photo
        )  # Generator trying to make photo_dis output all 1s indicating a real photo

        # Evaluate loss - cycled
        total_cycle_loss = (
            CycleLoss()(real_monet, cycled_monet, self.cycle) 
            + CycleLoss()(real_photo, cycled_photo, self.cycle)
        )

        # Evaluate loss - generator
        monet_gen_loss = GeneratorLoss()(disc_fake_monet)  # Compares against a matrix of ones
        photo_gen_loss = GeneratorLoss()(disc_fake_photo)  # Compares against a matrix of ones

        # Evaluate loss - discriminator
        monet_dis_loss = DiscriminatorLoss()(real=disc_real_monet, generated=disc_fake_monet)
        photo_dis_loss = DiscriminatorLoss()(real=disc_real_photo, generated=disc_fake_photo)

        # Total generator loss
        total_monet_gen_loss = (
            monet_gen_loss + total_cycle_loss + IdentityLoss()(real_monet, regenerated_monet, self.cycle)
        )
        total_photo_gen_loss = (
            photo_gen_loss + total_cycle_loss + IdentityLoss()(real_photo, regenerated_photo, self.cycle)
        )

        return {
            "monet_gen_loss": total_monet_gen_loss,
            "photo_gen_loss": total_photo_gen_loss,
            "monet_disc_loss": monet_dis_loss,
            "photo_disc_loss": photo_dis_loss,
        }

    def __input_check(self, input:torch.Tensor):
        input_ndim = input.dim()
        input_dim = input.size()
        
        if input_ndim != 3:
            raise ValueError(f"Input has to be unbatched, single image. Got input dimension of {input_dim}")
        if input_dim != torch.Size([3, 256, 256]):
            raise ValueError(f"Input dimension error, expected (3, 256, 256), got {input_dim}.")

    def generate_monet(self, input:torch.Tensor):
        self.__input_check(input)
        self.eval()
        input = input.reshape((1, 3, 256, 256))
        generated = self.monet_gen(input)
        output = generated.reshape((3, 256, 256))
        return output
    
    def generate_photo(self, input:torch.Tensor):
        self.__input_check(input)
        self.eval()
        input = input.reshape((1, 3, 256, 256))
        generated = self.monet_gen(input)
        output = generated.reshape((3, 256, 256))
        return output
        
def evaluate_model():
    # model.eval()  # Set model to eval mode
    raise NotImplementedError

def checkpoint_save(epoch:int, 
                    save_path:Union[str, Path],
                    model:nn.Module, 
                    optimizer:torch.optim.Optimizer,
                    loss_tracker:dict,
                    ):
    filename = f"torch_epoch_{epoch}.checkpoint"

    save_path = (Path(save_path)/Path(filename)).resolve()

    dict_to_serialize = {
        "epoch": epoch, 
        "model_state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss_tracker": loss_tracker,
    }
    torch.save(dict_to_serialize, save_path)

    return dict_to_serialize

def check_batch_size_eq(real_monet_batch:torch.Tensor, real_photo_batch:torch.Tensor) -> bool:
    return real_monet_batch.size()[0] == real_photo_batch.size()[0]

def train_one_epoch(
        monet_dataloader: ImageDataLoader,
        photo_dataloader: ImageDataLoader,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device:torch.cuda.device = torch.device("cpu"),
    ):
        model = model.to(device=device)  # Make sure to update what model references to
        model.train()  # Set model to training mode
        loss_tracker = {
            "monet_gen_loss_epoch_sum": 0,
            "photo_gen_loss_epoch_sum": 0,
            "monet_disc_loss_epoch_sum": 0,
            "photo_disc_loss_epoch_sum": 0, 
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
                optimizer = optimizer,
                device=device)

            # Update loss tracker - Make sure to not accumulate history
            # https://pytorch.org/docs/stable/notes/faq.html
            for key in loss_dict:
                loss_tracker[f"{key}_epoch_sum"] += loss_dict[key].item()  # Python number thus detached from comp graph
            num_batches += 1
        
        output = {"loss_tracker": loss_tracker, 
                "num_batches": num_batches}
        
        return output


def train_one_batch(
        real_monet_batch: torch.Tensor, 
        real_photo_batch: torch.Tensor,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device:torch.cuda.device
        ):

    if not check_batch_size_eq(real_monet_batch, real_photo_batch):
        raise ValueError(f"Inputs are expected to have the same batch size, got real_monet_batch.size(): {real_monet_batch.size()}, real_photo_batch.size(): {real_photo_batch.size()}")

    real_monet_batch = normalize_image(real_monet_batch).to(device=device)
    real_photo_batch = normalize_image(real_photo_batch).to(device=device)

    # Forward pass to get the loss
    loss_dict = model(real_monet_batch, real_photo_batch)
    combined_loss = torch.zeros((),device=device)
    for loss_name, loss_value in loss_dict.items():
        combined_loss += loss_value
    optimizer.zero_grad()
    combined_loss.backward()
    optimizer.step()

    return loss_dict