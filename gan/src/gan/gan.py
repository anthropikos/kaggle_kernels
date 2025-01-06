## Anthony Lee 2024-12-29

import torch
from torch import nn
from .generator import Generator
from .discriminator import Discriminator
from .loss import GeneratorLoss, DiscriminatorLoss, CycleLoss, IdentityLoss
from .data import ImageDataset, ImageDataLoader
from .utility_data import map_rgb_to_tanh, map_tanh_to_rgb
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
            "monet_dis_loss": monet_dis_loss,
            "photo_dis_loss": photo_dis_loss,
        }

    def __input_check(self, input:torch.Tensor):
        input_ndim = input.dim()
        input_dim = input.size()
        
        if input_ndim != 3:
            raise ValueError(f"Input has to be unbatched, single image. Got input dimension of {input_dim}")
        if input_dim != torch.Size([3, 256, 256]):
            raise ValueError(f"Input dimension error, expected (3, 256, 256), got {input_dim}.")

    def generate_monet(self, input:torch.Tensor):
        """Input should be in tanh range"""
        self.__input_check(input)
        self.eval()
        input = input.reshape((1, 3, 256, 256))
        generated = self.monet_gen(input)
        output = generated.reshape((3, 256, 256))
        return output
    
    def generate_photo(self, input:torch.Tensor):
        """Input should be in tanh range"""
        self.__input_check(input)
        self.eval()
        input = input.reshape((1, 3, 256, 256))
        generated = self.photo_gen(input)
        output = generated.reshape((3, 256, 256))
        return output