## Anthony Lee 2024-12-29

import torch
from torch import nn
from .generator import Generator
from .discriminator import Discriminator
from .loss import GeneratorLoss, DiscriminatorLoss, CycleLoss, IdentityLoss

class CycleGAN(nn.Module):
    def __init__(
            self,
            monet_generator:Generator, 
            photo_generator:Generator,
            monet_discriminator:Discriminator,
            photo_discriminator:Discriminator,
            cycle:int=10):
        
        super().__init__()
        self.monet_gen = monet_generator
        self.photo_gen = photo_generator
        self.monet_dis = monet_discriminator
        self.photo_dis = photo_discriminator
        self.cycle = cycle
        self.monet_gen_optim = torch.optim.Adam(self.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.photo_gen_optim = torch.optim.Adam(self.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.monet_dis_optim = torch.optim.Adam(self.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.photo_dis_optim = torch.optim.Adam(self.parameters(), lr=2e-4, betas=(0.5, 0.999))

    def forward(self, real_monet:torch.Tensor, real_photo:torch.Tensor):

        # Generator working - 6x outputs to work with
        fake_monet = self.monet_gen(real_photo)
        fake_photo = self.photo_gen(real_monet)
        cycled_monet = self.monet_gen(fake_photo)
        cycled_photo = self.photo_gen(fake_monet)
        regenerated_monet = self.monet_gen(real_monet)
        regenerated_photo = self.photo_gen(real_photo)

        # Discriminator working
        disc_real_monet = self.monet_dis(real_monet)
        disc_fake_monet = self.monet_dis(fake_monet)  # Generator trying to make monet_dis output all 1s indicating a real Monet
        disc_real_photo = self.photo_dis(real_photo)
        disc_fake_photo = self.photo_dis(fake_photo)  # Generator trying to make photo_dis output all 1s indicating a real photo

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
            monet_gen_loss 
            + total_cycle_loss 
            + IdentityLoss()(real_monet, regenerated_monet, self.cycle)
        )
        total_photo_gen_loss = (
            photo_gen_loss
            + total_cycle_loss
            + IdentityLoss()(real_photo, regenerated_photo, self.cycle)
        )

        return {
            "monet_gen_loss": total_monet_gen_loss, 
            "photo_gen_loss": total_photo_gen_loss,
            "monet_disc_loss": monet_dis_loss,
            "photo_disc_loss": photo_dis_loss,
        }


    def train(self, real_monet, real_photo, monet_gen_optim=None, photo_gen_optim=None, monet_dis_optim=None, photo_dis_optim=None):
        self.train()
        if monet_gen_optim is not None:
            self.monet_gen_optim = monet_gen_optim
        if photo_gen_optim is not None: 
            self.photo_gen_optim = photo_gen_optim
        if monet_dis_optim is not None:
            self.monet_dis_optim = monet_dis_optim
        if photo_dis_optim is not None:
            self.photo_dis_optim = photo_dis_optim

        self.monet_gen_optim.zero_grad()
        self.photo_gen_optim.zero_grad()
        self.monet_dis_optim.zero_grad()
        self.photo_dis_optim.zero_grad()

        output = self(real_monet, real_photo)
        
        raise NotImplementedError

    def evaluate(self):
        self.eval()
        raise NotImplementedError

    def checkpoint_save(self):
        raise NotImplementedError

    def checkpoint_load(self):
        raise NotImplementedError