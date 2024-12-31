## Anthony Lee 2024-12-29

import torch
from torch import nn
from .generator import Generator
from .discriminator import Discriminator
from .loss import GeneratorLoss, DiscriminatorLoss, CycleLoss, IdentityLoss
from .data import ImageDataLoader


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
        total_cycle_loss = CycleLoss()(real_monet, cycled_monet, self.cycle) + CycleLoss()(
            real_photo, cycled_photo, self.cycle
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

    def train_model(
        self,
        monet_dataloader: ImageDataLoader,
        image_dataloader: ImageDataLoader,
        monet_gen_optim: torch.optim.Optimizer = None,
        photo_gen_optim: torch.optim.Optimizer = None,
        monet_dis_optim: torch.optim.Optimizer = None,
        photo_dis_optim: torch.optim.Optimizer = None,
    ):

        self.train()  # Set model to training mode

        # Create optimizers if not provided
        if monet_gen_optim is not None:
            self.monet_gen_optim = monet_gen_optim
        else:
            self.monet_gen_optim = torch.optim.Adam(self.parameters(), lr=2e-4, betas=(0.5, 0.999))
        if photo_gen_optim is not None:
            self.photo_gen_optim = photo_gen_optim
        else:
            self.photo_gen_optim = torch.optim.Adam(self.parameters(), lr=2e-4, betas=(0.5, 0.999))
        if monet_dis_optim is not None:
            self.monet_dis_optim = monet_dis_optim
        else:
            self.monet_dis_optim = torch.optim.Adam(self.parameters(), lr=2e-4, betas=(0.5, 0.999))
        if photo_dis_optim is not None:
            self.photo_dis_optim = photo_dis_optim
        else:
            self.photo_dis_optim = torch.optim.Adam(self.parameters(), lr=2e-4, betas=(0.5, 0.999))

        # Train the model one batch at a time
        for idx_batch, (real_monet_batch, real_photo_batch) in enumerate(zip(monet_dataloader, image_dataloader)):
            print(f"Training {idx_batch}-th batch...")
            self.__train_one_batch(real_monet_batch, real_photo_batch)

        return

    def __train_one_batch(self, real_monet_batch: torch.Tensor, real_photo_batch: torch.Tensor):

        # Reset gradient on Autograd tracked parameters
        self.monet_gen_optim.zero_grad()
        self.photo_gen_optim.zero_grad()
        self.monet_dis_optim.zero_grad()
        self.photo_dis_optim.zero_grad()

        # Forward pass to get the loss
        loss_dict = self(real_monet_batch, real_photo_batch)

        # Backpropagte to calculate the gradient
        for loss_name, loss_tensor in loss_dict.items():
            loss_tensor.backward()

        # Adjust parameters
        self.monet_gen_optim.step()
        self.photo_gen_optim.step()
        self.monet_dis_optim.step()
        self.photo_dis_optim.step()

        return

    def evaluate_model(self):
        self.eval()  # Set model to eval mode
        raise NotImplementedError

    def checkpoint_save(self):
        raise NotImplementedError

    def checkpoint_load(self):
        raise NotImplementedError
