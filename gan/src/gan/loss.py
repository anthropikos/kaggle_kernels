## Anthony Lee 2024-12-29

import torch
from torch import nn


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        """Loss function that compares the discriminator performance on a real Monet/photo vs a fake Monet/photo.

        "real_loss" is comparing the real Monet/photo with the generated Monet/photo.
        """
        super().__init__()

    def forward(self, real: torch.Tensor, generated: torch.Tensor):
        real_loss = nn.BCEWithLogitsLoss(reduction="mean")(input=real, target=torch.ones_like(real))
        generated_loss = nn.BCEWithLogitsLoss(reduction="mean")(input=generated, target=torch.zeros_like(generated))
        avg_discriminator_loss = (real_loss + generated_loss) / 2

        return avg_discriminator_loss


class GeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, generated: torch.Tensor):
        return nn.BCEWithLogitsLoss(reduction="mean")(input=generated, target=torch.ones_like(generated))


class CycleLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, real: torch.Tensor, cycled: torch.Tensor, scale: float):
        loss = torch.abs(real-cycled).mean()  # ??? Consider why cycle loss is abs(real-cycled)
        scaled_loss = loss * scale
        return scaled_loss


class IdentityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, real_image: torch.Tensor, same_image: torch.Tensor, scale: float):
        loss = torch.abs(real_image-same_image).mean()
        scaled_loss = loss * 0.5 * scale
        return scaled_loss
