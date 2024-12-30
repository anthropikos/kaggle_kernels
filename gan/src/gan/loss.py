## Anthony Lee 2024-12-29

import torch
from torch import nn

class DiscriminatorLoss(nn.Module):
    def __init__(self):
        """Loss function that compares the discriminator performance on a real Monet/photo vs a fake Monet/photo.
        
        "real_loss" is comparing the real Monet/photo with the generated Monet/photo.
        """
        super().__init__()

    def forward(self, real:torch.Tensor, generated:torch.Tensor):
        real_loss = nn.BCELoss(reduction="none")(input=real, target=torch.ones_like(real))
        generated_loss = nn.BCELoss(reduction="none")(input=generated, target=torch.zeros_like(generated))
        total_discriminator_loss = real_loss + generated_loss
        avg_discriminator_loss = total_discriminator_loss / 2
        
        return avg_discriminator_loss
    

class GeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, generated:torch.Tensor):
        return nn.BCELoss(reduction="none")(input=generated, target=torch.ones_like(generated))
    
class CycleLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, real:torch.Tensor, cycled:torch.Tensor, scale:float):
        loss = (cycled-real).mean()
        return loss*scale
    

class IdentityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, real_image:torch.Tensor, same_image:torch.Tensor, scale:float):
        loss = (same_image - real_image).mean()
        return loss*scale
