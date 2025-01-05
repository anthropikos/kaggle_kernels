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
        """Generator Loss function - Provides a score for how well the generator performs.

        This loss function is used for both the monet_generator (takes a real
        photo and generates a fake monet), and a photo_generator (takes a real
        monet and generates a photo).

        The generator's goal is for the discriminator to return a tensor of ones
        because it is trying to make the discriminator believe that the image
        given to it is indeed real when in fact it was generated. 

        Being that is the goal of the generator, the loss function uses a
        tensor of all ones as the objective. Its takes the tensor returned by a 
        discriminator passed with a generator generated image and compare it
        with a tensor of ones to calculate the loss.

        According to the documentation of nn.BCEWithLogitLoss(), this binary
        cross entropy (BCE) loss is a combination of a sigmoid layer and a BCE
        layer. It has additional optimizations for numerical stability than a
        sigmoid layer plus a BCE layer.
        """
        super().__init__()

    def forward(self, generated: torch.Tensor):
        return nn.BCEWithLogitsLoss(reduction="mean")(input=generated, target=torch.ones_like(generated))


class CycleLoss(nn.Module):
    def __init__(self):
        """Cycle Loss function - A generator performance evaluation.

        The cycle loss evaluates the real image against the cycled image where
        the real image could be either monet or photo and the cycled image could
        be either a cycled monet or cycled photo.
        
        A cycled monet is created from a real monet to a generated photo using
        the photo generator, from the generated photo to a generated monet with
        the monet generator, thus the entire computation graph it utilized both 
        the photo generator and the monet generator.

        A cycled photo is created from a real photo to a generated monet using a 
        monet generator, from a generated monet to a generated photo using the 
        photo generator, thus the entire computation grpah it utilized both the
        monet generator and the photo generator.
        
        """
        super().__init__()

    def forward(self, real: torch.Tensor, cycled: torch.Tensor, scale: float):
        """Callable implementation.
        
        Args:
            real (torch.Tensor): The real or original image.
            cycled (torch.Tensor): The image that has been cycle generated.
            scale (float): Loss metric scaling factor.
        """
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
