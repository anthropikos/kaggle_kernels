## Different generator model constructors for the GAN
## Anthony Lee 2024-12-23

## Questions:
## - What if the num_features argument of the instance norm layer does not match that of the number of layers in input? Why doesn't it raise an error?
## - What if in_channels of conv2d or convtranspose2d does not match input's feature number size?

## Reference:
##   - Amy Jang's CycleGAN notebook: https://www.kaggle.com/code/amyjang/monet-cyclegan-tutorial

from typing import Iterable, Callable
import torch
from torch import nn
import numpy as np
from .upsampler import Upsampler
from .downsampler import Downsampler


class Generator(nn.Module):
    def __init__(self, 
                 downsampler_factory: Callable = None, 
                 upsampler_factory: Callable = None):
        """Generator, inspired by UNET architecture."""
        super().__init__()

        if downsampler_factory is None:
            downsampler_factory = Downsampler
        if upsampler_factory is None:
            upsampler_factory = Upsampler

        # Intput size (batch-size, 3, 256, 256)
        self.downsampling_stack = nn.ParameterList(
            [
                downsampler_factory(64, apply_instancenorm=False),   # Output: (batch-size, 32, 128, 128)
                downsampler_factory(128),  # Output: (batch-size, 64, 64, 64)
                downsampler_factory(256),  # Output: (batch-size, 128, 32, 32)
                downsampler_factory(512),  # Output: (batch-size, 256, 16, 16)
                downsampler_factory(512),  # Output: (batch-size, 512, 8, 8)
                downsampler_factory(512),  # Output: (batch-size, 512, 4, 4)
                downsampler_factory(512),  # Output: (batch-size, 512, 2, 2)
                downsampler_factory(512, apply_instancenorm=False),  # Output: (batch-size, 512, 1, 1)
            ]
        )

        self.upsampling_stack = nn.ParameterList(
            [
                upsampler_factory(512, apply_dropout=True),  # Output: (batch-size, 1024, 2, 2)
                upsampler_factory(512, apply_dropout=True),  # Output: (batch-size, 1024, 4, 4)
                upsampler_factory(512, apply_dropout=True),  # Output: (batch-size, 1024, 8, 8)
                upsampler_factory(512),  # Output: (batch-size, 1024, 16, 16)
                upsampler_factory(256),  # Output: (batch-size, 512, 32, 32)
                upsampler_factory(128),  # Output: (batch-size, 256, 64, 64)
                upsampler_factory(64),   # Output: (batch-size, 128, 128, 128)
            ]
        )

        # Create last layer - Output: (batch-size, 3, 256, 256)
        self.final_layer = nn.Sequential()
        self.final_layer.append(
            nn.LazyConvTranspose2d(
                out_channels=3,
                kernel_size=4, 
                stride=2, 
                padding=1
            )
        )
        self.final_layer.append(
            nn.Tanh()
        )

    def __input_size_okay(self, input: torch.Tensor) -> bool:
        """Checks if input dimension is acceptable."""
        if input.dim() != 4:
            raise ValueError(f"Expected input dimension to be 4, got {input.dim()} instead.")
        input_size = input.size()
        if (input_size[2] != 256) | (input_size[3] != 256):
            raise ValueError(f"Input images have to be 256x256, got {input_size[2]}x{input_size[3]}")

        return

    def forward(self, input: torch.Tensor):
        self.__input_size_okay(input=input)


        # Model downsampling steps
        skips_holder = []
        output = input  # Makes it easier to pass to layers with for-loop
        for downsampling_layer in self.downsampling_stack:
            output = downsampling_layer(output)
            skips_holder.append(output)

        # Model upsampling steps
        # skips_holder = skips_holder[-2::-1]  # Not very Pythonic
        skips_holder = reversed(skips_holder[:-1])
        for upsampling_layer, skip in zip(self.upsampling_stack, skips_holder):
            output = upsampling_layer(output)

            output = torch.cat((skip, output), 1)

        # Last upsampling layer (does NOT need to concat)
        output = self.final_layer(output)

        return output
