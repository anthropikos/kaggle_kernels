# Anthony Lee 2024-12-28

from .generator import Downsampler
from typing import Callable
import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, downsampler_factory: Callable = None):
        """Discriminator model; inspired by UNET architecture."""

        super().__init__()
        if downsampler_factory is None:
            downsampler_factory = Downsampler

        self.layers = nn.Sequential()
        self.layers.append(downsampler_factory(filters=64, kernel_size=4))                         # output: (batch-size, 64, 128, 128)
        self.layers.append(downsampler_factory(128, 4))                                            # output: (batch-size, 128, 64, 64)
        self.layers.append(downsampler_factory(256, 4))                                            # output: (batch_size, 256, 32, 32)
        self.layers.append(nn.ZeroPad2d(padding=1))                                                # output: (batch-size, 256, 34, 34)
        self.layers.append(downsampler_factory(512, 4, stride=1, padding=0))                       # output: (batch-size, 512, 31, 31) InstanceNormalization and ReLU included; 
        self.layers.append(nn.ZeroPad2d(padding=1))                                                # output: (batch-size, 512, 33, 33)
        self.layers.append(nn.LazyConv2d(out_channels=1, kernel_size=4, stride=1, padding=0))      # output: (batch-size, 1, 30, 30)

    def forward(self, input: torch.Tensor):
        self.__input_size_okay(input=input)
        output = self.layers(input)

        return output

    def __input_size_okay(self, input: torch.Tensor) -> bool:
        """Checks if input dimension is acceptable."""
        if input.dim() != 4:
            raise ValueError(f"Expected input dimension size to be 4, got {input.dim()} instead.")
        input_size = input.size()
        if (input_size[2] != 256) | (input_size[3] != 256):
            raise ValueError(f"Input have to have (batch-size, 3, 256, 256), got {input_size} instead.")
        return
