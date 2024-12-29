# Anthony Lee 2024-12-28

from .generator import Downsampler
import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

    def __input_size_okay(self, input:torch.Tensor) -> bool:
        """Checks if input dimension is acceptable."""
        match input.dim():
            case 3:
                dim = input.size()
                if (dim[1]!=256) | (dim[2]!=256):
                    raise AttributeError(f"Input images have to be 256x256, got {dim[1]}x{dim[2]}")
            case 4:
                dim = input.size()
                if (dim[2]!=256) | (dim[3]!=256):
                    raise AttributeError(f"Input images have to be 256x256, got {dim[2]}x{dim[3]}")
            case _:
                raise AttributeError(f"Expect input dimension size of either 3 or 4, got {input.dim()}")
    
    def forward(self, input:torch.Tensor):
        self.__input_size_okay(input=input)

        # TODO: Check the outputs of the downsamplers
        # TODO: Check the outputs of of each layers in the Generators

        output = input
        output = Downsampler(filters=64, kernel_size=4, apply_instanceNorm=False)(output)  # (batch-size, 128, 128, 64)
        output = Downsampler(128, 4)(output)  # output: (batch-size, 128, 64, 64)
        output = Downsampler(256, 4)(output)  # output: (batch_size, 256, 32, 32)
        output = nn.ZeroPad2d(padding=1)(output)  # output: (batch-size, 256, 34, 34)
        output = Downsampler(512, 4, stride=1, padding=0) # IN and ReLU included
        output = nn.ZeroPad1d(padding=1)(output)  # output: (batch-size, 512, 31, 31)
        output = Downsampler(1, 4, stride=1, padding=0)(output)

        return output