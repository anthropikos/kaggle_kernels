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

class Downsampler(nn.Module):
    # TODODONE: Uses a conv2d layer to convolve the image smaller
    # TODODONE: conv2d -> instance-normalizer -> leakyrelu activation
    # TODODONE: Create an init dunder and a override the forward method
    # NOTE: `nn.InstanceNorm2d` `num_features` parameter does not impact the output since the `affine` param is `False`.

    def __init__(self, filters:int, kernel_size:int=4, stride:int=2, padding:int=1, apply_instanceNorm:bool=True, apply_dropout:bool=False):
        """Simple downsampler using 2D conv, instance-normalization, and ReLU activation.
        """
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.apply_instanceNorm = apply_instanceNorm
        self.apply_dropout = apply_dropout
        self.padding = padding

        # Create layers
        if apply_instanceNorm:
            self.instance_norm = nn.InstanceNorm2d(num_features=filters)
        if apply_dropout:
            self.dropout = nn.Dropout(p=0.1)  # 10% dropout
        self.relu = nn.ReLU()

    def forward(self, input:torch.Tensor):
        match input.dim():
            case 3:
                in_channels = input.shape[0]
            case 4:
                in_channels = input.shape[1]
            case _:
                raise AttributeError(f"Unable to determine input channels, expected 4 or 3 but got {input.dim()}")
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=self.filters, kernel_size=self.kernel_size, stride=self.stride, padding=1)

        # Pass through the layers
        output = self.conv2d(input)
        if self.apply_instanceNorm:
            output = self.instance_norm(output)
        if self.apply_dropout:
            output = self.dropout(output)
        output = self.relu(output)

        return output

class Upsampler(nn.Module):
    def __init__(self, filters:int, kernel_size:int=4, stride:int=2, padding:int=1, apply_instanceNorm:bool=True, apply_dropout:bool=False):
        """Simple upsampler using 2D transposed-conv, instance-normalization, and ReLU activation.
        """
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.apply_instanceNorm = apply_instanceNorm
        self.apply_dropout = apply_dropout
        self.padding = padding

        # Create layers
        if apply_instanceNorm:
            self.instance_norm = nn.InstanceNorm2d(num_features=filters)
        if apply_dropout:
            self.dropout = nn.Dropout(p=0.1)  # 10% dropout
        self.relu = nn.ReLU()

    def forward(self, input:torch.Tensor):
        match input.dim(): 
            case 3:
                in_channels = input.shape[0]
            case 4:
                in_channels = input.shape[1]
            case _:
                raise AttributeError(f"Unable to determine input channels, expected 4 or 3 but got {input.dim()}")
        self.convtranspose2d = nn.ConvTranspose2d(in_channels=in_channels, out_channels=self.filters, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

        # Pass through the layers
        output = self.convtranspose2d(input)
        if self.apply_instanceNorm:
            output = self.instance_norm(output)
        if self.apply_dropout:
            output = self.dropout(output)
        output = self.relu(output)

        return output

class Generator(nn.Module):
    def __init__(self, Downsampler:Callable, Upsampler:Callable):
        super().__init__()

        # Intput size (batch-size, 3, 256, 256)
        self.downsampling_stack = [
            Downsampler(32), # Output: (batch-size, 32, 128, 128)
            Downsampler(64), # Output: (batch-size, 64, 64, 64)
            Downsampler(128), # Output: (batch-size, 128, 32, 32)
            Downsampler(256), # Output: (batch-size, 256, 16, 16)
            Downsampler(512), # Output: (batch-size, 512, 8, 8)
            Downsampler(512), # Output: (batch-size, 512, 4, 4)
            Downsampler(512), # Output: (batch-size, 512, 2, 2)
        ]

        self.upsampling_stack = [
            Upsampler(512), # Output: (batch-size, 1024, 4, 4)
            Upsampler(512), # Output: (batch-size, 1024, 8, 8)
            Upsampler(256), # Output: (batch-size, 512, 16, 16)
            Upsampler(128), # Output: (batch-size, 256, 32, 32)
            Upsampler(64), # Output: (batch-size, 128, 64, 64)
            Upsampler(32), # Output: (batch-size, 64, 128, 128)
        ]

        self.final_layer = Upsampler(3)

    def __channel_concat(self, layers:Iterable[np.ndarray]):
        """Concat the ndarray along channel dimension depending on whether input is batched."""
        match layers[0].dim():
            case 3:  # Input is NOT batched thus have ndim of 3
                return torch.cat(layers, 0)
            case 4:  # Input is batched thus have ndim of 4
                return torch.cat(layers, 1)
            case _:
                raise AttributeError(f"Expect input array dimension size of either 3 or 4, got {layers[0].dim()}")
            
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

        # Unet downsampling steps
        skips_holder = []
        output = input  # Makes it easier to work with
        for idx, downsampling_layer in enumerate(self.downsampling_stack):
            output = downsampling_layer(output)
            skips_holder.append(output)

        # Unet upsampling steps
        # skips_holder = skips_holder[-2::-1]  # Not very Pythonic
        skips_holder = reversed(skips_holder[:-1])
        for idx, (upsampling_layer, skip) in enumerate(zip(self.upsampling_stack, skips_holder)):
            output = upsampling_layer(output)
            print(f"{idx}, output.size(): {output.size()}, skip.size(): {skip.size()}")
            ouptut = self.__channel_concat((skip, output))

        # Last upsampling layer (does NOT need to concat)
        output = self.final_layer(ouptut)

        return output

