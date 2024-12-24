## Different generator model constructors for the GAN
## Anthony Lee 2024-12-23

## Reference: 
##   - Amy Jang's CycleGAN notebook: https://www.kaggle.com/code/amyjang/monet-cyclegan-tutorial

from torch import nn
import numpy as np

class Downsampler(nn.Module):
    # TODO: DONE Uses a conv2d layer to convolve the image smaller
    # TODO: DONE conv2d -> instance-normalizer -> leakyrelu activation
    # TODO: DONE Create an init dunder and a override the forward method

    def __init__(self, filters:int, kernel_size:int, stride:int=1, apply_instanceNorm:bool=True):
        self.filters = filters
        self.kernel_size = kernel_size
        self.apply_instanceNorm = apply_instanceNorm
        self.stride = stride

        if apply_instanceNorm:
            self.instance_norm = nn.InstanceNorm2d(num_features=filters)
        self.relu = nn.ReLU()

    def forward(self, input:np.ndarray):
        if input.dim() == 4:
            in_channels = input.shape[1]
        elif input.dim() == 3:
            in_channels = input.shape[0]
        else:
            raise ValueError(f"Unable to determine input channels, expected 4 or 3 but got {input.dim()}")
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=self.filters, kernel_size=self.kernel_size, stride=self.stride)

        output = self.conv2d(input)
        if self.apply_instanceNorm:
            output = self.instance_norm(output)
        output = self.relu(output)

        return output

class Upsampler(nn.Module):
    pass

class Generator(nn.Module):
    pass


