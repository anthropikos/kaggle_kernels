## Different generator model constructors for the GAN
## Anthony Lee 2024-12-23

## Reference: 
##   - Amy Jang's CycleGAN notebook: https://www.kaggle.com/code/amyjang/monet-cyclegan-tutorial

from torch import nn

class Downsampler(nn.Module):
    # - Uses a conv2d layer to convolve the image smaller
    # - conv2d -> instance-normalizer -> leakyrelu activation
    # - Create an init dunder and a override the forward method

    def __init__(self, filters:int, kernel_size:int, apply_instanceNorm:bool=True):
       pass 

    def forward(self):
        pass

class Upsampler(nn.Module):
    pass

class Generator(nn.Module):
    pass


