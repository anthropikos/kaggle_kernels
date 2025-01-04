## Anthony Lee 2024-12-29

import torch
from torch import nn


class Upsampler(nn.Module):
    def __init__(
        self,
        filters: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        apply_dropout:bool = False,
    ):
        """Simple upsampler.

        With the default kernel_size, stirde, and padding arguemnts it should double the
        height and width.
        
        Instance normalization is also called contrast normalization, which is useful for
        stylization learning (Ulyanov et al., 2017).

        Ulyanov, D., Vedaldi, A., & Lempitsky, V. (2017). 
        Instance Normalization: The Missing Ingredient for Fast Stylization 
        (No. arXiv:1607.08022). arXiv. https://doi.org/10.48550/arXiv.1607.08022
        """
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.apply_dropout = apply_dropout
        self.layers = nn.Sequential()

        self.layers.append(
            nn.LazyConvTranspose2d(
                out_channels=self.filters, 
                kernel_size=self.kernel_size, 
                stride=self.stride, 
                padding=self.padding,
            )
        )
        self.layers.append(nn.InstanceNorm2d(num_features=self.filters))
        if self.apply_dropout:
            self.layers.append(nn.Dropout2d(0.5))
        self.layers.append(nn.LeakyReLU())

    def forward(self, input: torch.Tensor):
        self.__verify_4dim_input(input)
        output = self.layers(input)

        return output

    def __verify_4dim_input(self, input: torch.Tensor):
        """Verify that the input has 4-dimensions. batch-size x channels x height x width"""
        if input.dim() != 4:
            raise ValueError(f"Input expected to have 4 dimensions, got {input.dim()} instead.")
