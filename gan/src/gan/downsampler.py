## Anthony Lee 2024-12-29
import torch
from torch import nn


class Downsampler(nn.Module):
    # NOTE: `nn.InstanceNorm2d` `num_features` parameter does not impact the output since the `affine` param is `False`.
    def __init__(
        self,
        filters: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
    ):
        """Simple downsampler.
        
        With the default kernel_size, stirde, and padding arguemnts it should half the
        height and width.

        Ulyanov, D., Vedaldi, A., & Lempitsky, V. (2017). 
        Instance Normalization: The Missing Ingredient for Fast Stylization 
        (No. arXiv:1607.08022). arXiv. https://doi.org/10.48550/arXiv.1607.08022
        """
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.layers = nn.Sequential()

        # Add layers
        self.layers.append(
            nn.LazyConv2d(  # Default initialization is Xavier unif initialization
                out_channels=self.filters, 
                kernel_size=self.kernel_size, 
                stride=self.stride, 
                padding=self.padding,
            )
        )
        self.layers.append(nn.InstanceNorm2d(num_features=self.filters))
        self.layers.append(nn.LeakyReLU())
        
    def forward(self, input: torch.Tensor):
        self.__verify_4dim_input(input)
        output = self.layers(input)
        
        return output

    def __verify_4dim_input(self, input: torch.Tensor):
        """Verify that the input has 4-dimensions. batch-size x channels x height x width"""
        if input.dim() != 4:
            raise ValueError(f"Input expected to have 4 dimensions, got {input.dim()} instead.")
