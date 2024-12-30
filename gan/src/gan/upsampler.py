## Anthony Lee 2024-12-29

import torch
from torch import nn

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