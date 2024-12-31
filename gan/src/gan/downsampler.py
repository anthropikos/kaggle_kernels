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
        apply_instanceNorm: bool = True,
        apply_dropout: bool = False,
    ):
        """Simple downsampler using 2D conv, instance-normalization, and ReLU activation."""
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.apply_instanceNorm = apply_instanceNorm
        self.apply_dropout = apply_dropout
        self.padding = padding

        # Create layers
        self.conv2d = nn.LazyConv2d(
            out_channels=self.filters, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding
        )
        if apply_instanceNorm:
            self.instance_norm = nn.InstanceNorm2d(num_features=filters)
        if apply_dropout:
            self.dropout = nn.Dropout(p=0.1)  # 10% dropout
        self.relu = nn.ReLU()

    def forward(self, input: torch.Tensor):
        self.__check_dim(input)

        # Pass through the layers
        output = self.conv2d(input)
        if self.apply_instanceNorm:
            output = self.instance_norm(output)
        if self.apply_dropout:
            output = self.dropout(output)
        output = self.relu(output)

        return output

    def __check_dim(self, input: torch.Tensor):
        if input.dim() != 4:
            raise ValueError(f"Input expected to have 4 dimensions, got {input.dim()} instead.")
