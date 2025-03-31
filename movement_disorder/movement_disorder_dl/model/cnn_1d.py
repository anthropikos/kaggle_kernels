# 2025-02-10 Anthony Lee

# Adapted from the following paper:
# Rodriguez, F., He, S., & Tan, H. (2023). The potential of convolutional 
# neural networks for identifying neural states based on electrophysiological 
# signals: Experiments on synthetic and real patient data. Frontiers in Human 
# Neuroscience, 17, 1134599. https://doi.org/10.3389/fnhum.2023.1134599

import torch
from torch import nn

class Compressor(nn.Module):
    def __init__(self, n_channels: int, amp: float = 0.0):
        """Sigmoid function that compresses the data range.
        
        This custom non-linear compression operation is able to reduce the 
        inference-time computational footprint of models while minimally 
        compromising on decoding performance (Rodriguez et al., 2023).

        Rodriguez, F., He, S., & Tan, H. (2023). The potential of 
        convolutional neural networks for identifying neural states based on 
        electrophysiological signals: Experiments on synthetic and real 
        patient data. Frontiers in Human Neuroscience, 17, 1134599. 
        https://doi.org/10.3389/fnhum.2023.1134599

        """
        super().__init__()
        self.slope = nn.Parameter(amp * torch.randn(1, n_channels, 1))

    def forward(self, x):
        eps = 1e-8
        slope_ = torch.exp(self.slope)

        return (
            torch.sign(x)
            * torch.log(torch.abs(x) * slope_ + 1.0)
            / (torch.log(slope_ + 1.0) + eps)
        )



class CNN1d(nn.Module):
    """Convolutional Neural Network with 1D kernels"""

    def __init__(self, config=None):
        super().__init__()
        
        if config is None: config = {}
        
        n_channels = config.get('n_channels', 1)
        n_hidden = config.get('n_hidden', 45)
        depth = config.get('depth', 7)
        kernel_size = config.get('kernel_size', 32)
        stride = config.get('stride', 1)
        compress_before_conv = config.get('compress_before_conv', False)

        self.n_hidden = n_hidden

        # Build convolution layers
        holder_conv_layers = []
        for idx_layer in range(depth):
            holder_conv_layers.extend(
                [
                    nn.BatchNorm1d(n_channels, affine=True)            # Initial normalization 
                        if idx_layer==0 else nn.Identity(),                         # Only at the beginning
                    Compressor(n_channels=n_channels)                               # Compress value range. TODO: Figure out what does the compressor does
                        if (compress_before_conv & (idx_layer==0)) else nn.Identity(),
                    Compressor(n_channels=self.n_hidden)                            # Compress value range. TODO: Figure out what does the compressor does
                        if (compress_before_conv & (idx_layer!=0)) else nn.Identity(),
                    nn.LazyConv1d(                                                  # 1D conv layer
                        out_channels=self.n_hidden,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding="same" if stride == 1 else 0,
                    ),
                    nn.SiLU(),                                                      # Swish activation function, better than ReLU
                    nn.AvgPool1d(2),                                                # Pooling Layer
                    nn.BatchNorm1d(self.n_hidden, affine=True),                     # Normalization
                ]
            )

        # Build classifier layers
        holder_classifier_layers = [
            nn.AdaptiveAvgPool1d(1),                                                # Maintains channel count, which is n_hidden
            nn.Flatten(),
            nn.Linear(self.n_hidden, 1),
            nn.Sigmoid(),                                                           # Probability
        ]

        # Combine all layers
        self.conv_layers = nn.Sequential(*holder_conv_layers)
        self.classifier = nn.Sequential(*holder_classifier_layers)

        return


    def forward(self, x):
        """
        Args
        ====
        x: Input of shape (n_batch, n_channels, data_length)

        Returns
        =======
        Tensor of shape (n_batch, 1) indicating the probability of pathological state.
        """

        conv_result = self.conv_layers(x)
        probability = self.classifier(conv_result)

        return probability