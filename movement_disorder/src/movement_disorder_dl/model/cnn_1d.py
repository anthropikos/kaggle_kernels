# 2025-02-10 Anthony Lee

# Adapted from the following paper:
# Rodriguez, F., He, S., & Tan, H. (2023). The potential of convolutional 
# neural networks for identifying neural states based on electrophysiological 
# signals: Experiments on synthetic and real patient data. Frontiers in Human 
# Neuroscience, 17, 1134599. https://doi.org/10.3389/fnhum.2023.1134599

import torch
from torch import nn
import lightning as L


class Compressor(nn.Module):
    def __init__(self, n_channels: int, amp_scalar: float = 0.0):
        """Sigmoid function that compresses the data range.
        
        This custom non-linear compression operation is able to reduce the 
        inference-time computational footprint of models while minimally 
        compromising on decoding performance (Rodriguez et al., 2023).

        When `amp_scalar` is at the default 0, the learnable
        parameter is effectively not-learnable because the updated value will 
        not have an effect due to being multiplied by zero.

        Rodriguez, F., He, S., & Tan, H. (2023). The potential of 
        convolutional neural networks for identifying neural states based on 
        electrophysiological signals: Experiments on synthetic and real 
        patient data. Frontiers in Human Neuroscience, 17, 1134599. 
        https://doi.org/10.3389/fnhum.2023.1134599

        """
        super().__init__()

        self.amp_restriction_factor = nn.Parameter(
            amp_scalar 
            * torch.randn(1, n_channels, requires_grad=True)  # Learnable parmaeter
        )
        self.amp_restriction_factor = torch.exp(self.amp_restriction_factor)

    def forward(self, x):
        eps = 1e-8  # Stabilizer to prevent division by zero

        return (
            torch.sign(x)
            * torch.log(torch.abs(x) * self.amp_restriction_factor + 1.0)
            / (torch.log(self.amp_restriction_factor + 1.0) + eps)
        )



class CNN1d(nn.Module):
    """Convolutional Neural Network with 1D kernels"""

    def __init__(
        self,
        n_channels: int = 1,
        n_out: int = 1,
        n_hidden: int = 45,
        depth: int = 7,
        kernel_size: int = 25,  # 35
        stride: int = 1,
        compress: bool = True,
    ):

        super().__init__()
        self.n_hidden = n_hidden

        norm = nn.BatchNorm1d
        affine = True  # Affine allows for beta and gamma in the formula to be learnable parameters - see affine transformation

        # Convolution layers - Feature extraction
        self.convolutional_layers = [
            norm(n_channels, affine=affine) if compress else nn.Identity(),    # Initial normalization layer
            Compressor(n_channels) if compress else nn.Identity(),
            norm(n_channels, affine=affine),
            nn.Conv1d(n_channels, self.n_hidden, kernel_size, padding="same"), # Initial Convolution
            nn.SiLU(),                                                         # Nonlinearity
            norm(self.n_hidden, affine=affine),                                # Normalization               
        ]

        for _ in range(depth - 1):  # One conv layer already above
            self.convolutional_layers.extend(
                [
                    Compressor(self.n_hidden) if compress else nn.Identity(),
                    nn.Conv1d(
                        self.n_hidden,
                        self.n_hidden,
                        kernel_size,
                        stride=stride,
                        padding="same" if stride == 1 else 0,
                    ),  # Convolution
                    nn.SiLU(),                                                 # Nonlinearity
                    nn.AvgPool1d(2),                                           # Pooling Layer
                    norm(self.n_hidden, affine=affine),                        # Normalization
                ]
            )

        self.convolutions = nn.Sequential(*self.convolutional_layers)
        
        # Classifier - Single fully connected linear layer
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.n_hidden, n_out),
        )

    def forward(self, x):
        """the forward function defines how the input x is processed through the network layers

        Input:
        ------
            x: Tensor of shape (n_batch, n_channels, n_samples [here 512 samples = 0.250 s at 2048 Hz])

        Returns:
        --------
            Logits. Shape (n_batch, n_out)
        """
        return self.classifier(self.convolutions(x))


class CNN1d_Lightning(L.LightningModule):
    def __init__(self):

        self.model = CNN1d()
        self.loss_module = nn.BCEWithLogitsLoss()

    def forward(self, input, target):
        return self.model(input, target)
    
    def training_step(self, batch, batch_idx):
        input, target = batch
        output = self(input, target)
        # loss = self.loss_module()
        return output
    
    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.model.parameters(),
            lr=1e-3,
            weight_decay=1e-8,
        )