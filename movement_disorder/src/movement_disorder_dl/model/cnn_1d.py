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

    def __init__(
        self,
        *,
        n_channels: int = 1,
        n_hidden: int = 45,
        depth: int = 7,
        kernel_size: int = 32,
        stride: int = 1,
        compress_before_conv: bool = False,
    ):

        super().__init__()

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
                    nn.BatchNorm1d(self.n_hidden, affine=True),           # Normalization
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


class CNN1d_Lightning(L.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = CNN1d()
        self.loss_module = nn.BCELoss(reduction="mean")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):

        input, target = batch
        output = self(input)
        loss = self.loss_module(output, target)

        # Logs the mean loss for each epoch to the logger
        self.log("train_loss", 
                 loss, 
                 on_step=True, 
                 on_epoch=True, 
                 prog_bar=True, 
                 logger=True,
                 reduce_fx=torch.mean,
        )

        return loss
    
    def validation_step(self, batch, batch_idx):
        input, target = batch
        output = self.model(input)
        loss = self.loss_module(output, target)
        self.log("val_loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        input, target = batch
        output = self.model(input)
        loss = self.loss_module(output, target)
        self.log("test_loss", loss)

        return loss

    def predict_step(self, batch):
        input, target = batch
        return self.model(input)
    
    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.model.parameters(),
            lr=1e-3,
            weight_decay=1e-8,
        )