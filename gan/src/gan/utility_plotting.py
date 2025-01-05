## Plotting utilities
## Anthony Lee 2024-12-23

import matplotlib as mpl
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import torch
from gan.gan import CycleGAN
from torchvision.transforms import ToPILImage

def image_preview(dataset: Dataset, row: int, col: int) -> mpl.figure.Figure:
    fig, axs = plt.subplots(row, col, figsize=(10, 10))
    axs = axs.flatten()

    for i in range(row * col):
        idx = np.random.randint(low=0, high=len(dataset))
        img = dataset[idx]
        axs[i].imshow(img)
        axs[i].set_axis_off()

    fig.tight_layout()
    return fig

def plot_before_after(real:torch.Tensor, generated:torch.Tensor, suptitle:str=None) -> mpl.figure.Figure:
    """Return an Figure of before and after."""
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))

    converter = ToPILImage(mode="RGB")
    real = converter(real)
    generated = converter(generated)
    
    for ax in axs:
        ax.set_axis_off()

    # The real image
    ax = axs[0]
    ax.imshow(real)
    ax.set_title("Real")

    # The generated image
    ax = axs[1]
    ax.imshow(generated)
    ax.set_title("Generated")

    # Figure settings
    if suptitle is not None:
        fig.suptitle(suptitle)
    # fig.patch.set_edgecolor("black")
    fig.tight_layout()
    return fig
    