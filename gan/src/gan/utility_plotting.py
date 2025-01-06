## Plotting utilities
## Anthony Lee 2024-12-23

import matplotlib as mpl
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import torch
from gan.gan import CycleGAN
from torchvision.transforms import ToPILImage
import warnings


def image_preview(dataset: Dataset, row: int, col: int) -> mpl.figure.Figure:
    fig, axs = plt.subplots(row, col, figsize=(10, 10))
    axs = axs.flatten()

    for i in range(row * col):
        ax = axs[i]
        idx = np.random.randint(low=0, high=len(dataset))
        img = dataset[idx]
        ax.imshow(img)
        ax.set_axis_off()

    fig.tight_layout()
    return fig

def plot_single_RGB_tensor(image_RGB_tensor:torch.Tensor, ax: mpl.axes.Axes=None) -> mpl.axes.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3))

    if (image_RGB_tensor.dtype != torch.uint8) & (image_RGB_tensor.dtype != torch.int8):
        warnings.warn(f"image_RGB_tensor expects torch.uint8 or torch.int8, got {image_RGB_tensor.dtype}")
    
    converter = ToPILImage(mode='RGB')
    img = converter(image_RGB_tensor)
    ax.imshow(img)
    ax.set_axis_off()

    return ax

def plot_before_after(real_tensor:torch.Tensor, generated_tensor:torch.Tensor, suptitle:str=None) -> mpl.figure.Figure:
    """Return an Figure of before and after."""

    nrows, ncols, inches = 1, 2, 3
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*inches, nrows*inches))

    converter = ToPILImage(mode="RGB")
    real_img = converter(real_tensor)
    generated_img = converter(generated_tensor)
    
    for ax in axs:
        ax.set_axis_off()

    # The real image
    ax = axs[0]
    ax.imshow(real_img)
    ax.set_title(f"Real (dtype: {real_tensor.dtype})")

    # The generated image
    ax = axs[1]
    ax.imshow(generated_img)
    ax.set_title(f"Generated (dtype: {generated_tensor.dtype})")

    # Figure settings
    if suptitle is not None:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig
    
def plot_different_dtypes(image:torch.Tensor) -> mpl.figure.Figure:
    """Plots images when converted to different dtypes.

    Args:
        image (torch.Tensor): A tensor representing an RGB image.
    
    This convenient plotting function shows that different tensor dtypes creates
    different images even through they were all converted by the
    `torchvision.transforms.ToPILImage(mode="RGB")` class instance. The slight 
    floating point error (I suspect it is floating point error) creates additional
    artifacts in the image.  
    """
    if image.size() != torch.Size([3, 256, 256]):
        raise ValueError(f"Expects image tensor to have size (3, 256, 256), got {image.size()}.")

    nrows, ncols, inches = 2, 4, 3
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*inches, nrows*inches))
    
    # Plot (0, 0) and (1, 0)
    ax = axs[0, 0]
    image = image.to(dtype=torch.int8)
    ax.set_title(image.dtype)
    plot_single_RGB_tensor(image, ax=ax)

    # Plot (0, 1) - int16
    ax = axs[0, 1]
    image = image.to(dtype=torch.int16)
    ax.set_title(image.dtype)
    plot_single_RGB_tensor(image, ax=ax)

    # Plot (0, 2) - int32
    ax = axs[0, 2]
    image = image.to(dtype=torch.int32)
    ax.set_title(image.dtype)
    plot_single_RGB_tensor(image, ax=ax)

    # Plot (0, 3) - int64
    ax = axs[0, 3]
    image = image.to(dtype=torch.int64)
    ax.set_title(image.dtype)
    plot_single_RGB_tensor(image, ax=ax)

    # Plot (1, 0) - uint 8
    ax = axs[1, 0]
    image = image.to(dtype=torch.uint8)
    ax.set_title(image.dtype)
    plot_single_RGB_tensor(image, ax=ax)

    # Plot (1, 1) - float16
    ax = axs[1, 1]
    image = image.to(dtype=torch.float16)
    ax.set_title(image.dtype)
    plot_single_RGB_tensor(image, ax=ax)

    # Plot (1, 2) - float32
    ax = axs[1, 2]
    image = image.to(dtype=torch.float32)
    ax.set_title(image.dtype)
    plot_single_RGB_tensor(image, ax=ax)

    # Plot (1, 3) - float64
    ax = axs[1, 3]
    image = image.to(dtype=torch.float64)
    ax.set_title(image.dtype)
    plot_single_RGB_tensor(image, ax=ax)


    fig.suptitle("Different dtype tensors create different results")
    fig.tight_layout()

    return fig