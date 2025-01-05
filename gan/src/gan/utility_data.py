# Anthony Lee 2025-01-03
import torch
from typing import Union
from PIL import Image
from pathlib import Path

def map_rgb_to_tanh(image:torch.Tensor):
    """Normalizes the image from the [0,255] range to [-1, 1] scale."""
    image = image.to(dtype=torch.float32)  # In case the image isn't in this dtype
    image = image / (255/2) - 1
    return image

def map_tanh_to_rgb(image:torch.Tensor):
    """Map from the tanh range to RGB range."""
    image = image.to(dtype=torch.float32)  # Just in case it is the wrong dtype
    image = (image+1) * (255/2)
    return image

def scale_image(img_path: Union[str, Path], scale: float) -> Image:
    img_path = Path(img_path)
    img = Image.open(img_path)
    width, height = img.size[0], img.size[1]
    new_width, new_height = int(width * scale), int(height * scale)
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return img
