# Anthony Lee 2025-01-03
import torch

def normalize_image(image:torch.Tensor):
    """Normalizes the image from the [0,255] range to [-1, 1] scale."""
    image = image.to(dtype=torch.float32)  # In case the image isn't in this dtype
    image = image / (255/2) - 1
    return image

def map_tanh_to_rgb(image:torch.Tensor):
    """Map from the tanh range to RGB range."""
    image = image.to(dtype=torch.float32)  # Just in case it is the wrong dtype
    image = (image+1) * (255/2)
    return image