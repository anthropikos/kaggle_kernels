## Various utilities related to images
## Anthony Lee 2024-12-22

from typing import Union
from PIL import Image
from pathlib import Path

def scale_image(img_path:Union[str, Path], scale:float) -> Image:
    img_path = Path(img_path)
    img = Image.open(img_path)
    width, height = img.size[0], img.size[1]
    new_width, new_height = int(width*scale), int(height*scale)
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return img