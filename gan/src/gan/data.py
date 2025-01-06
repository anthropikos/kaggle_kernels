## Abstraction of the dataset
## Anthony Lee 2024-12-22

from typing import Union, Iterable
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import warnings
from torchvision.io import decode_jpeg
from torchvision.transforms import ToTensor, PILToTensor


class ImageDataset(Dataset):
    """Abstract warpper for the Kaggle Monet dataset."""

    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = Path(data_dir)
        self.datapath_list = self.__findall_jpg(self.data_dir)

        if not self.data_dir.is_dir():
            raise ValueError("Provided data_dir is not a directory.")

        if len(self.datapath_list) == 0:
            warnings.warn("There are no images in the dataset.", UserWarning)

        return

    def __getitem__(self, index) -> torch.Tensor:
        img = self.__to_RGB_tensor(self.datapath_list[index])
        return img

    def __len__(self) -> int:
        return len(self.datapath_list)

    def get_image(self, index: int) -> Image:
        return self.__to_PIL_image(self.datapath_list[index])

    def __to_PIL_image(self, image_path: Path) -> Image:
        """Return an PIL Image object."""
        img = Image.open(image_path)
        return img

    def __to_RGB_tensor(self, image_path: Path) -> torch.Tensor:
        """Return the image as a tensor of dtype int."""
        img = self.__to_PIL_image(image_path)
        converter = PILToTensor()
        img = converter(img)
        return img

    def __sort_datapath_list(self, datapath_list: Iterable[Path]) -> Iterable[Path]:
        datapath_list = sorted(datapath_list, key=lambda path: path.stem)
        return datapath_list

    def __findall_jpg(self, data_dir: Path) -> Iterable[Path]:
        datapath_list = list(data_dir.glob("*.jpg"))
        datapath_list = self.__sort_datapath_list(datapath_list)
        return datapath_list


class ImageDataLoader(DataLoader):
    def __init__(self, dataset: ImageDataset):
        super().__init__(
            dataset=dataset, 
            batch_size=5, 
            num_workers=2, 
            shuffle=True,
            drop_last=True,  # Ensures that all outputs have same batch sizes
        )
        return
