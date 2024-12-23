## Abstraction of the dataset
## Anthony Lee 2024-12-22

from typing import Union, Iterable
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class ImageData(Dataset):
    """Abstract warpper for the Kaggle Monet dataset.
    
    For more information, see https://www.kaggle.com/competitions/gan-getting-started
    """
    def __init__(self, data_dir:Union[str,Path]):
        self.data_dir = Path(data_dir)
        self.datapath_list = self._ImageData__findall_jpg(self.data_dir)

    def __getitem__(self, index):
        img = self._ImageData__extract_image(self.datapath_list[index])
        return img

    def __len__(self):
        return len(self.datapath_list)

    def _ImageData__extract_image(self, image_path:Path) -> Image:
        img = Image.open(image_path)
        return img

    def _ImageData__sort_datapath_list(self, datapath_list: Iterable[Path]) -> Iterable[Path]:
        datapath_list = sorted(datapath_list, key=lambda path: path.stem)
        return datapath_list
    
    def _ImageData__findall_jpg(self, data_dir:Path) -> Iterable[Path]:
        datapath_list = list( data_dir.glob("*.jpg") )
        datapath_list = self._ImageData__sort_datapath_list(datapath_list)
        return datapath_list