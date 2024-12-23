## Abstraction of the dataset
## Anthony Lee 2024-12-22

from typing import Union
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

class KaggleMonet(Dataset):
    def __init__(self, data_dir:Union[str,Path]):
        self.data_dir = Path(data_dir)
        self.data_path_list = data_dir.glob("*.jpg")


    def __getitem__(self, index):
        return super().__getitem__(index)
    pass

class KagglePhoto(Dataset):
    pass