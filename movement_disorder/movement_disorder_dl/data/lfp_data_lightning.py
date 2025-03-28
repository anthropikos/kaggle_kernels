# 2025-03-27 Anthony Lee

from torch.utils.data import Dataset, DataLoader, random_split
from .lfp_data import EssentialTremorLFPDataset_Posture
import pytorch_lightning as pl


class EssentialTremorLFPDataset_Posture_Lightning(pl.LightningDataModule):
    def __init__(self, batch_size:int=None, num_workers:int=None):
        """PyTorch Lightning wrapper for the PyTorch Dataset wrapper dataset."""
        super().__init__()

        if batch_size is None:
            batch_size = 50
        if num_workers is None: 
            num_workers = 2

        if not isinstance(batch_size, int): 
            raise TypeError(f"`batch_size` is expected to be an `int`, got {type(batch_size)}")

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.all_data = EssentialTremorLFPDataset_Posture()
        self.holdout_set, temp = random_split(self.all_data, [.1, .9])
        self.train_set, self.validation_set = random_split(temp, [.7, .3])
        
        return

    # TODO: (Later) Figure out what the LightningModule.prepare_data() does

    # TODO: (Later) Figure out what the LightningModule.setup() does

    def train_dataloader(self):
        return DataLoader(
            self.train_set, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            )
    
    def val_dataloader(self):
        return DataLoader(
            self.validation_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            )
    
    def test_dataloader(self):
        return DataLoader(
            self.holdout_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            )