# 2025-02-12 Anthony Lee

from pathlib import Path
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.plugins.environments import LightningEnvironment, SLURMEnvironment
from src import movement_disorder_dl as md
from src.movement_disorder_dl.lfp_data import EssentialTremorLFPDataset_Posture_Lightning
from src.movement_disorder_dl.model import CNN1d_Lightning


def test():
    data_dir = Path("data/essential_tremor")
    dataset = md.lfp_data.EssentialTremorLFPDataset_Posture(data_dir=data_dir)
    dataloader = DataLoader(dataset=dataset, batch_size=10)
    model = md.model.CNN1d()

    for idx, batch in enumerate(dataloader):
        if idx > 5: break

        input, label = batch

        output = model(input)


def main():

    if SLURMEnvironment().detect():
        # https://github.com/Lightning-AI/pytorch-lightning/issues/18650#issuecomment-1747669666
        trainer = L.Trainer(max_epochs=100, plugins=[LightningEnvironment()])
    else: 
        trainer = L.Trainer(max_epoch=100)
        
    model = CNN1d_Lightning()
    dataset = EssentialTremorLFPDataset_Posture_Lightning()
    
    trainer.fit(model=model, datamodule=dataset)


if __name__ == "__main__":
    main()

