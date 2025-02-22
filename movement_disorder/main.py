# 2025-02-12 Anthony Lee

from pathlib import Path
from torch.utils.data import DataLoader
import lightning as L
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
    #trainer = L.Trainer()
    trainer = L.Trainer(accelerator='gpu', devices=1, num_nodes=1, strategy='ddp')
    model = CNN1d_Lightning()
    dataset = EssentialTremorLFPDataset_Posture_Lightning()
    
    trainer.fit(model=model, datamodule=dataset)


if __name__ == "__main__":
    main()

