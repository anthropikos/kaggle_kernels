# 2025-02-12 Anthony Lee

import os
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


def determine_slurm_allocated_cpu_count(): 

    num_cpus = int(os.environ['SLURM_CPUS_PER_TASK'])

    return num_cpus

def main():

    if SLURMEnvironment().detect():
        # https://github.com/Lightning-AI/pytorch-lightning/issues/18650#issuecomment-1747669666
        trainer = L.Trainer(max_epochs=20, plugins=[LightningEnvironment()])
        num_workers = determine_slurm_allocated_cpu_count()
        dataset = EssentialTremorLFPDataset_Posture_Lightning(batch_size=500, num_workers=num_workers)

    else: 
        trainer = L.Trainer(max_epochs=20)
        num_workers = int(os.cpu_count()//2) if int(os.cpu_count()) > 2 else 1
        dataset = EssentialTremorLFPDataset_Posture_Lightning(batch_size=500, num_workers=num_workers)

    model = CNN1d_Lightning()
    
    trainer.fit(model=model, datamodule=dataset)


if __name__ == "__main__":
    main()

