# 2025-02-12 Anthony Lee

import os
from pathlib import Path
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.plugins.environments import LightningEnvironment, SLURMEnvironment
import movement_disorder_dl as md
from movement_disorder_dl.data.lfp_data_lightning import EssentialTremorLFPDataset_Posture_Lightning
from movement_disorder_dl.model.cnn_1d_lightning import CNN1d_Lightning
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


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
        trainer = pl.Trainer(max_epochs=20, plugins=[LightningEnvironment()])
        num_workers = determine_slurm_allocated_cpu_count()
        dataset = EssentialTremorLFPDataset_Posture_Lightning(batch_size=500, num_workers=num_workers)

    else: 
        # The trainer default parameter for `logger` is True and would use TensorBoard if installed, otherwise CSVLogger.
        trainer = pl.Trainer(max_epochs=20)
        num_workers = int(os.cpu_count()//2) if int(os.cpu_count()) > 2 else 1
        dataset = EssentialTremorLFPDataset_Posture_Lightning(batch_size=500, num_workers=num_workers)

    model = CNN1d_Lightning()
    
    trainer.fit(model=model, datamodule=dataset)


def tune():
    from movement_disorder_dl.tuner.hyperparameter import tune_cnn_1d 

    result_grid = tuner_cnn_1d()

    return result_grid

if __name__ == "__main__":
#    main()
    tune()
