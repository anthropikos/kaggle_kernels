import os
import pytorch_lightning as pl
from pytorch_lightning.plugins.environments import LightningEnvironment, SLURMEnvironment
from ..data.lfp_data_lightning import EssentialTremorLFPDataset_Posture_Lightning
from ..model.cnn_1d_lightning import CNN1d_Lightning

def train_model():
    """Convenience function to train the model."""

    def determine_slurm_allocated_cpu_count(): 
        num_cpus = int(os.environ['SLURM_CPUS_PER_TASK'])
        return num_cpus
    
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