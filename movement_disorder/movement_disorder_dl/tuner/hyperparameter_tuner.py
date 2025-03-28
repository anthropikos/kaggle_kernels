# Use ray tune to tune hyperparameters: n_hidden, depth, kernel
import logging
from ray.train.lightning import (
    RayDDPStrategy, 
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
import pytorch_lightning as pl
from ray import tune, train
from ray.tune import ResultGrid
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
from ray.tune.schedulers import ASHAScheduler
from ..model.cnn_1d_lightning import CNN1d_Lightning
from ..data.lfp_data_lightning import EssentialTremorLFPDataset_Posture_Lightning

logger = logging.getLogger(__name__)

def train_loop_per_worker(config, model=None, datamodule=None) -> None:
    
    if model is None: 
        model = CNN1d_Lightning()
    if datamodule is None: 
        datamodule = EssentialTremorLFPDataset_Posture_Lightning()
    
    lightning_trainer = pl.Trainer(
        devices='auto', 
        accelerator='auto',
        strategy=RayDDPStrategy(),
        callbacks=[
            RayTrainReportCallback(),
        ],
        plugins=[
            RayLightningEnvironment(),    
        ],
        enable_progress_bar=False,
    )
    
    lightning_trainer = prepare_trainer(lightning_trainer)  # Prepare for distributed execution - https://docs.ray.io/en/latest/train/api/doc/ray.train.lightning.prepare_trainer.html
    lightning_trainer.fit(model, datamodule)

    return
    

# TODO: (Later) Try to implement distributed tuning process such that I can tune in embarrasingly parallel way at least
def tune_cnn_1d() -> ResultGrid:
    
    # Parameter for scheduler
    scheduler_max_num_epoch = 3 # Would like to use 5 but that may result in too long of a tuning process

    # Parameter for tuner
    num_samples_from_search_space = 10 # Can be set to -1 for infinite samples until stop condition for the Tuner is met

    
    # Create the Ray Trainer
    scaling_config = ScalingConfig(
        num_workers=1, 
        use_gpu=True, 
        resources_per_worker={'CPU': 4, 'GPU': 1},
    )
    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=3,
            checkpoint_score_attribute='val_loss', 
            checkpoint_score_order='min',
        )
    )

    ray_trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        scaling_config=scaling_config, 
        run_config=run_config,
    )
    
    # Parameters for the Ray Tuner
    search_space = {
        'n_hidden':          tune.qrandint(5, 100, q=2),
        'depth':             tune.qrandint(2, 20, q=2),
        'kernel':            tune.qlograndint(10, 200, q=2, base=2),
        'lr':                tune.loguniform(1e-5, 1e-1),
        'weight_decay':      tune.loguniform(1e-1, 1e-15),
    }
    scheduler = ASHAScheduler(
        time_attr='training_iteration', 
        max_t=scheduler_max_num_epoch,
        grace_period=1,
        reduction_factor=2,
    )
    tune_config = tune.TuneConfig(
        metric='val_loss_epoch',  # This metric is auto created by the log method in validation_step when on_epoch param is True
        mode='min',
        num_samples=num_samples_from_search_space,
        scheduler=scheduler,
    )
    
    tuner = tune.Tuner(
        trainable=ray_trainer, 
        param_space={'train_loop_config': search_space}, # Passed as param to the trainable
        tune_config=tune_config,
    )
    
    result_grid = tuner.fit()
    
    return result_grid