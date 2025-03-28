# Use ray tune to tune hyperparameters: n_hidden, depth, kernel

from ray.train.lightning import (
    RayDDPStrategy, 
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)

import pytorch_lightning as pl

def train_loop_per_worker(config, model, datamodule):
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
    
    
    
from ray import tune, train
# from ray.tune.schedulers import ASHAScheduler
# from ray.train import RunConfig, ScalingConfig, CheckpointConfig
# from ray.train.torch import TorchTrainer



num_epochs = 5
num_samples = 10
scheduler = ASHAScheduler(
    time_attr='training_iteration', 
    max_t=num_epochs,
    grace_period=1,
    reduction_factor=2,
)




ray_trainer = TorchTrainer(
    train_func, 
    scaling_config=scaling_config, 
    run_config=run_config,
)

from ray.train.torch import TorchTrainer

def tune_cnn_1d():
    
    # Parameters for the ray trainer
    scaling_config = train.ScalingConfig(
        num_workers=1, 
        use_gpu=True, 
        resources_per_worker={'CPU': 4, 'GPU': 1},
    )
    run_config = train.RunConfig(
        checkpoint_config=train.CheckpointConfig(
            num_to_keep=3,
            checkpoint_score_attribute='val_loss', 
            checkpoint_score_order='min',
        )
    )

    # Create the ray trainer
    ray_trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        scaling_config=scaling_config, 
        run_config=run_config,
    )
    
    # Parameters for the RayTune Tuner
    search_space = {
        'n_hidden': tune.uniform(5, 100),
        'depth': tune.uniform(2, 20),
        'kernel': tune.quniform(10, 200),
        'lr': tune.loguniform(1e-5, 1e-1),
        'weight_decay': tune.loguniform(1e-1, 1e-15),
    }
    tune_config = tune.TuneConfig(
        metric='val_loss'
    )
    
    tuner = tune.Tuner(
        trainable=ray_trainer, 
        param_space={'train_loop_config': search_space}, # Passed as param to the trainable
        tune_config=tune_config,
    )
    
    result_grid = tuner.fit()
    
    return result_grid