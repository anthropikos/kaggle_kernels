# 2025-03-27 Anthony Lee

from .cnn_1d import CNN1d
from torch import nn, optim
import pytorch_lightning as pl


class CNN1d_Lightning(pl.LightningModule):
    def __init__(self, config=None):
        super().__init__()
        
        if config is None: config = {}

        self.model = CNN1d(config=config)
        self.loss_module = nn.BCELoss(reduction="mean")
        self.config = config

        return

    def forward(self, x):
        output = self.model(x)

        return output

    def training_step(self, batch, batch_idx):
        """Training step hook.
        
        The training step hook is called in the `fit_loop()` [^1] calls the
        `traininig_step` on each batch of data loaded from the train_dataloader.
        
        The log function `on_epoch` allows the logger to log accumulated train_loss
        and reduce it with `reduce_fx`. In the case below, the train_loss for
        the epoch is accumulated and the average is sent to the logger.
        
        References:
        [^1]: [Pseudocode describing the structure of `fit()` invocation](https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#hooks)
        """

        input, target = batch
        output = self(input)
        loss = self.loss_module(output, target)

        # Log metrics to logger
        self.log(
            "train_loss", 
            loss, 
            on_step=True,
            on_epoch=True,     # Logs epoch accumulated metrics, reduced by `reduce_fx`
            reduce_fx='mean',  # Reduction function over step values for end of epoch.
        )

        return loss
    
    def validation_step(self, batch, batch_idx):
        """Hook for the validation loop.
        
        This hook is run by the validation loop and the loop's frequency is
        determined by the `check_val_every_n_epoch` parameter of the
        `ray.train.Trainer` object, which defaults to 1 [^1].
        
        Thus, by default, the validation loop is only ran once at the end of an
        epoch.
        
        Reference:
        [^1]: [Lightning documentation for `ray.train.Trainer`](https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api)
        """
        input, target = batch
        output = self.model(input)
        loss = self.loss_module(output, target)

        # Log metrics to logger
        #################### 
        # NOTE: 
        # on_epoch vs on_step: 
        # The behavior of how `on_step` when set to True changes the self.global_step as stated in this forum discussion 
        # is not well documented. It seems that `val_loss_step`'s `global_step` does not track the `self.global_step`,
        # but `val_loss_epoch`'s `global_step` tracks the `self.global_step`.
        #
        # I think this behavior makes sense in that when setting `on_step` to True for the logger, it logs the results
        # of each batch from the `val_dataloader()` when running the `validation_step` hook at the end of an epoch, hence
        # the step is essentially the last global step, and thus doesn't make sense to track the global_step.
        #
        # However, on the other hand, `val_loss_epoch` reduces all the step metrics from the `validation_step` hook thus 
        # producing a single value instead of the number of values equaling the length of `val_dataloader`, thus it is 
        # okay to use the global_step, and would be in alignment with what validation loss is, which is the loss at the
        # end of each epoch.
        #
        # I think the way this works is that the metric `val_loss_epoch` is passed up the hierarchy of loggers so that
        # the global_step can be used to record this validation loss value.
        #
        # https://lightning.ai/forums/t/understanding-logging-and-validation-step-validation-epoch-end/291
        ####################
        
        
        self.log(
            # In validation loss, the steps are still calculated, just that the 
            # steps are val_dataloader steps intead of steps from the train_dataloader.
            
            "val_loss", 
            loss,
            on_step=True,        # Log each step of the val_dataloader
            on_epoch=True,       # True by default for `validation_step` - https://lightning.ai/forums/t/understanding-logging-and-validation-step-validation-epoch-end/291/2
            reduce_fx='mean',    # Method param default - Explicitly stated for clarity
        )

        return loss
    # TODO: (Later) Understand how the on_validation_epoch_end param `sync_dist` work to reduce metric across devices. - https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#log

    def test_step(self, batch, batch_idx):
        """Test step for the testing loop.
        
        The test loop is similar to the validation loop, however, the difference
        is that the test loop is only called when the `test()` method of
        `ray.tune.Tuner` object is called [^1].
        
        The test loop is NOT used during training, but ONLY used once the model
        has been trained to see how the model will do in the real world [^2].
        
        References:
        [^1]: [Lightning documentation](https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#testing)
        [^2]: [Lightning documentation on the test loop](https://lightning.ai/docs/pytorch/stable/common/evaluation_basic.html)
        """
        input, target = batch
        output = self.model(input)
        loss = self.loss_module(output, target)
        
        # Lots of logs for testing purpose
        self.log(
            "test_loss", 
            loss, 
            on_step=True, 
            on_epoch=True, 
            reduce_fx='mean', 
        )

        return loss

    def predict_step(self, batch):
        input, target = batch
        output = self.model(input)
        
        return output
    
    def configure_optimizers(self):

        lr = self.config.get('lr', 1e-3)
        weight_decay = self.config.get('weight_decay', 1e-8)

        optimizer = optim.AdamW(
            params=self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        
        return optimizer