import abc

import pytorch_lightning as pl


class Experiment(abc.ABC):
    def train():
        raise NotImplementedError


class LightningExperiment(Experiment):
    trainer: pl.Trainer
    model: pl.LightningModel

    def __init__(self, model: pl.LightningModel):
        self.model = model

        self.trainer = pl.Trainer()

    def fit(self):

        self.trainer.fit(
            self.model,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.val_dataloader,
        )
