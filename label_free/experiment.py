import abc

import pytorch_lightning as pl
import torch.utils.data


class Experiment(abc.ABC):
    def train():
        raise NotImplementedError


class LightningExperiment(Experiment):
    trainer: pl.Trainer
    model: pl.LightningModel

    def __init__(
        self,
        model: pl.LightningModel,
        train_dataloader: torch.utils.daPta.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader = None,
    ):
        self.model = model

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.trainer = pl.Trainer()

        
    def fit(self):

        self.trainer.fit(
            self.model,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.val_dataloader,
        )
