#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# public
import lightning.pytorch as pl
from torch.utils.data import DataLoader
# private
from src.datasets import LSPDataset


class DataModule(pl.LightningDataModule):
    """docstring for DataModule"""
    def __init__(self, config):
        super(DataModule, self).__init__()
        self.config = config

    def setup(self, stage: str):
        match stage:
            case 'fit':
                self.train_dataset = LSPDataset('train', self.config)
                self.val_dataset = LSPDataset('val', self.config)
                self.config.train_size = len(self.train_dataset)
                self.config.val_size = len(self.val_dataset)
            case 'validate':
                self.val_dataset = LSPDataset('val', self.config)
                self.config.val_size = len(self.val_dataset)
            case 'test':
                self.test_dataset = LSPDataset('test', self.config)
                self.config.test_size = len(self.test_dataset)
            case 'predict':
                self.predict_dataset = LSPDataset('test', self.config)
                self.config.predict_size = len(self.predict_dataset)
            case _:
                raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset
            , batch_size=self.config.train_batch_size
            , collate_fn=self.train_dataset.collate_fn
            , shuffle=True
            , num_workers=self.config.num_workers
            , pin_memory=True
            , drop_last=True
            )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset
            , batch_size=self.config.train_batch_size
            , collate_fn=self.val_dataset.collate_fn
            , shuffle=False
            , num_workers=self.config.num_workers
            , pin_memory=True
            , drop_last=False
            )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset
            , batch_size=self.config.eval_batch_size
            , collate_fn=self.test_dataset.collate_fn
            , shuffle=False
            , num_workers=self.config.num_workers
            , pin_memory=True
            , drop_last=False
            )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset
            , batch_size=self.config.eval_batch_size
            , collate_fn=self.predict_dataset.collate_fn
            , shuffle=False
            , num_workers=self.config.num_workers
            , pin_memory=True
            , drop_last=False
            )