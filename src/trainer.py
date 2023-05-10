#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# public
import lightning.pytorch as pl
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
# private
from src import helper
from src.models import LM
from src.datamodule import DataModule


class LitTrainer(object):
    """docstring for LitTrainer"""
    def __init__(self, config, **kwargs):
        super(LitTrainer, self).__init__()
        self.config = config
        self.update_config(**kwargs)
        self.initialize()

    def update_config(self, **kwargs):
        # update configuration accordingly
        for k,v in kwargs.items():
            setattr(self.config, k, v)

    def initialize(self):
        # model
        self.model = LM(self.config)
        # datamodule
        self.dm = DataModule(self.config)
        # callbacks
        filename = '{epoch}-{step}-{val_p1:.4f}'
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config.CKPT_PATH
            , filename=filename
            , monitor='val_p1'
            , mode='max'
            , verbose=True
            , save_last=True
            , save_top_k=1
            )
        early_stop_callback = EarlyStopping(
            monitor='val_p1'
            , min_delta=.0
            , patience=self.config.patience
            , verbose=True
            , mode='max'
            )
        # logger
        self.logger = helper.init_logger(self.config)
        self.logger.info('Logger initialized.')
        self.wandb_logger = WandbLogger(
            name=self.config.NAME
            , save_dir=self.config.LOG_PATH
            , offline=self.config.offline
            , project=self.config.PROJECT
            , log_model=self.config.log_model
            , entity=self.config.ENTITY
            , save_code=False
            , mode=self.config.wandb_mode
            )
        self.wandb_logger.experiment.config.update(self.config)
        # trainer
        self.trainer = pl.Trainer(
            logger = self.wandb_logger
            , callbacks=[checkpoint_callback, early_stop_callback, RichProgressBar()]
            , max_epochs=self.config.max_epochs
            , val_check_interval=self.config.val_check_interval
            , enable_checkpointing=True
            , enable_progress_bar=True
            , gradient_clip_val=self.config.gradient_clip_val
            , deterministic=True
            , inference_mode=True
            , profiler=self.config.profiler if self.config.profiler else None
            )

    def train(self):
        self.logger.info('*Configurations:*')
        for k, v in self.config.__dict__.items():
            self.logger.info(f'\t{k}: {v}')
        # training
        self.logger.info("Start training...")
        self.trainer.fit(
            model=self.model
            , datamodule=self.dm
            , ckpt_path= 'last' if self.config.load_ckpt else None
            )
        self.logger.info('Done.')

    def validate(self, ckpt_path=None):
        self.trainer.validate(
            model=self.model
            , datamodule=self.dm
            , ckpt_path=ckpt_path
            , verbose=True
            )

    # def test(self, ckpt_path=None):
    #     self.trainer.test(
    #         model=self.model
    #         , datamodule=self.dm
    #         , ckpt_path=ckpt_path
    #         , verbose=True
    #         )