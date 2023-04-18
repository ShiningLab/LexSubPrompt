#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# public
import torch
import numpy as np
import lightning.pytorch as pl
from transformers import AutoModelForPreTraining


class LM(pl.LightningModule):
    """docstring for LM"""
    def __init__(self, config, **kwargs):
        super(LM, self).__init__()
        self.config = config
        self.update_config(**kwargs)
        self.model = AutoModelForPreTraining.from_pretrained(config.LM_PATH)
        self.train_loss_list = []
        self.save_hyperparameters()

    def update_config(self, **kwargs):
        # update configuration accordingly
        for k,v in kwargs.items():
            setattr(self.config, k, v)

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch, labels=batch)
        loss = outputs.loss.item()
        self.train_loss_list.append(loss)
        ys_ = outputs.logits.softmax(dim=1).argmax(dim=1)
        self.log('train_step_loss', loss, prog_bar=True)
        return outputs.loss

    def on_train_epoch_end(self):
        self.log_dict({
            'train_epoch_loss': np.mean(self.train_loss_list, dtype='float32')
            })
        self.train_loss_list = []

    def validation_step(self, batch, batch_idx):
        outputs = self.model(batch, labels=batch)
        self.log('val_loss', outputs.loss)

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)]
                , "weight_decay": self.config.weight_decay
                }
            , {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)]
                , "weight_decay": 0.0
                }
            ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters
            , lr=self.config.learning_rate
            , eps=self.config.adam_epsilon
            )
        return optimizer