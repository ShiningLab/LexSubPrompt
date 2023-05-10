#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# public
import torch
import numpy as np
import lightning.pytorch as pl
from transformers import AutoTokenizer
# private
from src import helper
from src.eval import overall_precision_at_k


class LM(pl.LightningModule):
    """docstring for LM"""
    def __init__(self, config, **kwargs):
        super(LM, self).__init__()
        self.config = config
        self.update_config(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(config.LM_PATH)
        self.model = helper.get_model(config)
        self.train_loss, self.val_loss = [], []
        self.val_xs, self.val_ys, self.val_subs, self.val_ys_, self.val_subs_= [], [], [], [], []
        self.save_hyperparameters()

    def update_config(self, **kwargs):
        # update configuration accordingly
        for k,v in kwargs.items():
            setattr(self.config, k, v)

    # def postprocess(self, logits):
    #     ys_ = logits.softmax(dim=-1).argmax(dim=-1)
    #     return self.tokenizer.batch_decode(ys_, skip_special_tokens=True)

    def lm2lexsub(self, sents):
        subs = []
        for s in sents:
            try:
                sub = s.split('"')[-2]
            except:
                sub = ''
            subs.append(sub)
        return subs

    def training_step(self, batch, batch_idx):
        # raw_xs, raw_ys, subs, xs, ys, labels
        _, _, _, _, ys, labels = batch
        loss = self.model(**ys, labels=labels).loss
        self.train_loss.append(loss.item())
        self.log('train_step_loss', loss.item(), prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        loss = np.mean(self.train_loss, dtype='float32')
        self.log('train_epoch_loss', loss)
        self.train_loss = []

    def validation_step(self, batch, batch_idx):
        # raw_xs, raw_ys, subs, xs, ys, labels
        raw_xs, raw_ys, subs, xs, ys, labels = batch
        # loss
        loss = self.model(**ys, labels=labels).loss.item()
        self.val_loss.append(loss)
        # generate
        ys_ = self.model.generate(
            **xs
            , max_new_tokens=self.config.max_new_tokens
            , pad_token_id=self.tokenizer.eos_token_id
            , num_beams=1
            , num_beam_groups=1
            , early_stopping=True
            , num_return_sequences=1
            )
        ys_ = self.tokenizer.batch_decode(ys_, skip_special_tokens=True)
        self.val_xs += raw_xs
        self.val_ys += raw_ys
        subs_ = self.lm2lexsub(ys_)
        self.val_subs += subs  # list[list[str]]
        self.val_subs_ += [[s] for s in subs_]  # list[str] -> list[list[str]]
        
    def on_validation_epoch_end(self):
        loss = np.mean(self.val_loss, dtype='float32')
        p1 = overall_precision_at_k(self.val_subs, self.val_subs_, 1)
        self.log_dict({'val_epoch_loss': loss, 'val_p1': p1})
        columns = ['xs', 'ys', 'subs', 'ys_', 'subs_']
        data = list(map(list, zip(self.val_xs, self.val_ys, self.val_subs, self.val_ys_, self.val_subs_)))
        self.logger.log_text(key='val', columns=columns, data=data)
        self.val_loss, self.val_xs, self.val_ys, self.val_subs, self.val_ys_, self.val_subs_= [], [], [], [], [], []

    # def predict_step(self, batch, batch_idx):
    #     # raw_xs, raw_ys, xs, ys, labels
    #     raw_xs, raw_ys, xs, ys, _ = batch
    #     bad_words_ids = [self.tokenizer(x.split('"')[1], add_special_tokens=False).input_ids for x in raw_xs]
    #     ys_ = self.model.generate(
    #         **xs
    #         , max_new_tokens=8
    #         # , num_beams=15
    #         # , early_stopping=True
    #         # , num_return_sequences=10
    #         , pad_token_id=self.tokenizer.eos_token_id
    #         , bad_words_ids = bad_words_ids
    #         )
    #     ys_ = self.tokenizer.batch_decode(ys_, skip_special_tokens=True)
    #     subs = self.lm2lexsub(raw_ys)
    #     subs_ = self.lm2lexsub(ys_)
    #     return {'raw_xs': raw_xs, 'raw_ys': raw_ys, 'ys_': ys_, 'subs': subs, 'subs_': subs_}

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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer
            , T_0=self.config.warmup_epoch
            )
        return [optimizer], [scheduler]