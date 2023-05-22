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


class LSP(pl.LightningModule):
    """docstring for LSP"""
    def __init__(self, config, **kwargs):
        super(LSP, self).__init__()
        self.config = config
        self.update_config(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
        self.model = helper.get_model(config)
        self.train_loss, self.val_loss = [], []
        self.val_subs, self.val_subs_= [], []
        self.test_subs, self.test_subs_= [], []
        self.save_hyperparameters()

    def update_config(self, **kwargs):
        # update configuration accordingly
        for k,v in kwargs.items():
            setattr(self.config, k, v)

    def generate(self, xs, num_beams, num_return_sequences, raw_xs=None):
        if raw_xs:
            bad_words_ids = [self.tokenizer(raw_xs[0].split('"')[1], add_special_tokens=False).input_ids]
        else:
            bad_words_ids = None
        ys_ = self.model.generate(
            **xs
            , max_new_tokens=self.config.max_new_tokens
            , num_beams=num_beams
            , num_beam_groups=1
            , early_stopping=True
            , num_return_sequences=num_return_sequences
            , pad_token_id=self.tokenizer.eos_token_id
            , bad_words_ids = bad_words_ids if bad_words_ids else None
        )
        ys_ = self.tokenizer.batch_decode(ys_, skip_special_tokens=True)
        return ys_

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
        ys, labels = batch
        loss = self.model(**ys, labels=labels).loss
        self.train_loss.append(loss.item())
        self.log('train_step_loss', loss.item(), prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        loss = np.mean(self.train_loss, dtype='float32')
        self.log('train_epoch_loss', loss)
        self.train_loss = []

    def validation_step(self, batch, batch_idx):
        ys, labels, xs, subs = batch
        # loss
        loss = self.model(**ys, labels=labels).loss.item()
        self.val_loss.append(loss)
        # generate
        ys_ = self.generate(xs, num_beams=1, num_return_sequences=1)
        subs_ = self.lm2lexsub(ys_)
        self.val_subs += subs  # list[list[str]]
        self.val_subs_ += [[s] for s in subs_]  # list[str] -> list[list[str]]
        
    def on_validation_epoch_end(self):
        loss = np.mean(self.val_loss, dtype='float32')
        p1 = overall_precision_at_k(self.val_subs, self.val_subs_, 1)
        self.log_dict({'val_epoch_loss': loss, 'val_p1': p1})
        self.val_loss, self.val_subs, self.val_subs_= [], [], []

    def predict_step(self, batch, batch_idx):
        raw_xs, xs, tgts, poss, ps, ctxs, subs = batch
        ys_ = self.generate(
            xs
            , raw_xs = raw_xs
            , num_beams=self.config.num_beams
            , num_return_sequences=self.config.num_return_sequences
            )
        subs_ = self.lm2lexsub(ys_)
        return {
        'target': tgts[0]
        , 'POS': poss[0]
        , 'position': ps[0]
        , 'context': ctxs[0]
        , 'input': raw_xs[0]
        , 'subs': subs[0]
        , 'subs_': subs_
        }

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