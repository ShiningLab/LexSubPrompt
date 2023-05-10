#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# built-in
import random
# public
from transformers import AutoTokenizer
# private
from src import helper

class LSPDataset(object):
    """docstring for LSPDataset"""
    def __init__(self, mode, config, sample_size=None):
        super(LSPDataset, self).__init__()
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.config = config
        self.sample_size = sample_size
        self.src_prompt = 'The {} "{}" at position {} in sentence "{}" can be substituted with "'
        self.tgt_prompt = 'The {} "{}" at position {} in sentence "{}" can be substituted with "{}".'
        self.get_data()
        self.get_tokenizer()

    def __len__(self): 
        return self.data_size

    def get_data(self):
        data_dict = helper.load_pickle(self.config.DATA_PKL)
        self.data_tuple = data_dict[self.mode]
        self.data_size = len(self.data_tuple[0])
        if self.sample_size:
            self.data_size = self.sample_size

    def get_tokenizer(self):
        # train
        self.train_tokenizer = AutoTokenizer.from_pretrained(self.config.LM_PATH)
        # https://github.com/huggingface/transformers/issues/2630
        # https://huggingface.co/patrickvonplaten/bert2gpt2-cnn_dailymail-fp16#training-script
        self.train_tokenizer.pad_token = self.train_tokenizer.unk_token
        # eval
        self.eval_tokenizer = AutoTokenizer.from_pretrained(self.config.LM_PATH)
        self.eval_tokenizer.padding_side = 'left'
        self.eval_tokenizer.pad_token = self.eval_tokenizer.unk_token

    def get_instance(self, idx):
        tgt, pos, p, ctx, subs, w = (d[idx] for d in self.data_tuple)
        if self.mode == 'train':
            sub = random.choices(subs, weights=w, k=1)[0]
            switch = random.randint(0, 1)
            if switch:
                tgt, sub = sub, tgt
            ctx = ctx.split()
            ctx[p] = tgt
            ctx = ' '.join(ctx)
        else:
            sub = subs[0]
        src_text = self.src_prompt.format(pos, tgt, p, ctx)
        tgt_text = self.tgt_prompt.format(pos, tgt, p, ctx, sub)
        return src_text, tgt_text, subs

    def collate_fn(self, batch):
        raw_xs, raw_ys, subs = map(list, zip(*batch))
        xs = self.eval_tokenizer.batch_encode_plus(
            raw_xs
            , add_special_tokens=True
            , return_tensors='pt'
            , padding=True
            , truncation=True
            , max_length=self.config.max_length
            )
        ys = self.train_tokenizer.batch_encode_plus(
            raw_ys
            , add_special_tokens=True
            , return_tensors='pt'
            , padding=True
            , truncation=True
            , max_length=self.config.max_length
            )
        # create label tensor
        labels = ys.input_ids.clone()
        for i, (x_m, y_m) in enumerate(zip(xs.attention_mask, ys.attention_mask)):
            labels[i, :x_m.sum()] = -100  # mask the prompt
            labels[i, y_m.sum()+1:] = -100  # mask the padding
        return raw_xs, raw_ys, subs, xs, ys, labels

    def __getitem__(self, idx):
        return self.get_instance(idx)