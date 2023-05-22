#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# built-in
import random, itertools
# public
from tqdm import trange
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
        # prompt templates for positive instances
        self.pos_src_prompt = 'The {} "{}" at position {} in sentence "{}" can be substituted with "'
        self.pos_tgt_prompt = 'The {} "{}" at position {} in sentence "{}" can be substituted with "{}".'
        # prompt templates for negative instances
        self.neg_src_prompt = 'The {} "{}" at position {} in sentence "{}" can not be substituted with "'
        self.neg_tgt_prompt = 'The {} "{}" at position {} in sentence "{}" can not be substituted with "{}".'
        self.get_data()
        self.get_tokenizer()
        if self.config.data_mode == 'contrast':
            self.get_tgt2subs()
            
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
        self.train_tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_PATH)
        # https://github.com/huggingface/transformers/issues/2630
        # https://huggingface.co/patrickvonplaten/bert2gpt2-cnn_dailymail-fp16#training-script
        self.train_tokenizer.pad_token = self.train_tokenizer.unk_token
        # eval
        self.eval_tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_PATH)
        self.eval_tokenizer.padding_side = 'left'
        self.eval_tokenizer.pad_token = self.eval_tokenizer.unk_token

    def get_tgt2subs(self):
        self.tgt2subs_dict = {}
        for i in range(self.data_size):
            tgt, _, _, _, subs, _ = (d[i] for d in self.data_tuple)
            pool = [tgt] + subs
            for t in pool:
                s = pool.copy()
                s.remove(t)
                if t in self.tgt2subs_dict:
                    self.tgt2subs_dict[t].update(s)
                else:
                    self.tgt2subs_dict[t] = set(s)
        self.tgt2subs_dict = {k: list(v) for k, v in self.tgt2subs_dict.items()}

    def get_instance(self, idx):
        tgt, pos, p, ctx, pos_subs, w = (d[idx] for d in self.data_tuple)
        if self.mode == 'train':
            pos_sub = random.choices(pos_subs, weights=w, k=1)[0]
            switch = random.randint(0, 1)
            if switch:
                # replace target with sampled positive subtitute
                tgt, pos_sub = pos_sub, tgt
                # update context
                ctx = ctx.split()
                ctx[p] = tgt
                ctx = ' '.join(ctx)
                # update positive subtitutes
                pos_subs.append(pos_sub)
                pos_subs.remove(tgt)
        else:
            pos_sub = pos_subs[0]
        pos_src_text = self.pos_src_prompt.format(pos, tgt, p, ctx)
        pos_tgt_text = self.pos_tgt_prompt.format(pos, tgt, p, ctx, pos_sub)
        # training
        if self.mode == 'train':
            # enable contrast prompting
            if self.config.data_mode == 'contrast':
                neg_subs = [s for s in self.tgt2subs_dict[tgt] if s not in pos_subs]
                if neg_subs:
                    # TODO: negative w
                    neg_sub = random.choice(neg_subs)
                    neg_src_text = self.neg_src_prompt.format(pos, tgt, p, ctx)
                    neg_tgt_text = self.neg_tgt_prompt.format(pos, tgt, p, ctx, neg_sub)
                else:
                    neg_src_text, neg_tgt_text = '', ''
                return pos_src_text, pos_tgt_text, neg_src_text, neg_tgt_text
            return pos_src_text, pos_tgt_text
        # validation
        elif self.mode == 'val':
            return pos_src_text, pos_tgt_text, pos_subs
        # testing and prediction
        else:
            return pos_src_text, pos_tgt_text, tgt, pos, p, ctx, pos_subs

    def collate_fn(self, batch):
        batch = list(map(list, zip(*batch)))
        match len(batch):
            case 2:
                raw_xs, raw_ys = batch
            case 3:
                raw_xs, raw_ys, subs = batch
            case 4:
                raw_pos_xs, raw_pos_ys, raw_neg_xs, raw_neg_ys = batch
                raw_xs = raw_pos_xs + [ x for x in raw_neg_xs if x]
                raw_ys = raw_pos_ys + [ y for y in raw_neg_ys if y]
            case 7:
                raw_xs, raw_ys, tgts, poss, ps, ctxs, subs = batch
            case _:
                raise NotImplementedError
        # tokenization
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
        # return
        match self.mode:
            case 'train':
                return ys, labels
            case 'val':
                return ys, labels, xs, subs
            case 'test':
                return raw_xs, xs, tgts, poss, ps, ctxs, subs
            case _:
                raise NotImplementedError

    def __getitem__(self, idx):
        return self.get_instance(idx)