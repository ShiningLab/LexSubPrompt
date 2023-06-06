#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# built-in
import random, itertools
# public
from transformers import AutoTokenizer
from torch.utils import data as torch_data
# private
from src import helper


class LMDataset(torch_data.Dataset):
    """docstring for LMDataset"""
    def __init__(self, mode, config, sample_size=None):
        super(LMDataset, self).__init__()
        assert mode in ['train', 'val']
        self.mode = mode
        self.config = config
        self.sample_size = sample_size
        self.get_tokenizer()
        self.get_data()

    def __len__(self): 
        return self.data_size

    def get_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_PATH)
        # https://github.com/huggingface/transformers/issues/2630
        # https://huggingface.co/patrickvonplaten/bert2gpt2-cnn_dailymail-fp16#training-script
        self.tokenizer.pad_token = self.tokenizer.unk_token

    def get_data(self):
        data_dict = helper.load_pickle(self.config.DATA_PKL)
        self.data_tuple = data_dict[self.mode]
        self.data_size = len(self.data_tuple[0])
        if self.sample_size:
            self.data_size = self.sample_size

    def get_instance(self, idx):
        instance = (d[idx] for d in self.data_tuple)
        # sample idx, target word, part-of-speech, position
        # context, vocab, substitutes, weights
        _, tgt, _, p, ctx, _, subs, _ = instance
        # random sampling
        sub = random.choice([tgt] + subs)
        # put in context
        ctx = ctx.split()
        ctx[p] = sub
        return ' '.join(ctx)

    def __getitem__(self, idx):
        return self.get_instance(idx)

    def collate_fn(self, data):
        # a customized collate function used in the data loader
        inputs = self.tokenizer.batch_encode_plus(
            data
            , add_special_tokens=True
            , return_tensors='pt'
            , padding=True
            , truncation=True
            , max_length=self.config.max_length
            )
        return inputs


class LSPDataset(torch_data.Dataset):
    """docstring for LSPDataset"""
    def __init__(self, mode, config, sample_size=None):
        super(LSPDataset, self).__init__()
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.config = config
        self.sample_size = sample_size
        self.get_data()
        self.get_tokenizer()
        if self.sample_size:
            self.data_size = self.sample_size
        self.get_prompt()
            
    def __len__(self): 
        return self.data_size

    def get_data(self):
        data_dict = helper.load_pickle(self.config.DATA_PKL)
        self.data_tuple = data_dict[self.mode]
        self.data_size = len(self.data_tuple[0])

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

    def get_prompt(self):
        # prompt templates with placeholders to fill in
        match self.config.prompt_mode:
            case 'base':
                self.src_prompt = 'The "{}" in the sentence "{}" can be substituted with "'
                self.tgt_prompt = 'The "{}" in the sentence "{}" can be substituted with "{}".'
            case 'full':
                self.src_prompt = 'At position {} in the sentence, "{}", the {} "{}", derived from the lemma "{}", can be substituted with "'
                self.tgt_prompt = 'At position {} in the sentence, "{}", the {} "{}", derived from the lemma "{}", can be substituted with "{}".'
            case 'best':
                self.src_prompt = 'At position {} in the sentence, "{}", the {} "{}", derived from the lemma "{}", can be substituted with '
                self.src_best_prompt = 'At position {} in the sentence, "{}", the {} "{}", derived from the lemma "{}", can be best substituted with "'
                self.tgt_best_prompt = 'At position {} in the sentence, "{}", the {} "{}", derived from the lemma "{}", can be best substituted with "{}".'
            case _:
                raise NotImplementedError

    def train_sample(self, idx):
        instance = (d[idx] for d in self.data_tuple)
        # sample idx, target word, part-of-speech, position
        # context, vocab, substitutes, weights
        _, tgt, pos, p, ctx, _, subs, ws = instance
        match self.config.data_mode:
            case 'base':  # the first
                sub = subs[0]
            case 'sample':  # sample
                sub = random.choice(subs)
            case 'wsample':  # weighted sample
                sub = random.choices(subs, weights=ws, k=1)[0]
            case 'switch':  # switch
                pool = [tgt] + subs
                t, sub = random.sample(pool, 2)
                if t != tgt:
                    tgt = t
                    # update context
                    ctx = ctx.split()
                    ctx[p] = tgt
                    ctx = ' '.join(ctx)
            case 'wswitch':  # weighted switch
                sub = random.choices(subs, weights=ws, k=1)[0]
                if random.randint(0, 1):
                    # replace target with sampled positive subtitute
                    tgt, sub = sub, tgt
                    # update context
                    ctx = ctx.split()
                    ctx[p] = tgt
                    ctx = ' '.join(ctx)
            case _:
                raise NotImplementedError
        match self.config.prompt_mode:
            case 'base':
                src_text = self.src_prompt.format(ctx.split()[p], ctx)
                tgt_text = self.tgt_prompt.format(ctx.split()[p], ctx, sub)
            case 'full':
                src_text = self.src_prompt.format(p, ctx, pos, ctx.split()[p], tgt)
                tgt_text = self.tgt_prompt.format(p, ctx, pos, ctx.split()[p], tgt, sub)
            case 'best':
                subs.append(sub)
                subs = [s for s in subs if s != tgt and s != ctx.split()[p]]
                if random.randint(0, 1) and len(subs) > 1:
                    src_text = self.src_prompt.format(p, ctx, pos, ctx.split()[p], tgt)
                    tgt_prompt = ''
                    for s in subs[:-1]:
                        tgt_prompt += f'"{s}", '
                    tgt_prompt += f'"{subs[-1]}".'
                    tgt_text = src_text + tgt_prompt
                else:
                    src_text = self.src_best_prompt.format(p, ctx, pos, ctx.split()[p], tgt)
                    tgt_text = self.tgt_best_prompt.format(p, ctx, pos, ctx.split()[p], tgt, sub)
            case _:
                raise NotImplementedError
        return src_text, tgt_text

    def val_sample(self, idx):
        instance = (d[idx] for d in self.data_tuple)
        # sample idx, target word, part-of-speech, position
        # context, vocab, substitutes, weights
        _, tgt, pos, p, ctx, _, subs, _ = instance
        match self.config.prompt_mode:
            case 'base':
                src_text = self.src_prompt.format(ctx.split()[p], ctx)
                tgt_text = self.tgt_prompt.format(ctx.split()[p], ctx, subs[0])
            case 'full':
                src_text = self.src_prompt.format(p, ctx, pos, ctx.split()[p], tgt)
                tgt_text = self.tgt_prompt.format(p, ctx, pos, ctx.split()[p], tgt, subs[0])
            case 'best':
                src_text = self.src_best_prompt.format(p, ctx, pos, ctx.split()[p], tgt)
                tgt_text = self.tgt_best_prompt.format(p, ctx, pos, ctx.split()[p], tgt, subs[0])
            case _:
                raise NotImplementedError
        return src_text, tgt_text, subs

    def test_sample(self, idx):
        instance = (d[idx] for d in self.data_tuple)
        # sample idx, target word, part-of-speech, position
        # context, vocab, substitutes, weights
        i, tgt, pos, p, ctx, vocab, subs, _ = instance
        match self.config.prompt_mode:
            case 'base':
                src_text = self.src_prompt.format(ctx.split()[p], ctx)
            case 'full':
                src_text = self.src_prompt.format(p, ctx, pos, ctx.split()[p], tgt)
            case 'best':
                src_text = self.src_best_prompt.format(p, ctx, pos, ctx.split()[p], tgt)
            case _:
                raise NotImplementedError
        return src_text, i, tgt, pos, p, ctx, vocab, subs

    def get_instance(self, idx):
        match self.mode:
            case 'train':
                return self.train_sample(idx)
            case 'val':
                return self.val_sample(idx)
            case 'test':
                return self.test_sample(idx)
            case _:
                raise NotImplementedError

    def collate_fn(self, batch):
        batch = list(map(list, zip(*batch)))
        match len(batch):
            case 2:
                raw_xs, raw_ys = batch
            case 3:
                raw_xs, raw_ys, subs = batch
            case 8:
                raw_xs, idx, tgt, pos, p, ctx, vocab, subs = batch
            case _:
                raise NotImplementedError
        # encode source texts
        xs = self.eval_tokenizer.batch_encode_plus(
            raw_xs
            , add_special_tokens=True
            , return_tensors='pt'
            , padding=True
            , truncation=True
            , max_length=self.config.max_length
            )
        if self.mode in ['train', 'val']:
            # encode target texts
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
                # mask the prompt
                labels[i, :x_m.sum()] = -100
                # mask the padding
                labels[i, y_m.sum()+1:] = -100  # +1 for EOS
        # return
        match self.mode:
            case 'train':
                return ys, labels
            case 'val':
                return ys, labels, xs, subs
            case 'test':
                # eval batch size as 1
                return raw_xs[0], xs, idx[0], tgt[0], pos[0], p[0], ctx[0], vocab[0], subs[0]
            case _:
                raise NotImplementedError

    def __getitem__(self, idx):
        return self.get_instance(idx)