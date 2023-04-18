#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# public
from transformers import AutoTokenizer
# private
from src import helper

class LSPDataset(object):
    """docstring for LSPDataset"""
    def __init__(self, mode, config, sample_size=100):
        super(LSPDataset, self).__init__()
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.config = config
        self.sample_size = sample_size
        self.get_data()
        self.get_tokenizer()

    def __len__(self): 
        return self.data_size

    def get_data(self):
        data_dict = helper.load_pickle(self.config.DATA_PKL)
        self.texts_list = data_dict[self.mode]
        self.data_size = len(self.texts_list)
        if self.sample_size:
            self.texts_list = self.texts_list[:self.sample_size]
            self.data_size = self.sample_size

    def get_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.LM_PATH)
        # https://github.com/huggingface/transformers/issues/2630
        # https://huggingface.co/patrickvonplaten/bert2gpt2-cnn_dailymail-fp16#training-script
        self.tokenizer.pad_token = self.tokenizer.unk_token

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
        return inputs.input_ids

    def __getitem__(self, idx):
        return self.texts_list[idx]