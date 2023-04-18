#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# built-in
import os
# public
import torch
from lightning.pytorch import seed_everything
# private
from config import Config
# from src import helper
from src.trainer import LitTrainer


class LSP(object):
    """docstring for LSP"""
    def __init__(self):
        super(LSP).__init__()
        self.config = Config()
        self.initialize()

    def initialize(self):
        # get trainer
        self.trainer = LitTrainer(self.config)
        # setup random seed
        seed_everything(self.config.seed, workers=True)
        # enable tokenizer multi-processing
        if self.config.num_workers > 0:
            os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        # others
        torch.set_float32_matmul_precision('high')

def main() -> None:
    lsp = LSP()
    lsp.trainer.train()

if __name__ == '__main__':
      main()