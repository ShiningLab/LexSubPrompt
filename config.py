#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'


# dependency
# built-in
import os, argparse
# private
from src import helper


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    # data
    # genesis for data of https://aclanthology.org/2021.emnlp-main.844.pdf
    # ls14, ls21
    parser.add_argument('--data', type=str, default='ls21')
    # base, sample, wsample
    parser.add_argument('--data_mode', type=str, default='wsample')
    # base, full, best, exbest
    parser.add_argument('--prompt_mode', type=str, default='best')
    # base, full
    parser.add_argument('--train_mode', type=str, default='base')
    # model
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--pdrop', type=float, default=0.2)
    # gpt2, gpt2-medium, gpt2-large, gpt2-xl
    # gpt-neo-125m, gpt-neo-350m, gpt-neo-1.3B, gpt-neo-2.7B
    parser.add_argument('--model', type=str, default='gpt2-medium')
    parser.add_argument('--load_ckpt', type=helper.str2bool, default=False)
    # bert-large-cased
    parser.add_argument('--lm', type=str, default='bert-large-cased')
    # training
    parser.add_argument('--train_batch_size', type=int, default=16)  # for training
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--val_batch_size', type=int, default=32)  # for valiation
    parser.add_argument('--test_batch_size', type=int, default=1)  # for test
    parser.add_argument('--max_epochs', type=int, default=-1)  # -1 to enable infinite training
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)
    parser.add_argument('--fast_dev_run', type=helper.str2bool, default=False)  # True for development
    # evaluation
    # val_p1, val_p10
    parser.add_argument('--monitor', type=str, default='val_p1')
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--val_check_interval', type=float, default=1.0)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=12)
    parser.add_argument('--num_beams', type=int, default=50)
    parser.add_argument('--num_return_sequences', type=int, default=50)
    # trainer
    # 16-mixed, 32-true
    parser.add_argument('--precision', type=str, default='32-true')
    # (str, optional) Can be 'simple' or 'advanced'. Defaults to ''.
    parser.add_argument('--profiler', type=str, default='')
    # logger
    parser.add_argument('--offline', type=helper.str2bool, default=True)  # True for development
    # (str, optional) Can be 'online', 'offline' or 'disabled'. Defaults to online.
    parser.add_argument('--wandb_mode', type=str, default='disabled')  # disabled for testing code
    parser.add_argument('--log_model', type=helper.str2bool, default=False)
    # save as argparse space
    return parser.parse_known_args()[0]

class Config(object):
    """docstring for Config"""
    def __init__(self):
        super(Config, self).__init__()
        self.update_config(**vars(init_args()))

    def update_config(self, **kwargs):
        # load config from parser
        for k,v in kwargs.items():
            setattr(self, k, v)
        # update config
        match self.model:
            case 'gpt2':
                pass
            case 'gpt2-medium':
                if self.data == 'ls14':
                    self.precision = '16-mixed'
            case 'gpt2-large':
                self.train_batch_size //= 2
                self.accumulate_grad_batches = 16 // self.train_batch_size
                if self.data == 'ls14':
                    self.precision = '16-mixed'
            case 'gpt2-xl':
                self.train_batch_size //= 4
                self.accumulate_grad_batches = 16 // self.train_batch_size
                self.precision = '16-mixed'
            case 'gpt-neo-125m':
                pass
            case 'gpt-neo-350m':
                if self.data == 'ls14':
                    self.precision = '16-mixed'
            case _:
                raise NotImplementedError
        # I/O
        self.CURR_PATH = './'
        self.RESOURCE_PATH = os.path.join(self.CURR_PATH, 'res')
        self.DATA_PATH = os.path.join(self.RESOURCE_PATH, 'data')
        os.makedirs(self.DATA_PATH, exist_ok=True)
        self.DATA_PKL = os.path.join(self.DATA_PATH, f'{self.data}.pkl')
        # language model
        self.LM_PATH = os.path.join(self.RESOURCE_PATH, 'lm', self.lm)  # rank
        self.MODEL_PATH = os.path.join(self.RESOURCE_PATH, 'lm', self.model)  # backbone
        # checkpoints
        self.CKPT_PATH = os.path.join(
            self.RESOURCE_PATH, 'ckpts', self.data
            , self.data_mode, self.prompt_mode, self.train_mode, self.model, str(self.seed)
            )
        os.makedirs(self.CKPT_PATH, exist_ok=True)
        self.CKPT_LAST = os.path.join(self.CKPT_PATH, 'last.ckpt')
        # log
        self.ENTITY = 'ENTITY'
        self.PROJECT = 'PromptSub'
        self.NAME = f'{self.data}-{self.data_mode}-{self.prompt_mode}-{self.train_mode}-{self.model}-{self.seed}'
        self.LOG_PATH = os.path.join(
            self.RESOURCE_PATH, 'log', self.data
            , self.data_mode, self.prompt_mode, self.train_mode, self.model, str(self.seed)
            )
        os.makedirs(self.LOG_PATH, exist_ok=True)
        self.LOG_TXT = os.path.join(self.LOG_PATH, 'console_log.txt')
        os.remove(self.LOG_TXT) if os.path.exists(self.LOG_TXT) else None
        # results
        self.RESULTS_PATH = os.path.join(
            self.RESOURCE_PATH, 'results', self.data
            , self.data_mode, self.prompt_mode, self.train_mode, self.model
            )
        os.makedirs(self.RESULTS_PATH, exist_ok=True)
        self.RESULTS_PKL = os.path.join(self.RESULTS_PATH, f'{self.seed}.pkl')