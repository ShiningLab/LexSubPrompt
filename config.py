#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# built-in
import os, argparse
# private
from src import helper


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    # data
    # genesis for data of https://aclanthology.org/2021.emnlp-main.844.pdf
    parser.add_argument('--data', type=str, default='genesis')
    # model
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--model', type=str, default='gpt2')
    parser.add_argument('--load_ckpt', type=helper.str2bool, default=False)
    # training
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--max_epochs', type=int, default=-1)  # to enable infinite training
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    # evaluation
    parser.add_argument('--patience', type=int, default=6)
    parser.add_argument("--val_check_interval", type=float, default=1.0)
    # trainer
    # (str, optional) Can be 'simple' or 'advanced'. Defaults to ''.
    parser.add_argument('--profiler', type=str, default='')
    # logger
    parser.add_argument('--offline', type=helper.str2bool, default=True) # True for development
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
        # I/O
        # self.CURR_PATH = os.path.dirname(os.path.realpath(__file__))
        self.CURR_PATH = './'
        self.RESOURCE_PATH = os.path.join(self.CURR_PATH, 'res')
        self.DATA_PATH = os.path.join(self.RESOURCE_PATH, 'data')
        os.makedirs(self.DATA_PATH, exist_ok=True)
        self.DATA_PKL = os.path.join(self.DATA_PATH, f'{self.data}.pkl')
        # # language model
        self.LM_PATH = os.path.join(self.RESOURCE_PATH, 'lm', self.model)
        # checkpoints
        self.CKPT_PATH = os.path.join(
            self.RESOURCE_PATH, 'ckpts', self.data, self.model, str(self.seed)
            )
        os.makedirs(self.CKPT_PATH, exist_ok=True)
        self.CKPT_LAST = os.path.join(self.CKPT_PATH, 'last.ckpt')
        # log
        self.ENTITY = 'mrshininnnnn'
        self.PROJECT = 'LSP'
        self.NAME = f'{self.data}-{self.model}-{self.seed}'
        self.LOG_PATH = os.path.join(
            self.RESOURCE_PATH, 'log', self.data, self.model, str(self.seed)
            )
        os.makedirs(self.LOG_PATH, exist_ok=True)
        self.LOG_TXT = os.path.join(self.LOG_PATH, 'console_log.txt')
        os.remove(self.LOG_TXT) if os.path.exists(self.LOG_TXT) else None