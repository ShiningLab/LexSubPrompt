#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# built-in
import sys, pickle, logging
# public
from transformers import AutoModelForPreTraining, GPTNeoForCausalLM


def save_pickle(obj, path):
    """
    To save a object as a pickle file.
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_pickle(path):
    """
    To load object from pickle file.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)

def str2bool(v):
    """Method to map string to bool for argument parser"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def init_logger(config):
    """initialize the logger"""
    file_handler = logging.FileHandler(filename=config.LOG_TXT)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(
        encoding='utf-8'
        , format='%(asctime)s | %(message)s'
        , datefmt='%Y-%m-%d %H:%M:%S'
        , level=logging.INFO
        , handlers=handlers
        )
    logger = logging.getLogger(__name__)
    return logger

def flatten_list(regular_list: list) -> list:
    return [item for sublist in regular_list for item in sublist]

def get_model(config):
    # gpt-neo-125m, gpt-neo-350m
    if 'neo' in config.model:
        return GPTNeoForCausalLM.from_pretrained(
            config.LM_PATH
            )
    else:
        return AutoModelForPreTraining.from_pretrained(
            config.LM_PATH
            , scale_attn_by_inverse_layer_idx=True
            , reorder_and_upcast_attn=True
            )