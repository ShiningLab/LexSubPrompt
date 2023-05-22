#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# built-in
import sys, pickle, logging
# public
import torch
import torch.nn.functional as F
from tqdm import trange
from transformers import (
    AutoModel
    , AutoTokenizer
    , AutoModelForPreTraining
    , GPTNeoForCausalLM
    )


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
            config.MODEL_PATH
            )
    else:
        return AutoModelForPreTraining.from_pretrained(
            config.MODEL_PATH
            , scale_attn_by_inverse_layer_idx=True
            , reorder_and_upcast_attn=True
            )

def postprocess(outputs_dict):
    # generate clean_subs_ based on subs_
    clean_subs_ = []
    for t, s, s_ in zip(outputs_dict['target'], outputs_dict['subs'], outputs_dict['subs_']):
        # remove spaces
        s_ = [i.strip() for i in s_]
        # remove duplicates but keep the order
        s_ = list(dict.fromkeys(s_))
        # remove the target word itself
        if t in s_:
            s_.remove(t)
        clean_subs_.append(s_)
    return clean_subs_

def rank(outputs_dict, config):
    # generate rank_subs_ based on clean_subs_
    # initialize
    rank_subs = []
    lm = AutoModel.from_pretrained(config.LM_PATH).to(config.device)
    tokenizer = AutoTokenizer.from_pretrained(config.LM_PATH)
    lm.eval()
    for idx in trange(len(outputs_dict['target'])):
        p = outputs_dict['position'][idx]  # target word position
        ctx = outputs_dict['context'][idx]  # original context 
        s_ = outputs_dict['clean_subs_'][idx]  # cleaned substitutes
        # get all contexts
        ctxs_ = []
        ctx = ctx.split()
        for i in s_:
            i = i.replace('_', ' ')
            ctx_ = ctx[:p] + [i] + ctx[p+1:]
            ctx_ = ' '.join(ctx_)
            ctxs_.append(ctx_)
        ctx = ' '.join(ctx)
        # contexts representation
        ctxs_ = [ctx] + ctxs_  # the first is the target context
        ctxs_ = tokenizer.batch_encode_plus(ctxs_, return_tensors='pt', padding=True).to(config.device)
        with torch.no_grad():
            hs = lm(**ctxs_, output_hidden_states=True).hidden_states
        # mean of all layers
        hs = torch.mean(torch.stack(hs), dim=0)
        # flatten each tensor into a single vector
        hs = hs.view(hs.shape[0], -1)
        # normalize each vector
        hs = F.normalize(hs, dim=-1)
        # take the first vector as the target one
        tgt_h = hs[0]
        # calculate the cosine similarities
        cos_sims = torch.matmul(hs[1:], tgt_h)
        # get the indices that would sort the scores
        rank_indices = cos_sims.argsort(descending=True).cpu().detach().numpy().tolist()
        # sort the items
        rank_subs.append([s_[i] for i in rank_indices])
    return rank_subs