#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'


# dependency
# built-in
import sys, pickle, logging, argparse
# public
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoConfig
    , AutoModel
    , AutoTokenizer
    , AutoModelForPreTraining
    , GPTNeoForCausalLM
    )
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
# private
from src import trainers, datasets


NLTK_POS_DICT = {
    'adjective':wordnet.ADJ, 'verb':wordnet.VERB, 'noun':wordnet.NOUN, 'adverb':wordnet.ADV
    , 'ADJ':wordnet.ADJ, 'VERB':wordnet.VERB, 'NOUN':wordnet.NOUN, 'ADV':wordnet.ADV
}

POS_DICT = {}
POS_DICT['ADJ'] = 'adjective'
POS_DICT['ADV'] = 'adverb'
POS_DICT['NOUN'] = 'noun'
POS_DICT['VERB'] = 'verb'
POS_DICT.update({v:k for k, v in POS_DICT.items()})


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
        model_config = AutoConfig.from_pretrained(config.MODEL_PATH)
        model_config.resid_pdrop = config.pdrop
        model_config.embd_pdrop = config.pdrop
        model_config.attn_pdrop = config.pdrop
        model_config.scale_attn_by_inverse_layer_idx = True
        model_config.reorder_and_upcast_attn = True
        return AutoModelForPreTraining.from_pretrained(
            config.MODEL_PATH
            , config=model_config
            )

def get_wordnet_synonyms(word, pos):
    """
    Retrieve synonyms for a given word from WordNet.
    """
    synonyms = set()
    wordnet_pos = NLTK_POS_DICT.get(pos, None)
    for syn in wordnet.synsets(word, pos=wordnet_pos):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', '-'))  # Replace underscores with spaces
    synonyms = set([s.lower() for s in synonyms])
    if word in synonyms:
        synonyms.remove(word)
    if synonyms:
        return 'synonyms ' + ', '.join([f'"{s}"' for s in sorted(synonyms)])
    else:
        return 'none synonyms'

def rank(target, position, context, subs_, config):
    # initialize
    rank_subs_ = []
    lm = AutoModel.from_pretrained(config.LM_PATH).to(config.device)
    tokenizer = AutoTokenizer.from_pretrained(config.LM_PATH)
    lm.eval()
    # iterate each instance
    for tgt, p, ctx, s_ in zip(tqdm(target), position, context, subs_):
        # get all contexts
        ctxs = []
        tk_ctx = ctx.split()
        for i in [tk_ctx[p], tgt] + s_:
            # processing
            for pun in ['_', '-']:
                i = i.replace(pun, ' ')
            # get all contexts
            ctx = ' '.join(tk_ctx[:p] + [i] + tk_ctx[p+1:])
            ctxs.append(ctx)
        # contexts representation
        ctxs = tokenizer.batch_encode_plus(
            ctxs, return_tensors='pt', padding=True).to(config.device)
        with torch.no_grad():
            hs = lm(**ctxs, output_hidden_states=True).hidden_states
        # mean of all layers
        hs = torch.stack(hs).mean(dim=0)
        # flatten each tensor into a vector
        hs = hs.reshape(hs.shape[0], -1)
        # take the mean of the first two as the target
        tgt_h = hs[:2].mean(dim=0)
        # calculate the cosine similarities
        cos_sims = []
        for h in hs[2:]:
            cos_sims.append(F.cosine_similarity(h, tgt_h, dim=0).item())
        cos_sims = torch.Tensor(cos_sims)
        # get the indices that would sort the scores
        rank_indices = cos_sims.argsort(descending=True).cpu().detach().numpy().tolist()
        # sort the items
        rank_subs_.append([s_[i] for i in rank_indices])
    return rank_subs_

def get_pos(word):
    # preprocessing
    word = word.replace('_', ' ')
    # get all synsets for the word
    synsets = wordnet.synsets(word)
    # collect all pos tags for each synset
    pos_tags = {synset.pos() for synset in synsets}
    # the tag "s" stands for "Satellite Adjective"
    if wordnet.ADJ_SAT in pos_tags:
        pos_tags.remove(wordnet.ADJ_SAT)
        pos_tags.add(wordnet.ADJ)
    return pos_tags

def postprocess(target, pos, subs_):
    clean_subs_ = []
    lemmatizer = WordNetLemmatizer()
    for tgt, pos, sub_ in zip(target, pos, subs_):
        # remove spaces
        sub_ = [s.strip() for s in sub_]
        # unify part-of-speech
        pos = NLTK_POS_DICT[pos]
        # PoS filtering
        sub_ = [s for s in sub_ if pos in get_pos(s)]
        # lemmatization
        sub_ = [lemmatizer.lemmatize(s, pos) for s in sub_]
        # remove the target word
        tgts = {tgt, lemmatizer.lemmatize(tgt, pos)}
        sub_ = [s for s in sub_ if s not in tgts]
        # remove duplicates but keep the order
        sub_ = list(dict.fromkeys(sub_))
        # save
        clean_subs_.append(sub_)
    return clean_subs_