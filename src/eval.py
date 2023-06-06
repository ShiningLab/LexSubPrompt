#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# public
import torch
import wandb
import numpy as np


class Evaluator(object):
    """docstring for Evaluator"""
    def __init__(self, results_dict):
        super(Evaluator, self).__init__()
        self.ys = results_dict['subs']
        self.ys_ = results_dict['subs_']
        self.rank_ys_ = results_dict['rank_subs_']
        self.clean_ys_ = results_dict['clean_subs_']
        self.get_metrics()
        self.get_info()

    def get_metrics(self):
        self.metrics_dict = {}
        for prefix, ys_ in zip(['ori_', 'rank_', 'clean_'], [self.ys_, self.rank_ys_, self.clean_ys_]):
            p1, p3, r10 = get_metrics(self.ys, ys_)
            self.metrics_dict[f'{prefix}p1'] = p1
            self.metrics_dict[f'{prefix}p3'] = p3
            self.metrics_dict[f'{prefix}r10'] = r10
        # update logger
        try:
            wandb.log(self.metrics_dict)
        except:
            pass

    def get_info(self):
        self.info = '|'
        for k, v in self.metrics_dict.items():
            self.info += '{}:{:.4f}|'.format(k, v)


def get_metrics(gold: list[list[str]], predicted: list[list[str]]) -> (float, float, float):
    p1, p3, r10 = [], [], []
    for g, p in zip(gold, predicted):
        p1.append(precision_at_k(g, p, 1))
        p3.append(precision_at_k(g, p, 3))
        r10.append(recall_at_k(g, p, 10))
    p1 = np.mean(p1, dtype='float32') * 100
    p3 = np.mean(p3, dtype='float32') * 100
    r10 = np.mean(r10, dtype='float32') * 100
    return p1, p3, r10

def overall_precision_at_k(ys, ys_, k) -> float:
    p = []
    for y, y_ in zip(ys, ys_):
        p.append(precision_at_k(y, y_, k))
    return np.mean(p, dtype='float32')

def precision_at_k(gold: list[str], predicted: list[str], k: int) -> float:
    num = len([x for x in predicted[:k] if x in gold])
    den = len(predicted[:k])
    if den == 0:
        return 0
    return num / den

def overall_recall_at_k(ys, ys_, k) -> float:
    p = []
    for y, y_ in zip(ys, ys_):
        p.append(recall_at_k(y, y_, k))
    return np.mean(p, dtype='float32')

def recall_at_k(gold: list[str], predicted: list[str], k: int) -> float:
    num = len([x for x in predicted[:k] if x in gold])
    den = len(gold[:k])
    return num / den