#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# public
import torch
import numpy as np
from torchmetrics import Metric

# borrowed from https://raw.githubusercontent.com/SapienzaNLP/genesis/main/src/metrics.py
class PrecisionAtOne(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")

    def update(self, preds: list[str], target: list[list[str]]):
        for i, top_prediction in enumerate(preds):
            if top_prediction in target[i]:
                self.correct += 1
        self.total += len(target)

    def compute(self):
        return self.correct.float() / self.total

# def accuracy(predicted: list[str], gold: list[str]) -> float:
#     return np.float32(sum(np.array(predicted) == np.array(gold)) / len(gold))

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

def recall_at_k(gold: list[str], predicted: list[str], k: int) -> float:
    num = len([x for x in predicted[:k] if x in gold])
    den = len(gold[:k])
    return num / den