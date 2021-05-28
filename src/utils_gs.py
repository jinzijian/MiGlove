#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/14 2:06 下午
# @Author  : Gear

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.nn.pytorch as dglnn
import time
import argparse
import tqdm
import networkx as nx
import dgl.function as fn
from sklearn.metrics import roc_auc_score
def compute_loss(args, pos_score, neg_score, use_cuda):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    if use_cuda:
        labels = labels.to(args.gpu)
    loss = F.binary_cross_entropy_with_logits(scores, labels)
    return loss

def compute_auc(pos_score, neg_score):
    pos_score = pos_score.cpu()
    neg_score = neg_score.cpu()
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)