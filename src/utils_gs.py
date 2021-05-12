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



# from load_graph import load_reddit, inductive_split, load_ogb


def load_subtensor(nfeat, labels, seeds, input_nodes, device):
    """
    Extracts features and labels for a subset of nodes.
    """
    batch_inputs = nfeat[input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels


def evaluate(model, my_net, labels, val_nid, val_mask, batch_s, num_worker, device):
    model.eval()
    with torch.no_grad():
        label_pred = model.inference(my_net, val_nid, batch_s, num_worker, device)
    model.train()
    return (torch.argmax(label_pred[val_mask], dim=1) == labels[val_mask]).float().sum() / len(label_pred[val_mask])


def construct_negative_graph(graph, k, args):
    src, dst = graph.edges()

    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(), (len(src) * k,)).to(args.gpu)
    return dgl.graph((neg_src, neg_dst), num_nodes=graph.num_nodes())


class DotProductPredictor(nn.Module):
    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return graph.edata['score']


def compute_loss(pos_score, neg_score):
    n_edges = pos_score.shape[0]
    return (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()