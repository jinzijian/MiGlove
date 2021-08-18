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
from utils_gs import *
import dgl.function as fn
from dgl.nn import SAGEConv
from dgl.nn import NNConv
import random


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean')

    def forward(self, blocks, in_feat):
        h = self.conv1(blocks[0], in_feat)
        h = F.relu(h)
        h = self.conv2(blocks[1], h)
        return h




class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]


######################################################################
# You can also write your own function if it is complex.
# For instance, the following module produces a scalar score on each edge
# by concatenating the incident nodes’ features and passing it to an MLP.
#

class MLPPredictor4deepwalk(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.apply_edges(self.apply_edges)
            return g.edata['score']


class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']

class NMP(nn.Module):
    def __init__(self, in_feats, h_feats, edge_feats):
        super(NMP, self).__init__()
        self.edge_feats = edge_feats
        edge_func1 = torch.nn.Linear(edge_feats, in_feats*h_feats)
        edge_func2 = torch.nn.Linear(edge_feats, h_feats * h_feats)
        self.conv1 = NNConv(in_feats, h_feats, edge_func1, 'mean')
        self.conv2 = NNConv(h_feats, h_feats, edge_func2, 'mean')

    def forward(self, blocks, in_feat, edge_feats, edge_feats1):
        h = self.conv1(blocks[0], in_feat, edge_feats)
        h = F.relu(h)
        h = self.conv2(blocks[1], h, edge_feats1)
        return h




class binary_classifer(nn.Module):
    def __init__(self,
                 layers_num=5,
                 feat_dim=0,
                 hidden_dim=128):
        super(binary_classifer, self).__init__()
        self.layers_num = layers_num
        self.linear_sh = nn.Linear(feat_dim, hidden_dim)
        self.linear_dh = nn.Linear(feat_dim, hidden_dim)
        self.linear_h1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.linear_h2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_h3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_h4 = nn.Linear(hidden_dim, hidden_dim)
        if layers_num == 0:  # 0
            self.linear_ca = nn.Linear(2 * feat_dim, 1)
        elif layers_num <= 1:  # 1
            self.linear_ca = nn.Linear(2 * hidden_dim, 1)
        else:  # 2, 3, 4, 5
            self.linear_ca = nn.Linear(hidden_dim, 1)
        nn.init.xavier_normal_(self.linear_ca.weight.data)

    def forward(self, src, dst):
        # layers_num = 0
        h = torch.cat((src, dst), dim=1)
        if self.layers_num == 0:
            return torch.sigmoid(self.linear_ca(h))

        # layers_num = 1
        s = F.relu(self.linear_sh(src))
        d = F.relu(self.linear_dh(dst))
        h = torch.cat((s, d), dim=1)
        if self.layers_num == 1:
            return torch.sigmoid(self.linear_ca(h))

        # layers_num = 2
        h = F.relu(self.linear_h1(h))
        if self.layers_num == 2:
            return torch.sigmoid(self.linear_ca(h))

        # layers_num = 3
        h = F.relu(self.linear_h2(h))
        if self.layers_num == 3:
            return torch.sigmoid(self.linear_ca(h))

        # layers_num = 4
        h = F.relu(self.linear_h3(h))
        if self.layers_num == 4:
            return torch.sigmoid(self.linear_ca(h))

        # layers_num = 5
        h = F.relu(self.linear_h4(h))
        if self.layers_num == 5:
            return torch.sigmoid(self.linear_ca(h))


'''
class mine_model(nn.Module):
    def __init__(self, 
                adj_dim,
                feat_dim,
                hidden_dim=64):
        super(mine_probe, self).__init__()
        self.linear_ah = nn.Linear(adj_dim, hidden_dim)
        self.linear_fh = nn.Linear(feat_dim, hidden_dim)
        self.linear_hm = nn.Linear(hidden_dim*2, 1)
        nn.init.xavier_normal_(self.linear_ah.weight.data)
        nn.init.xavier_normal_(self.linear_fh.weight.data)
        nn.init.xavier_normal_(self.linear_hm.weight.data)

    def forward(self, a, f):
        a = self.linear_ah(a)
        f = self.linear_fh(f)
        h = torch.cat((f, a), dim=1)

        return F.elu(self.linear_hm(h))
'''


class MINE(nn.Module):
    def __init__(self, args, x_dim, y_dim, hidden_size):
        super(MINE, self).__init__()
        if args.nonlinear == 'relu':
            nonlinear = nn.ReLU()
        if args.nonlinear == 'sigmoid':
            nonlinear = nn.Sigmoid()
        if args.nonlinear == 'elu':
            nonlinear = nn.ELU()
        if args.nonlinear == 'tanh':
            nonlinear = nn.Tanh()
        self.T_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nonlinear,
                                    nn.Linear(hidden_size, 1))

    def forward(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()

        y_shuffle = y_samples[random_index]

        T0 = self.T_func(torch.cat([x_samples, y_samples], dim=-1))
        T1 = self.T_func(torch.cat([x_samples, y_shuffle], dim=-1))

        lower_bound = T0.mean() - torch.log(T1.exp().mean())

        # compute the negative loss (maximise loss == minimise -loss)
        return lower_bound

    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)


class NWJ(nn.Module):
    def __init__(self, args, x_dim, y_dim, hidden_size):
        super(NWJ, self).__init__()
        if args.nonlinear == 'relu':
            nonlinear = nn.ReLU()
        if args.nonlinear == 'sigmoid':
            nonlinear = nn.Sigmoid()
        if args.nonlinear == 'elu':
            nonlinear = nn.ELU()
        if args.nonlinear == 'tanh':
            nonlinear = nn.Tanh()
        self.F_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nonlinear,
                                    nn.Linear(hidden_size, 1))

    def forward(self, x_samples, y_samples):
        # shuffle and concatenate
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        guess = torch.cat([x_samples, y_samples], dim=-1)
        T0 = self.F_func(guess)
        guess2 = torch.cat([x_tile, y_tile], dim=-1)
        T1 = self.F_func(guess2) - 1.  # shape [sample_size, sample_size, 1]

        lower_bound = T0.mean() - (T1.logsumexp(dim=1) - np.log(sample_size)).exp().mean()
        return lower_bound

    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)


class InfoNCE(nn.Module):
    def __init__(self, args, x_dim, y_dim, hidden_size):
        super(InfoNCE, self).__init__()
        if args.nonlinear == 'relu':
            nonlinear = nn.ReLU()
        if args.nonlinear == 'sigmoid':
            nonlinear = nn.Sigmoid()
        if args.nonlinear == 'elu':
            nonlinear = nn.ELU()
        if args.nonlinear == 'tanh':
            nonlinear = nn.Tanh()
        self.F_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nonlinear,
                                    nn.Linear(hidden_size, 1),
                                    nn.Softplus())

    def forward(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self.F_func(torch.cat([x_samples, y_samples], dim=-1))
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim=-1))  # [sample_size, sample_size, 1]

        lower_bound = T0.mean() - (T1.logsumexp(dim=1).mean() - np.log(sample_size))
        return lower_bound

    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)

