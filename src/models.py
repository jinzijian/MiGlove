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

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
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

    def forward(self, g, in_feat, edge_feats):
        h = self.conv1(g, in_feat, edge_feats)
        h = F.relu(h)
        h = self.conv2(g, h, edge_feats)
        return h

def random_walks(G, num_walks=100, walk_len=10, string_nid=False):
    paths = []
    # add self loop
    for nid in G.nodes(): G.add_edge(nid, nid)
    if not string_nid:
        for nid in G.nodes():
            if G.degree(nid) == 0: continue
            for i in range(num_walks):
                tmp_path = [str(nid)]
                for j in range(walk_len):
                    neighbors = [str(n) for n in G.neighbors(int(tmp_path[-1]))]
                    tmp_path.append(random.choice(neighbors))
                paths.append(tmp_path)
    else:
        for nid in G.nodes():
            if G.degree(nid) == 0: continue
            for i in range(num_walks):
                tmp_path = [nid]
                for j in range(walk_len):
                    neighbors = [n for n in G.neighbors(tmp_path[-1])]
                    tmp_path.append(random.choice(neighbors))
                paths.append(tmp_path)

    return paths