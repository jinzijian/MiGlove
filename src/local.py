# coding:utf-8
import random
import sklearn
import dgl
import torch
import os
import re
import sys
from bert_emb import *
import numpy as np
import importlib
import argparse
import itertools
from createGraph import construct_graph
import scipy.sparse as sp
from models import *
import networkx as nx
from utils import random_walks, setup_seed
import tqdm
from sklearn.metrics import roc_auc_score
from probe import *
from template import *
from eval_func import gen_deepwalkemb, test_embedding
from sklearn.preprocessing import StandardScaler

class NegativeSampler(object):
    def __init__(self, g, k, neg_share=False):
        self.weights = g.in_degrees().float() ** 0.75
        self.k = k
        self.neg_share = neg_share

    def __call__(self, g, eids):
        src, _ = g.find_edges(eids)
        n = len(src)
        if self.neg_share and n % self.k == 0:
            dst = self.weights.multinomial(n, replacement=True)
            dst = dst.view(-1, 1, self.k).expand(-1, self.k, -1).flatten()
        else:
            dst = self.weights.multinomial(n*self.k, replacement=True)
        src = src.repeat_interleave(self.k)
        return src, dst


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_models', type=str, default='bert-base-uncased', help="pretrain models")
    parser.add_argument('--batch_size', type=int, default=2048, help="mi batch_size")
    parser.add_argument('--g_batch_size', type=int, default=20, help="graph batch_size")
    parser.add_argument('--num_workers', type=int, default=1, help="num workers")
    parser.add_argument('--task', type=str, default='probe', help="probe or just eval graph embeddings")
    parser.add_argument('--mode', type=str, default='toy', help="use which dataset to train")
    parser.add_argument('--epoch', type=int, default=200, help="max state of GNN model")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--method", type=str, default='graphsage', help="the method to get graph embeddings")
    parser.add_argument("--repeat", type=int, default=1, help="repeat times")
    parser.add_argument("--mimethod", type=str, default='mine', help="type of mi method'")
    parser.add_argument("--milr", type=float, default=1e-6 , help="learning rate of compute mutual information")
    parser.add_argument("--hidden_size", type=int, default=64, help="probe hidden size")
    parser.add_argument("--nonlinear", type=str, default='sigmoid', help="nonlinear")
    parser.add_argument("--baselines", type=bool, default=True, help="whether calculate baselines of MI")
    parser.add_argument('--onlybaseline', type=bool, default= False, help="only gg and gr")
    parser.add_argument('--noglove', type=bool, default=False, help="without using glove")
    parser.add_argument('--nonode', type=bool, default=False, help="without using glove")
    parser.add_argument('--norelation', type=bool, default=False, help="without using glove")
    parser.add_argument('--g_hiddensize', type=int, default=128, help="hidden size of GNN method")
    parser.add_argument('--seed', type=int, default=10, help="random seed")
    args = parser.parse_args()
    #set seed
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

# set device
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(args.gpu)

# construct graph
train_path = '/p300/MiGlove/atomic2020/event_center/forgraph/processed_train_split_graph1.txt'
test_path = '/p300/MiGlove/atomic2020/event_center/forgraph/processed_test_split_graph1.txt'
dev_path = '/p300/MiGlove/atomic2020/event_center/forgraph/processed_dev_split_graph1.txt'
emb_path = '/p300/TensorFSARNN/data/emb/glove.6B'
# todo: Sample a small datasets to choose parameters
if args.mode == 'train':
    train_path = '/p300/MiGlove/atomic2020/event_center/forgraph/processed_train_split_graph1.txt'
if args.mode == 'toy':
    train_path = '/p300/MiGlove/atomic2020/event_center/forgraph/toy_g_train.txt'
if args.mode == 'sample':
    train_path = '/p300/MiGlove/atomic2020/event_center/forgraph/processed_dev_split_graph1.txt'
if args.mode == 'hinder':
    train_path = '/p300/MiGlove/atomic2020/event_center/forgraph/hinder.txt'
if args.mode == 'before':
    train_path = '/p300/MiGlove/atomic2020/event_center/forgraph/before.txt'
if args.mode == 'after':
    train_path = '/p300/MiGlove/atomic2020/event_center/forgraph/after.txt'
if args.mode == 'reason':
    train_path = '/p300/MiGlove/atomic2020/event_center/forgraph/reason.txt'
if args.mode == 'causes':
    train_path = '/p300/MiGlove/atomic2020/event_center/forgraph/causes.txt'
if args.mode == 'subevent':
    train_path = '/p300/MiGlove/atomic2020/event_center/forgraph/subevent.txt'
if args.mode == 'filled':
    train_path = '/p300/MiGlove/atomic2020/event_center/forgraph/filled.txt'

train_g, train_node2id, train_id2node, train_edgelist, train_word2idx, train_idx2word, train_node_feats, train_edge_feats, train_emb_vectors = construct_graph(
    train_path, emb_path)
test_g, test_node2id, test_id2node, test_edgelist, test_word2idx, test_idx2word, test_node_feats, test_edge_feats, test_emb_vectors = construct_graph(
    test_path, emb_path)
dev_g, dev_node2id, dev_id2node, dev_edgelist, dev_word2idx, dev_idx2word, dev_node_feats, dev_edge_feats, dev_emb_vectors = construct_graph(
    dev_path, emb_path)

# Split edge set for training and testing
g = train_g
u, v = g.edges()
print(g.num_nodes())
eids = np.arange(g.number_of_edges())
eids = np.random.permutation(eids)
test_size = int(len(eids) * 0.1)
train_size = g.number_of_edges() - test_size
test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

print('1')


n_edges = g.num_edges()
train_seeds = torch.arange(n_edges)
u, v = torch.tensor([0, 0, 0, 1]), torch.tensor([1, 2, 3, 3])
g = dgl.graph((u, v))
n_edges = g.num_edges()
train_seeds = torch.arange(n_edges)
sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
dataloader = dgl.dataloading.EdgeDataLoader(
        g, train_seeds, sampler, exclude='reverse_id',
        # For each edge with ID e in Reddit dataset, the reverse edge is e ± |E|/2.
        reverse_eids=torch.cat([
            torch.arange(n_edges // 2, n_edges),
            torch.arange(0, n_edges // 2)]).to(train_seeds),
        negative_sampler=NegativeSampler(g, 2,2),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)
print('ok')
for batch, (input_nodes, positive_graph, negative_graph, blocks) in enumerate(dataloader):
    print('start')
    blocks = [b.to(torch.device('cuda')) for b in blocks]
    positive_graph = positive_graph.to(torch.device('cuda'))
    negative_graph = negative_graph.to(torch.device('cuda'))
print('finish')