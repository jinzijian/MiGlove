# coding:utf-8
import dgl
import torch
import os
import re
import sys
import importlib
import numpy as np
import argparse
import itertools
from createGraph import construct_graph
import scipy.sparse as sp
from models import*
from gensim.models import Word2Vec
import networkx as nx

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='toy', help="use which dataset to train")
    parser.add_argument('--epoch', type=int, default=200, help="max state of GNN model")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--method", type=str, default='graphsage', help="the method to get graph embeddings")
    args = parser.parse_args()

# set device
use_cuda = torch.cuda.is_available()
if use_cuda :
    torch.cuda.set_device(args.gpu)

# construct graph
train_path = '/p300/MiGlove/atomic2020/event_center/forgraph/processed_train_split_graph1.txt'
test_path = '/p300/MiGlove/atomic2020/event_center/forgraph/processed_test_split_graph1.txt'
dev_path = '/p300/MiGlove/atomic2020/event_center/forgraph/processed_dev_split_graph1.txt'
emb_path = '/p300/TensorFSARNN/data/emb/glove.6B'
# todo: Sample a small datasets to choose parameters
if args.mode == 'small':
    train_path = 'pass'
if args.mode == 'toy':
    train_path = '/p300/MiGlove/atomic2020/event_center/forgraph/toy_g_train.txt'
if args.mode == 'sample':
    train_path = '/p300/MiGlove/atomic2020/event_center/forgraph/sample_g.txt'
train_g, train_node2id, train_id2node, train_edgelist, train_word2idx, train_idx2word, train_node_feats, train_edge_feats, train_emb_vectors = construct_graph(train_path, emb_path)
test_g, test_node2id, test_id2node, test_edgelist, test_word2idx, test_idx2word, test_node_feats, test_edge_feats, test_emb_vectors = construct_graph(test_path, emb_path)
dev_g, dev_node2id, dev_id2node, dev_edgelist, dev_word2idx, dev_idx2word, dev_node_feats, dev_edge_feats, dev_emb_vectors = construct_graph(dev_path, emb_path)



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

# Find all negative edges and split them for training and testing
adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())), shape=(g.number_of_nodes(), g.number_of_nodes()))
adj_neg = 1 - adj.todense()
adj_neg = adj_neg - np.eye(g.number_of_nodes())
neg_u, neg_v = np.where(adj_neg != 0)

neg_eids = np.random.choice(len(neg_u), g.number_of_edges() // 2)
test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]

train_g = dgl.remove_edges(g, eids[:test_size])
train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())

test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())

if args.method == 'deepwalk':
    src, tgt = train_g.edges()
    G = nx.Graph()
    for i in range(len(src)):
        G.add_edge(int(src[i]), int(tgt[i]))
    local_walks = random_walks(G, 100, 10)
    local_model = Word2Vec(local_walks,
                           size=128,
                           window=2,
                           min_count=0,
                           sg=1,
                           hs=1,
                           workers=20)
    node_embedding = local_model.wv[]
    print(node_embedding)
if args.method == 'graphsage':
    # Define Model
    model = GraphSAGE(train_g.ndata['feats'].squeeze().shape[1], 128)
    # You can replace DotPredictor with MLPPredictor.
    pred = MLPPredictor(16)
    #pred = DotPredictor()
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=args.lr)
if args.method == 'nmp':
    model = NMP(train_g.ndata['feats'].squeeze().shape[1], 128, train_g.edata['feats'].squeeze().shape[1])
    pred = DotPredictor()
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=args.lr)

# ----------- 4. training -------------------------------- #
all_logits = []
for e in range(args.epoch):
    # forward
    if(args.method == 'graphsage'):
        h = model(train_g, train_g.ndata['feats'].squeeze())
    if(args.method == 'nmp'):
        h = model(train_g, train_g.ndata['feats'].squeeze(), train_g.edata['feats'].squeeze())
    pos_score = pred(train_pos_g, h)
    neg_score = pred(train_neg_g, h)
    loss = compute_loss(pos_score, neg_score)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if e % 5 == 0:
        print('In epoch {}, loss: {}'.format(e, loss))

# ----------- 5. check results ------------------------ #
from sklearn.metrics import roc_auc_score
with torch.no_grad():
    pos_score = pred(test_pos_g, h)
    neg_score = pred(test_neg_g, h)
    print('AUC', compute_auc(pos_score, neg_score))