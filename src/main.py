# coding:utf-8
import dgl
import torch
import os
import re
import sys
from bert_emb import get_bert_embedding
import numpy as np
import importlib
import argparse
import itertools
from createGraph import construct_graph
import scipy.sparse as sp
from models import *
from gensim.models import Word2Vec
import networkx as nx
from utils import random_walks
import tqdm
from sklearn.metrics import roc_auc_score
import probe
from template import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='toy', help="use which dataset to train")
    parser.add_argument('--epoch', type=int, default=200, help="max state of GNN model")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--method", type=str, default='graphsage', help="the method to get graph embeddings")
    parser.add_argument("--repeat", type=int, default=1, help="repeat times")
    parser.add_argument("--mimethod", type=str, default='mine', help="type of mi method'")
    parser.add_argument("--milr", type=float, default=1e-6 , help="learning rate of compute mutual information")
    args = parser.parse_args()

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

# get bert embeddings
bert_embedding = torch.randn(g.num_nodes(), 768)
if args.mode == 'toy':
    old_path = '/p300/MiGlove/atomic2020/event_center/forgraph/toy_g_train.txt'
if args.mode == 'sample':
    old_path = '/p300/MiGlove/atomic2020/event_center/processed_dev_split_graph.txt'
if args.mode == 'train':
    old_path = '/p300/MiGlove/atomic2020/event_center/processed_train_split_graph.txt'
new_path = '/p300/MiGlove/atomic2020/' + args.mode + 'bert_pretest.txt'
gen_sentences(old_path, new_path)
old_lines, new_lines, src_b, src_e, tgt_b, tgt_e = get_node_ids(old_path, new_path)

# 得到bert embbeddings
if os.path.exists(args.mode + "bert_embedding.pt"):
    bert_embs = torch.load(args.mode + "bert_embedding.pt")
else:
    bert_embs = get_bert_embedding(new_lines, args)
    torch.save(bert_embs, args.mode + "bert_embedding.pt")
old_lines, node2id, id2node, edge2id, edgelist = read_data(old_path)
# 取出node embeddings
node_emb = get_node_emb(old_lines, node2id, bert_embs, src_b, src_e, tgt_b, tgt_e)
for i in range(len(node_emb)):
    node_len = node_emb[i].shape
    if (node_len != (768,)):
        print(i)
        print(id2node[i])
node_emb = np.stack(node_emb, axis=0)
bert_embedding = torch.from_numpy(node_emb)
print('node')
# for i in range(len(node_emb)):
#     for j in range(768):
#         bert_embedding[i][j] = node_emb[i][j]
# print('convert')

# get graph emb
if (args.method == 'graphsage' or args.method == 'nmp'):
    # Find all negative edges and split them for training and testing
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())), shape=(g.number_of_nodes(), g.number_of_nodes()))
    adj_neg = 1 - adj.todense()
    adj_neg = adj_neg - np.eye(g.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)

    neg_eids = np.random.choice(len(neg_u), g.number_of_edges() // 2)
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]

    train_g = dgl.remove_edges(g, eids[:test_size])
    # print(train_g.num_nodes())
    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())

    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())
    #move to gpu
    if use_cuda:
        g = g.to(args.gpu)
        train_g = train_g.to(args.gpu)
        train_pos_g = train_pos_g.to(args.gpu)
        train_neg_g = train_neg_g.to(args.gpu)
        test_pos_g = test_pos_g.to(args.gpu)
        test_neg_g = test_neg_g.to(args.gpu)

if args.method == 'deepwalk':
    src, tgt = train_g.edges()
    local_graph = nx.Graph()
    ge_vec = []
    for i in range(len(src)):
        local_graph.add_edge(int(src[i]), int(tgt[i]))
    local_walks = random_walks(local_graph, 100, 10)
    local_walks = local_walks
    local_idx = []
    print(len(local_graph.nodes()))
    for node in local_graph.nodes():
        local_idx.append(str(node))
    if len(local_idx) > 1:
        local_model = Word2Vec(local_walks,
                               size=128,
                               window=2,
                               min_count=0,
                               sg=1,
                               hs=1,
                               workers=20)
        local_model = local_model
        local_vec = local_model.wv[local_idx]
    else:
        local_vec = np.zeros((1, 128))
    # save graph embeddings (global + local)
    # global_vec = global_model.wv[global_idx]
    # ge_vec.append(np.concatenate((global_vec, local_vec), axis=1))
    print(len(local_vec))
    node_embedding = torch.Tensor(local_vec)


if args.method == 'graphsage':
    # Define Model
    model = GraphSAGE(train_g.ndata['feats'].squeeze().shape[1], 128)
    model = model.to(args.gpu)
    # You can replace DotPredictor with MLPPredictor.
    # pred = MLPPredictor(16)
    pred = DotPredictor()
    pred = pred.to(args.gpu)
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=args.lr)
    # ----------- training -------------------------------- #
    all_logits = []
    for e in range(args.epoch):
        # forward
        if (args.method == 'graphsage'):
            h = model(train_g, train_g.ndata['feats'].squeeze())
        if (args.method == 'nmp'):
            h = model(train_g, train_g.ndata['feats'].squeeze(), train_g.edata['feats'].squeeze())
        pos_score = pred(train_pos_g, h)
        neg_score = pred(train_neg_g, h)
        loss = compute_loss(args, pos_score, neg_score, use_cuda)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            print('In epoch {}, loss: {}'.format(e, loss))

    with torch.no_grad():
        pos_score = pred(test_pos_g, h)
        neg_score = pred(test_neg_g, h)
        print('AUC', compute_auc(pos_score, neg_score))
    # ----------- get node emb --------------------------------
    # evaluate model:
    model.eval()
    with torch.no_grad():
        h = model(g, g.ndata['feats'].squeeze())
        print(h.type())
        print(h.size())
    node_embedding = h

if args.method == 'nmp':
    model = NMP(train_g.ndata['feats'].squeeze().shape[1], 128, train_g.edata['feats'].squeeze().shape[1])
    pred = DotPredictor()
    model = model.to(args.gpu)
    pred = pred.to(args.gpu)
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=args.lr)

    # ----------- training -------------------------------- #
    all_logits = []
    for e in range(args.epoch):
        # forward
        if (args.method == 'graphsage'):
            h = model(train_g, train_g.ndata['feats'].squeeze())
        if (args.method == 'nmp'):
            h = model(train_g, train_g.ndata['feats'].squeeze(), train_g.edata['feats'].squeeze())
        pos_score = pred(train_pos_g, h)
        neg_score = pred(train_neg_g, h)
        loss = compute_loss(args, pos_score, neg_score, use_cuda)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            print('In epoch {}, loss: {}'.format(e, loss))

    # ----------- check results ------------------------ #
    with torch.no_grad():
        pos_score = pred(test_pos_g, h)
        neg_score = pred(test_neg_g, h)
        print('AUC', compute_auc(pos_score, neg_score))
        # evaluate model:
    model.eval()
    with torch.no_grad():
        h = model(train_g, train_g.ndata['feats'].squeeze(), train_g.edata['feats'].squeeze())
        print(h.type())
        print(h.size())
    node_embedding = h

# probe
sen_sum = len(bert_embedding)
bert_layers_num = 12
mir, mig, mib = [], [], []
for l in range(bert_layers_num): mib.append([])
for r in range(args.repeat):
    tmp_mir = probe.mi_probe(args, node_embedding, bert_embedding, sen_sum, 'lower')
    tmp_mig = probe.mi_probe(args, node_embedding, bert_embedding, sen_sum, 'upper')
    # get sum value
    if len(mir) == 0:
        mir = tmp_mir
    else:
        mir = [mir[s] + tmp_mir[s] for s in range(len(tmp_mir))]
    if len(mig) == 0:
        mig = tmp_mig
    else:
        mig = [mig[s] + tmp_mig[s] for s in range(len(tmp_mig))]
for l in range(bert_layers_num):
    # bert_emb = np.load(bert_emb_paths[l], allow_pickle=True)
    for r in range(args.repeat):
        tmp_mib = probe.mi_probe(args, node_embedding, bert_embedding, sen_sum, l)
        if len(mib[l]) == 0:
            mib[l] = tmp_mib
        else:
            mib[l] = [mib[l][s] + tmp_mib[s] for s in range(len(tmp_mib))]

# compute average values for all results
mir = [mi / args.repeat for mi in mir]
mig = [mi / args.repeat for mi in mig]
for l in range(bert_layers_num):
    mib[l] = [mi / args.repeat for mi in mib[l]]
# torch.save(mi_eval, 'result.pt ')
mib_layers = [sum(mib[l]) / len(mib[l]) for l in range(len(mib)) if len(mib)]
mir, mig, mib_layers = sum(mir) / len(mir), sum(mig) / len(mig), mib_layers
print('MI(G, R): {} | MI(G, G): {}| MI(G, BERT): {} |'.format(mir, mig, mib_layers))
#
# mi_eval = probe.mi_probe(args, node_embedding, bert_embedding, sen_sum, 'upper')
# print(mi_eval)
