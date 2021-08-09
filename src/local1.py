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
    parser.add_argument('--num_workers', type=int, default=0, help="num workers")
    parser.add_argument('--task', type=str, default='probe', help="probe or just eval graph embeddings")
    parser.add_argument('--mode', type=str, default='toy', help="use which dataset to train")
    parser.add_argument('--epoch', type=int, default=200, help="max state of GNN model")
    parser.add_argument("--gpu", type=int, default=1, help="gpu")
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
# if args.mode == 'hinder':
#     train_path = '/p300/MiGlove/atomic2020/event_center/forgraph/hinder.txt'
# if args.mode == 'before':
#     train_path = '/p300/MiGlove/atomic2020/event_center/forgraph/before.txt'
# if args.mode == 'after':
#     train_path = '/p300/MiGlove/atomic2020/event_center/forgraph/after.txt'
# if args.mode == 'reason':
#     train_path = '/p300/MiGlove/atomic2020/event_center/forgraph/reason.txt'
# if args.mode == 'causes':
#     train_path = '/p300/MiGlove/atomic2020/event_center/forgraph/causes.txt'
# if args.mode == 'subevent':
#     train_path = '/p300/MiGlove/atomic2020/event_center/forgraph/subevent.txt'
# if args.mode == 'filled':
#     train_path = '/p300/MiGlove/atomic2020/event_center/forgraph/filled.txt'

train_g, train_node2id, train_id2node, train_edgelist, train_word2idx, train_idx2word, train_node_feats, train_edge_feats, train_emb_vectors =construct_graph(
    train_path, emb_path)
test_g, test_node2id, test_id2node, test_edgelist, test_word2idx, test_idx2word, test_node_feats, test_edge_feats, test_emb_vectors= construct_graph(
    test_path, emb_path)
dev_g, dev_node2id, dev_id2node, dev_edgelist, dev_word2idx, dev_idx2word, dev_node_feats, dev_edge_feats, dev_emb_vectors = construct_graph(
    dev_path, emb_path)

# Split edge set for training and testing
# g = train_g
# u, v = g.edges()
# print(g.num_nodes())
# eids = np.arange(g.number_of_edges())
# eids = np.random.permutation(eids)
# test_size = int(len(eids) * 0.1)
# train_size = g.number_of_edges() - test_size
# test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
# train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]
#
# print('1')
from dgl.data import RedditDataset
print('FULL GRAPH INFO')
print(train_g)

# SET dataloader
n_edges = train_g.num_edges()
train_seeds = torch.arange(n_edges)
sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
dataloader = dgl.dataloading.EdgeDataLoader(
        train_g, train_seeds, sampler, exclude='reverse_id',
        # For each edge with ID e in Reddit dataset, the reverse edge is e ± |E|/2.
        reverse_eids=torch.cat([
            torch.arange(n_edges // 2, n_edges),
            torch.arange(0, n_edges // 2)]).to(train_seeds),
        negative_sampler=NegativeSampler(train_g, 2,2),
        batch_size=50000,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)
print('ok')
from createGraph import getNE
for batch, (input_nodes, positive_graph, negative_graph, blocks) in enumerate(dataloader):
    print(positive_graph)
    u1,v1=dgl.block_to_graph(blocks[0]).edges()
    g=dgl.graph((u1,v1))
    g = g.to(args.gpu)
    g=getNE(args, train_path, emb_path, g)
    print(g.num_edges())
    # g=positive_graph.clone()
    u,v=g.edges()
    eids = np.arange(g.number_of_edges())
    eids = np.random.permutation(eids)
    # eids=input_nodes.numpy()
    # print(eids)
    test_size = int(len(eids) * 0.1)
    train_size = g.number_of_edges() - test_size
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]


    # get graph emb
    if (args.method == 'graphsage' or args.method == 'nmp'):
        # Find all negative edges and split them for training and testing
        u = u.cpu()
        v = v.cpu()
        adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())), shape=(g.number_of_nodes(), g.number_of_nodes()))
        adj_neg = 1 - adj.todense()
        adj_neg = adj_neg - np.eye(g.number_of_nodes())
        neg_u, neg_v = np.where(adj_neg != 0)

        neg_eids = np.random.choice(len(neg_u), g.number_of_edges() // 2)
        test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
        train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]
        train_g = dgl.remove_edges(g,eids[test_size:])

        train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
        train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())

        test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
        test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())
        # move to gpu
        if use_cuda:
            # g = g.to(args.gpu)
            train_g = train_g.to(args.gpu)
            train_pos_g = train_pos_g.to(args.gpu)
            train_neg_g = train_neg_g.to(args.gpu)
            test_pos_g = test_pos_g.to(args.gpu)
            test_neg_g = test_neg_g.to(args.gpu)

    if (args.method == 'deepwalk'):
        adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())), shape=(g.number_of_nodes(), g.number_of_nodes()))
        adj_neg = 1 - adj.todense().astype('float16')
        eye = np.eye(g.number_of_nodes())
        adj_neg = adj_neg.astype('float16')
        adj_neg = adj_neg - eye.astype('float16')
        neg_u, neg_v = np.where(adj_neg != 0, dtype=np.int8)

        neg_eids = np.random.choice(len(neg_u), g.number_of_edges() // 2)
        test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
        train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]

        train_g = dgl.remove_edges(g, eids[:test_size])
        # print(train_g.num_nodes())
        train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
        train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())

        test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
        test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())
        # move to gpu
        if use_cuda:
            g = g.to(args.gpu)
            train_g = train_g.to(args.gpu)
            train_pos_g = train_pos_g.to(args.gpu)
            train_neg_g = train_neg_g.to(args.gpu)
            test_pos_g = test_pos_g.to(args.gpu)
            test_neg_g = test_neg_g.to(args.gpu)

    if args.method == 'deepwalk':
        train_emb = gen_deepwalkemb(train_g)
        # eval and test
        auc = test_embedding(args, train_emb, train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g)
        node_embedding = train_emb.to(args.gpu)

    if args.norelation:
        edata = train_g.edata['feats']
        train_g.edata['feats'] = torch.randn(edata.size()).to(args.gpu)

    if args.nonode:
        ndata = train_g.ndata['feats']
        train_g.ndata['feats'] = torch.randn(ndata.size()).to(args.gpu)

    if args.noglove:
        ndata = train_g.ndata['feats']
        edata = train_g.edata['feats']
        train_g.ndata['feats'] = torch.randn(ndata.size()).to(args.gpu)
        train_g.edata['feats'] = torch.randn(edata.size()).to(args.gpu)

    if args.method == 'graphsage':
        # Define Model
        model = GraphSAGE(train_g.ndata['feats'].squeeze().shape[1], args.g_hiddensize)
        model = model.to(args.gpu)
        # You can replace DotPredictor with MLPPredictor.
        # pred = MLPPredictor(128)
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
            auc = compute_auc(pos_score, neg_score)
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
        model = NMP(train_g.ndata['feats'].squeeze().shape[1], args.g_hiddensize,
                    train_g.edata['feats'].squeeze().shape[1])
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
            auc = compute_auc(pos_score, neg_score)
            print('AUC', compute_auc(pos_score, neg_score))
            # evaluate model:
        model.eval()
        with torch.no_grad():
            h = model(train_g, train_g.ndata['feats'].squeeze(), train_g.edata['feats'].squeeze())
            print(h.type())
            print(h.size())
        node_embedding = h
        node_embedding = node_embedding.cpu()
        node_embedding = sklearn.preprocessing.normalize(node_embedding)
        node_embedding = torch.from_numpy(node_embedding).float()
        node_embedding = node_embedding.to(args.gpu)

    print('here is ')
    print(auc)

    # save reslut
    path_file_name = '/p300/MiGlove/src/auc_result.txt'
    if not os.path.exists(path_file_name):
        fileObject = open('/p300/MiGlove/src/auc_result.txt', 'w', encoding='utf-8')
    else:
        fileObject = open('/p300/MiGlove/src/auc_result.txt', 'a', encoding='utf-8')
    if args.norelation:
        fileObject.write('no relation' + str(auc) + args.mode + ' ' + args.method + ' ' + args.mimethod + ' ' + str(
            args.lr) + ' ' + str(args.milr) + ' ' + str(args.hidden_size) + ' ' + str(args.batch_size))
    elif args.noglove:
        fileObject.write('no glove' + str(auc) + args.mode + ' ' + args.method + ' ' + args.mimethod + ' ' + str(
            args.milr) + ' ' + str(args.hidden_size) + ' ' + str(args.batch_size))
    elif args.nonode:
        fileObject.write('no node' + str(auc) + args.mode + ' ' + args.method + ' ' + args.mimethod + ' ' + str(
            args.milr) + ' ' + str(args.hidden_size) + ' ' + str(args.batch_size))
    else:
        fileObject.write(
            str(auc) + args.mode + ' ' + args.method + ' ' + args.mimethod + ' ' + str(args.milr) + ' ' + str(
                args.hidden_size) + ' ' + str(args.batch_size))
    fileObject.write('\n')
    fileObject.close()

    if args.task == 'probe':
        # set seed
        random.seed()
        os.environ['PYTHONHASHSEED'] = 'random'
        np.random.seed()
        torch.backends.cudnn.deterministic = False
        # get bert embeddings
        bert_embedding = torch.randn(g.num_nodes(), 768)
        if args.mode == 'toy':
            old_path = '/p300/MiGlove/atomic2020/event_center/forgraph/toy_g_train.txt'
        if args.mode == 'sample':
            old_path = '/p300/MiGlove/atomic2020/event_center/processed_dev_split_graph.txt'
        if args.mode == 'train':
            old_path = '/p300/MiGlove/atomic2020/event_center/processed_train_split_graph.txt'
        if args.mode == 'hinder':
            old_path = '/p300/MiGlove/atomic2020/event_center/hinder.txt'
        if args.mode == 'before':
            old_path = '/p300/MiGlove/atomic2020/event_center/before.txt'
        if args.mode == 'after':
            old_path = '/p300/MiGlove/atomic2020/event_center/after.txt'
        if args.mode == 'reason':
            old_path = '/p300/MiGlove/atomic2020/event_center/reason.txt'
        if args.mode == 'causes':
            old_path = '/p300/MiGlove/atomic2020/event_center/causes.txt'
        if args.mode == 'subevent':
            old_path = '/p300/MiGlove/atomic2020/event_center/subevent.txt'
        if args.mode == 'filled':
            old_path = '/p300/MiGlove/atomic2020/event_center/filled.txt'
        new_path = '/p300/MiGlove/atomic2020/' + args.mode + 'bert_pretest.txt'
        gen_sentences(old_path, new_path)
        old_lines, new_lines, src_b, src_e, tgt_b, tgt_e = get_node_ids(old_path, new_path)

        # 得到bert embbeddings
        if os.path.exists(args.mode + "all_new_bert_embedding.pt"):
            bert_embs = torch.load(args.mode + "0610new_bert_embedding.pt")
        else:
            bert_embs = get_bert_embedding(new_lines, args)
            torch.save(bert_embs, args.mode + "0610new_bert_embedding.pt")
        old_lines, node2id, id2node, edge2id, edgelist = read_data(old_path)
        # 取出node embeddings
        for i in range(len(bert_embs)):
            bert_embs[i] = convertBert(old_lines, node2id, bert_embs[i], src_b, src_e, tgt_b, tgt_e)
        # bert_embedding = convertBert(old_lines, node2id, bert_embs, src_b, src_e, tgt_b, tgt_e)
        # for i in range(len(node_emb)):
        #     for j in range(768):
        #         bert_embedding[i][j] = node_emb[i][j]
        # print('convert')
        bert_embedding = bert_embs
        # probe
        # graph -> node_embedding; bert -> bert_embedding
        res = probe_plain(args, node_embedding, bert_embedding)

        # save reslut
        path_file_name = '/p300/MiGlove/src/result.txt'
        if not os.path.exists(path_file_name):
            fileObject = open('/p300/MiGlove/src/result.txt', 'w', encoding='utf-8')
        else:
            fileObject = open('/p300/MiGlove/src/result.txt', 'a', encoding='utf-8')
        if args.norelation:
            fileObject.write('no relation' + res + args.mode + ' ' + args.method + ' ' + args.mimethod + ' ' + str(
                args.milr) + ' ' + str(args.hidden_size) + ' ' + str(args.batch_size))
        elif args.noglove:
            fileObject.write('no glove' + res + args.mode + ' ' + args.method + ' ' + args.mimethod + ' ' + str(
                args.milr) + ' ' + str(args.hidden_size) + ' ' + str(args.batch_size))
        elif args.nonode:
            fileObject.write('no node' + res + args.mode + ' ' + args.method + ' ' + args.mimethod + ' ' + str(
                args.milr) + ' ' + str(args.hidden_size) + ' ' + str(args.batch_size))
        else:
            fileObject.write(
                res + args.mode + ' ' + args.method + ' ' + args.mimethod + ' ' + str(args.milr) + ' ' + str(
                    args.hidden_size) + ' ' + str(args.batch_size))
        fileObject.write('\n')
        fileObject.close()
    print('start')
    # blocks = [b for b in blocks]
    # positive_graph = positive_graph
    # negative_graph = negative_graph
print('finish')