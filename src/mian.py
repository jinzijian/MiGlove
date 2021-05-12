# coding:utf-8
import dgl
import torch
import os
import re
import sys
import importlib
import numpy as np
import argparse
from createGraph import construct_graph

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help="use which dataset to train")
    parser.add_argument('--epoch', type=int, default=200, help="max state of GNN model")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    args = parser.parse_args()

# set device
use_cuda = torch.cuda.is_available()
if use_cuda and not args.no_cuda:
    torch.cuda.set_device(args.gpu)

# construct graph
train_path = '/p300/MiGlove/atomic2020/event_center/forgraph/processed_train_split_graph1.txt'
test_path = '/p300/MiGlove/atomic2020/event_center/forgraph/processed_test_split_graph1.txt'
dev_path = '/p300/MiGlove/atomic2020/event_center/forgraph/processed_dev_split_graph1.txt'
emb_path = '/p300/TensorFSARNN/data/emb/glove.6B'
# todo: Sample a small datasets to choose parameters
if args.mode == 'small':
    train_path = 'pass'
trian_g, train_node2id, train_id2node, train_edgelist, train_word2idx, train_idx2word, train_node_feats, train_edge_feats, train_emb_vectors = construct_graph(train_path, emb_path)
test_g, test_node2id, test_id2node, test_edgelist, test_word2idx, test_idx2word, test_node_feats, test_edge_feats, test_emb_vectors = construct_graph(test_path, emb_path)
dev_g, dev_node2id, dev_id2node, dev_edgelist, dev_word2idx, dev_idx2word, dev_node_feats, dev_edge_feats, dev_emb_vectors = construct_graph(dev_path, emb_path)

