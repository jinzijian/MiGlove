#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/12 6:41 下午
# @Author  : Gear

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
from torch.utils.data import TensorDataset

 


#构建bert_embedding
old_path = '../atomic2020/event_center/processed_train_split_graph.txt'
new_path = '../atomic2020/new_bert_pretest.txt'
gen_sentences(old_path, new_path)
old_lines, new_lines, src_b, src_e, tgt_b, tgt_e = get_node_ids(old_path, new_path)
old_lines, node2id, id2node, edge2id, edgelist = read_data(old_lines)

print("len of lines")
print(len(old_lines))
print(len(new_lines))
if os.path.exists("train_new_bert_embedding4.pt"):
     bert_embs = torch.load("train_new_bert_embedding4.pt")
else:
     bert_embs = get_bert_embedding(new_lines)
     torch.save(bert_embs, "train_new_bert_embedding4.pt")
bert_embs = convertBert(old_lines, node2id, bert_embs[0], src_b, src_e, tgt_b, tgt_e)