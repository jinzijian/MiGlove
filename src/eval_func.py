import torch
import numpy as np
from sklearn.metrics import roc_auc_score, jaccard_score
from tqdm import tqdm
from models import binary_classifer
from utils import random_walks
from gensim.models import Word2Vec
import networkx as nx
from models import *
import torch
import itertools

def gen_deepwalkemb(graph):
    graph = graph.cpu()
    local_graph = graph.to_networkx()
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
    node_embedding = torch.Tensor(local_vec)
    return node_embedding

def test_embedding(args, train_emb, train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)
    hfeats = 128
    pred = MLPPredictor(hfeats).to(args.gpu)
    optimizer = torch.optim.Adam(itertools.chain(pred.parameters()), lr=args.lr)
    # backward
    for e in range(args.epoch):
        # forward
        h = train_emb.to(args.gpu)
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
