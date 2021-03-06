# coding:utf-8
import dgl
import torch
import os
import re
import sys
import importlib
import numpy as np
importlib.reload(sys)
import torch.nn as nn

import utils_gs
import utils



def read_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    node2id = {}
    edge2id = {}
    id2node = {}
    edgelist = []

    nid = 0
    eid = 0
    for line in lines:
        line = line.split('\t')
        #print(len(line))
        if line[0] in node2id:
            pass
        else:
            node2id[line[0]] = nid
            id2node[nid] = line[0]
            nid += 1
        if line[2] in node2id:
            pass
        else:
            node2id[line[2]] = nid
            id2node[nid] = line[2]
            nid += 1
        if line[1] in edge2id:
            pass
        else:
            edge2id[line[1]] = eid
            eid += 1
        edgelist.append(line[1])
    return lines, node2id, id2node, edge2id, edgelist

def construct_plain_graph(lines, node2id):
    src = []
    tgt = []
    for line in lines:
        line = line.split('\t')
        src.append(node2id[line[0]])
        tgt.append(node2id[line[2]])
    src = torch.tensor(src)
    tgt = torch.tensor(tgt)
    g = dgl.graph((src, tgt))
    return g

def get_w2iandi2w(lines):
    word2idx = {}
    idx2word = {}
    wid = 0
    for line in lines:
        line = line.split('\t')
        for sentence in line[:3]:
            sentence = sentence.strip('.')
            sentence = sentence.split(' ')
            for word in sentence:
                if word not in word2idx:
                    word2idx[word] = wid
                    idx2word[wid] = word
                    wid += 1
    return word2idx, idx2word

def make_glove_embed(glove_path, i2t, embed_dim='100'):
    glove = {}
    vecs = [] # use to produce unk

    # load glove
    with open(os.path.join(glove_path, 'glove.6B.{}d.txt'.format(embed_dim)),
              'r', encoding='utf-8') as f:
        for line in f.readlines():
            split_line = line.split()
            word = split_line[0]
            embed_str = split_line[1:]
            embed_float = [float(i) for i in embed_str]
            if word not in glove:
                glove[word] = embed_float
                vecs.append(embed_float)
    unk = np.mean(vecs, axis=0)

    # load glove to task vocab
    embed = []
    for i in i2t:
        word = i2t[i].lower()
        if word in glove:
            embed.append(glove[word])
        else:
            embed.append(unk)

    final_embed = np.array(embed, dtype=np.float)
    return final_embed

def get_node_feats(g, emb_vectors, word2idx, idx2word, id2node):
    num_nodes = g.num_nodes()
    embeddings = np.zeros((1, 100))
    node_feats = []
    for i in range(num_nodes):
        tmp = id2node[i]
        tmp = tmp.strip('.')
        tmp = tmp.split(' ')
        embeddings = np.zeros((1, 100))
        length = len(tmp)
        for word in tmp:
            embb = emb_vectors[word2idx[word]]
            embeddings += embb
        embeddings = embeddings / length
        node_feats.append(embeddings)
    node_feats = torch.tensor(node_feats, dtype=torch.float32)
    return node_feats



def get_edge_feats(edgelist, emb_vectors, word2idx):
    edge_feats = []
    for edge in edgelist:
        edge = edge.strip('.')
        edge = edge.split(' ')
        embeddings = np.zeros((1, 100))
        length = len(edge)
        for word in edge:
            embb = emb_vectors[word2idx[word]]
            embeddings += embb
        embeddings = embeddings / length
        if args.relation == edge:
            embeddings = np.randn((1, 100))
        edge_feats.append(embeddings)
    edge_feats = torch.tensor(edge_feats, dtype=torch.float32)
    return edge_feats


def construct_graph(path, emb_path) -> object:
    lines, node2id, id2node, edge2id, edgelist = read_data(path)
    word2idx, idx2word = get_w2iandi2w(lines)
    g = construct_plain_graph(lines, node2id)
    emb_vectors = make_glove_embed(emb_path, idx2word)
    node_feats = get_node_feats(g, emb_vectors, word2idx, idx2word, id2node)
    edge_feats = get_edge_feats(edgelist, emb_vectors, word2idx)
    g.ndata['feats'] = node_feats
    g.edata['feats'] = edge_feats
    return g, node2id, id2node, edgelist, word2idx, idx2word, node_feats, edge_feats, emb_vectors



if __name__ == "__main__":
    path = '/p300/MiGlove/atomic2020/event_center/forgraph/processed_train_split_graph1.txt'
    emb_path = '/p300/TensorFSARNN/data/emb/glove.6B'
    g, node2id, id2node, edgelist, word2idx, idx2word, node_feats, edge_feats, emb_vectors = construct_graph(path, emb_path)
    print(g.nodes())
    print(g.edges(form='all'))
    print(edgelist[0])
    print(edge_feats[0])
    A = g.edata['infeats'][torch.tensor([0, 1])] #????????????????????? 2 x 1 x 100 ???tensor ???????????????????????????????????? ???????????????
    B = emb_vectors[word2idx[edgelist[0]]]
    C = edge_feats[0]
    if(A == B):
        print('no bug')
    else:
        print('loser')
        print(A.shape)
        print(B.shape)
        print()
        print(A)
        print(B)
    embedding_dim = 128
    k = 5
    epochs = 5
    lpmodel = Pmodel(embedding_dim, 128, 128)
    for i in range(epochs):
        lpmodel.train()
        negative_graph = construct_negative_graph(g, k)
        pos_score, neg_score = lpmodel(g, negative_graph, g.ndata['infeats'])
        loss = compute_loss(pos_score, neg_score)
        opt.zero_grad()
        loss.backward(retain_graph=True)
        opt.step()
        print(loss.item())
    
    
    