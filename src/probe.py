import os
import gc
import random
import torch
import networkx as nx
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from models import MINE, NWJ, InfoNCE
#from embed import graph_embeddings, bert_embeddings, get_embeddings
#from utils2 import load_data, construct_graph, load_glove, load_elmo, load_elmos

'''
def mine_probe(args, graph_emb, bert_emb, sen_num, task_name, noisy_id=[]):
    bert_dim = bert_emb['s0'].shape[1]
    graph_dim = graph_emb['s0'].shape[1]
    if task_name == 'upper':
        bert_dim = graph_dim
    model = graph_probe(graph_dim, bert_dim).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    bad_np = [39927]
    mi_es = [-1 for _ in range(args.patience)]
    model.train()
    for epoch in range(10):  # epoch
        mi_train = []
        for i in range(sen_num):  # batch
            if i in bad_np: continue
            graph_vec = graph_emb['s'+str(i)]
            if task_name == 'lower':
                feat_vec = torch.randn(size=bert_emb['s'+str(i)].shape)
            elif task_name == 'upper':
                feat_vec = graph_vec
            elif type(task_name) == int:
                feat_vec = bert_emb['s'+str(i)]
            else:
                print('Error probe task name: ', task_name)
            graph_vec = torch.FloatTensor(graph_vec).to(args.device)
            feat_vec = torch.FloatTensor(feat_vec).to(args.device)

            optimizer.zero_grad()
            if graph_vec.shape[0] <= 1: continue
            if feat_vec.shape[0] <= 1: continue
            joint = torch.mean(model(graph_vec, feat_vec))
            feat_shuffle = feat_vec[torch.randperm(feat_vec.shape[0])]
            marginal = torch.exp(torch.clamp(model(graph_vec, feat_shuffle), max=88))
            marginal = torch.log(torch.mean(marginal))
            mi = joint - marginal
            loss = -mi
            mi_train.append(mi.data.item())

            loss.backward()
            optimizer.step()
        # early stop
        if sum(mi_train)/len(mi_train) > min(mi_es):
            mi_es.remove(min(mi_es))
            mi_es.append(sum(mi_train)/len(mi_train))
        else:
            break

    mi_eval = []
    model.eval()
    for i in range(sen_num):  # batch
        graph_vec = graph_emb['s'+str(i)]
        if task_name == 'lower':
            feat_vec = torch.randn(size=bert_emb['s'+str(i)].shape)
        elif task_name == 'upper':
            feat_vec = graph_vec
        elif type(task_name) == int:
            feat_vec = bert_emb['s'+str(i)]
        else:
            print('Error probe task name: ', task_name)
        graph_vec = torch.FloatTensor(graph_vec).to(args.device)
        feat_vec = torch.FloatTensor(feat_vec).to(args.device)

        optimizer.zero_grad()
        if graph_vec.shape[0] <= 1: 
            mi_eval.append(0.)
            continue
        if feat_vec.shape[0] <= 1: 
            mi_eval.append(0.)
            continue
        joint = torch.mean(model(graph_vec, feat_vec))
        feat_shuffle = feat_vec[torch.randperm(feat_vec.shape[0])]
        marginal = torch.exp(torch.clamp(model(graph_vec, feat_shuffle), max=88))
        marginal = torch.log(torch.mean(marginal))
        mi = joint - marginal
        loss = -mi
        mi_eval.append(mi.data.item())

        loss.backward()
        optimizer.step()

    print(" ----Testing probe model: {} | Epoch {:05d} | MI: {:.4f}".format(
                                                        task_name,
                                                        epoch + 1, 
                                                        sum(mi_eval)/len(mi_eval)))

    # free memory
    model = None
    optimizer = None
    torch.cuda.empty_cache()
    gc.collect()

    return mi_eval
'''


def mi_probe(args, graph_emb, bert_emb, sen_num, task_name, mi_method="mine"):
    mi_method = args.mimethod
    bert_dim = bert_emb.shape[1]
    graph_dim = graph_emb.shape[1]
    if task_name == 'upper':
        bert_dim = graph_dim
    print(graph_emb.shape, bert_emb[0].shape)
    if mi_method == 'mine':
        model = MINE(args, graph_dim, bert_dim, hidden_size = args.hidden_size).to(args.gpu)
    elif mi_method == 'nwj':
        model = NWJ(args, graph_dim, bert_dim, hidden_size = args.hidden_size).to(args.gpu)
    elif mi_method == 'nce':
        model = InfoNCE(args, graph_dim, bert_dim, hidden_size = args.hidden_size).to(args.gpu)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.milr)
    batch_size = args.batch_size
    loader_graph = DataLoader(graph_emb, batch_size=batch_size)
    loader_bert = DataLoader(bert_emb, batch_size=batch_size)

    bad_np = []
    mi_es = [-1 for _ in range(2)]
    model.train()
    for epoch in range(10):  # epoch
        mi_train = []
        print('MI epoch is' + str(epoch))
        # for i in tqdm(range(sen_num)):  # batch
        for graph_vec, feat_vec in zip(loader_graph, loader_bert):
            # if i in bad_np: continue
            if task_name == 'lower':
                feat_vec = torch.randn(feat_vec.shape)
            elif task_name == 'upper':
                feat_vec = graph_vec
            elif type(task_name) == int:
                feat_vec = feat_vec
            else:
                print('Error probe task name: ', task_name)

            graph_vec = graph_vec.to(args.gpu)
            feat_vec = feat_vec.to(args.gpu)
            if graph_vec.shape[0] <= 1: continue
            if feat_vec.shape[0] <= 1: continue
            # feat_vec = feat_vec.unsqueeze(0)
            # graph_vec = graph_vec.unsqueeze(0)
            loss = model.learning_loss(feat_vec, graph_vec)
            mi = -loss
            mi_train.append(mi.data.item())

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        # early stop
        if sum(mi_train)/len(mi_train) > min(mi_es):
            mi_es.remove(min(mi_es))
            mi_es.append(sum(mi_train)/len(mi_train))
        else:
            break

    mi_eval = []
    model.eval()
    # for i in tqdm(range(sen_num)):  # batch
    for graph_vec, feat_vec in zip(loader_graph, loader_bert):
        if task_name == 'lower':
            feat_vec = torch.randn(size=feat_vec.shape)
        elif task_name == 'upper':
            feat_vec = graph_vec
        elif type(task_name) == int:
            feat_vec = feat_vec
        else:
            print('Error probe task name: ', task_name)
        graph_vec = graph_vec.to(args.gpu)
        feat_vec = feat_vec.to(args.gpu)
        if graph_vec.shape[0] <= 1:
            mi_eval.append(0.)
            continue
        if feat_vec.shape[0] <= 1:
            mi_eval.append(0.)
            continue

        # feat_vec = feat_vec.unsqueeze(0)
        # graph_vec = graph_vec.unsqueeze(0)
        loss = model.learning_loss(feat_vec, graph_vec)
        mi = -loss
        mi_eval.append(mi.data.item())

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

    print(" ----Testing probe model: {} | Epoch {:05d} | MI: {:.4f}".format(
                                                        task_name,
                                                        epoch + 1,
                                                        sum(mi_eval)/len(mi_eval)))

    # free memory
    model = None
    optimizer = None
    torch.cuda.empty_cache()
    gc.collect()

    return mi_eval


def probe_func_probe(args, graph_emb, bert_emb, uncontext=False):
    if args.pretrained_models == 'bert-base-uncased':
        bert_layers_num = 13
    else:
        bert_layers_num = 25

    # initialize mi
    mir, mig, mib = [], [], []
    for l in range(bert_layers_num): mib.append([])
    s_num = len(graph_emb)

    if args.baselines:
        print('3.1 start to calculate baselines of MI...')
        # calculate MI baselines
        for r in range(args.repeat):
            tmp_mir = mi_probe(args, graph_emb, bert_emb[0], s_num, 'lower')
            tmp_mig = mi_probe(args, graph_emb, bert_emb[0], s_num, 'upper')
            # get sum value
            if len(mir) == 0:
                mir = tmp_mir
            else:
                mir = [mir[s] + tmp_mir[s] for s in range(len(tmp_mir))]
            if len(mig) == 0:
                mig = tmp_mig
            else:
                mig = [mig[s] + tmp_mig[s] for s in range(len(tmp_mig))]
    if not args.onlybaseline:
        print('2.2 start to calculate BERT hidden states of MI...')
        # calculate MI of BERT

        if uncontext:
            for r in range(args.repeat):
                tmp_mib = mi_probe(args, graph_emb, bert_emb, s_num, bert_layers_num - 1)
                if len(mib[-1]) == 0:
                    mib[-1] = tmp_mib
                else:
                    mib[-1] = [mib[-1][s] + tmp_mib[s] for s in range(len(tmp_mib))]
            mib_layers = sum(mib[-1]) / (len(mib[-1]) * args.repeat)
            print('MI(G, Glove): {} |'.format(mib_layers))
        else:
            for l in range(bert_layers_num):
                print(l)
                # bert_emb = np.load(bert_emb_paths[l], allow_pickle=True)
                for r in range(args.repeat):
                    tmp_mib = mi_probe(args, graph_emb, bert_emb[l], s_num, l)
                    if len(mib[l]) == 0:
                        mib[l] = tmp_mib
                    else:
                        mib[l] = [mib[l][s] + tmp_mib[s] for s in range(len(tmp_mib))]

            # compute average values for all results
            mir = [mi / args.repeat for mi in mir]
            mig = [mi / args.repeat for mi in mig]
            for l in range(bert_layers_num):
                print(l)
                mib[l] = [mi / args.repeat for mi in mib[l]]

            # print general results
            # results = {'lower:': mir, 'upper': mig, 'bert': mib}
            # print('\n', results, '\n')
            mib_layers = [sum(mib[l]) / len(mib[l]) for l in range(len(mib)) if len(mib)]
    if args.onlybaseline:
        mib_layers = 0
    return sum(mir) / len(mir), sum(mig) / len(mig), mib_layers

def probe_plain(args, graph_emb, bert_emb):
    # probe
    mir, mig, mib_layers = probe_func_probe(args, graph_emb, bert_emb)
    if args.onlybaseline:
        res = 'MI(G, R): {} | MI(G, G): {} |'.format(mir, mig)
        print('MI(G, R): {} | MI(G, G): {}|'.format(mir, mig))
    else:
        res = 'MI(G, R): {} | MI(G, G): {}| MI(G, BERT): {} |'.format(mir, mig, mib_layers)
        print('MI(G, R): {} | MI(G, G): {}| MI(G, BERT): {} |'.format(mir, mig, mib_layers))
    return res