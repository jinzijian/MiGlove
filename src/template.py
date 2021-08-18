#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/25 0:13 上午
# @Author  : Gear
# event_center = ['HinderedBy', 'isAfter', 'HasSubEvent', 'isBefore', 'xReason','Causes','isFilledBy']
from bert_emb import get_bert_embedding
import torch
import os
import numpy as np

def deal_sentence(line):
    if 'hindered by' in line[1]:
        p = line.index('hindered by')
        line[p] = 'is hindered by'
        sentence = ''
        for word in line[:-1]:
            if (word == '\n'):
                continue
            sentence += word
            sentence += ' '
        sentence += '\n'
        return sentence

    if 'causes' in line[1]:
        sentence = ''
        for word in line[:-1]:
            if (word == '\n'):
                continue
            sentence += word
            sentence += ' '
        sentence += '\n'
        return sentence

    if 'has sub event' in line[1]:
        p = line.index('has sub event')
        line[p] = 'has a sub event that'
        sentence = ''
        for word in line[:-1]:
            if (word == '\n'):
                continue
            sentence += word
            sentence += ' '
        sentence += '\n'
        return sentence

    if 'after' in line[1]:
        sentence = ''
        for word in line[:-1]:
            if (word == '\n'):
                continue
            sentence += word
            sentence += ' '
        sentence += '\n'
        return sentence

    if 'before' in line[1]:
        sentence = ''
        for word in line[:-1]:
            if (word == '\n'):
                continue
            sentence += word
            sentence += ' '
        sentence += '\n'
        return sentence

    if 'reason' in line[1]:
        p = line.index('reason')
        line[p] = 'because'
        sentence = ''
        for word in line:
            if (word == '\n'):
                continue
            sentence += word
            sentence += ' '
        sentence += '\n'
        return sentence

    if 'is filled by' in line[1]:
        # # p = line.index('reason')
        # line_tmp = line[0].split(' ')
        # p = line_tmp.index('kkk')
        # line_tmp[p] = line[-2]
        # print(line_tmp)
        sentence = ''
        for word in line[:-1]:
            if (word == '\n'):
                continue
            sentence += word
            sentence += ' '
        sentence += '\n'
        return sentence


def equal_list(list1, list2):
    if (len(list1) != len(list2)):
        return False
    for i in range(len(list1)):
        if (list1[i] != list2[i]):
            return False
    return True


def find_idx(string_list, node_list):
    length = len(node_list)
    for i in range(len(string_list) - length + 1):
        sub = string_list[i:i + length]
        if (equal_list(sub, node_list)):
            return i
    return -1

def gen_sentences(old_path, new_path):
    with open(old_path, "r", encoding='utf-8') as f:
        lines = f.readlines()
    sentences = []
    print(len(lines))
    for line in lines:
        line = line.split('\t')
        sentence = deal_sentence(line)
        sentences.append(sentence)
    print(len(sentences))
    file = open(new_path, 'w', encoding='utf-8')
    for sentence in sentences:
        if (sentence.count('\n') == 2):
            print(sentence)
        file.write(sentence)
    file.close()
    return 0




def get_node_ids(old_path, new_path):
    with open(old_path, "r", encoding='utf-8') as f:
        old_lines = f.readlines()
    with open(new_path, 'r', encoding='utf-8') as f:
        new_lines = f.readlines()
        src_b = []
        src_e = []
        tgt_b = []
        tgt_e = []
        for i in range(len(old_lines)):
            old_line = old_lines[i].split('\t')
            new_lines[i] = new_lines[i].lstrip()
            new_line = new_lines[i].split(' ')
            sb = find_idx(new_line, old_line[0].split(' '))
            se = sb + len(old_line[0].split(' '))
            tb = find_idx(new_line, old_line[2].split(' '))
            te = tb + len(old_line[2].split(' '))
            src_b.append(sb)
            src_e.append(se)
            tgt_b.append(tb)
            tgt_e.append(te)
    return old_lines, new_lines, src_b, src_e, tgt_b, tgt_e

def get_node_emb(old_lines, node2id, bert_embs, src_b, src_e, tgt_b, tgt_e):
    node_emb = [[] for i in range(len(node2id))]
    for i in range(len(old_lines)):
        line = old_lines[i]
        line = line.split('\t')
        bert_emb = bert_embs[i]
        src_emb = bert_emb[src_b[i]:src_e[i]]
        tgt_emb = bert_emb[tgt_b[i]: tgt_e[i]]
        src = line[0]
        rel = line[1]
        tgt = line[2]
        src_id = node2id[src]
        tgt_id = node2id[tgt]
        node_emb[src_id].append(src_emb)
        node_emb[tgt_id].append(tgt_emb)
    #convert to np
    print('converting to np')
    node_emb = np.array(node_emb)
    for i in range(len(node_emb)):
        for j in range(len(node_emb[i])):
            for k in range(len(node_emb[i][j])):
                node_emb[i][j][k] = node_emb[i][j][k].cpu()
                node_emb[i][j][k] = node_emb[i][j][k].numpy()
    for i in range(len(node_emb)):
        node_emb[i] = np.array(node_emb[i])
        for j in range(len(node_emb[i])):
            node_emb[i][j] = np.array(node_emb[i][j])
    for i in range(len(node_emb)):
        node_emb[i] = np.mean(node_emb[i], axis=0)
    for i in range(len(node_emb)):
        node_emb[i] = np.mean(node_emb[i], axis=0)
    return node_emb

def convertBert(old_lines, node2id, bert_embs, src_b, src_e, tgt_b, tgt_e):
    node_emb = get_node_emb(old_lines, node2id, bert_embs, src_b, src_e, tgt_b, tgt_e)
    for i in range(len(node_emb)):
        node_len = node_emb[i].shape
        if (node_len != (768,)):
            print(i)
            #print(id2node[i])
            print(node_len)
    node_emb = np.stack(node_emb, axis=0)
    bert_embedding = torch.from_numpy(node_emb)
    return bert_embedding


import argparse
from createGraph import read_data
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='toy', help="use which dataset to train")
    parser.add_argument('--epoch', type=int, default=200, help="max state of GNN model")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--method", type=str, default='graphsage', help="the method to get graph embeddings")
    parser.add_argument("--repeat", type=int, default=1, help="repeat times")
    parser.add_argument("--mimethod", type=str, default='mine', help="type of mi method'")
    args = parser.parse_args()


    if args.mode == 'toy':
        old_path = '/p300/MiGlove/atomic2020/event_center/forgraph/toy_g_train.txt'
    if args.mode == 'sample':
        old_path = '/p300/MiGlove/atomic2020/event_center/processed_dev_split_graph.txt'
    if args.mode == 'train':
        old_path = '/p300/MiGlove/atomic2020/event_center/processed_train_split_graph.txt'
    new_path = '/p300/MiGlove/atomic2020/' + args.mode + 'bert_pretest.txt'
    with open(old_path, "r", encoding='utf-8') as f:
        lines = f.readlines()
    print('bug')




