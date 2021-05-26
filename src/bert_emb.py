#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/1 3:40 上午
# @Author  : Gear
import torch
from transformers import BertModel, BertTokenizer

import torch
from transformers import BertModel, BertTokenizer


def get_bert_embedding(node_list, args):
    # 这里我们调用bert-base模型，同时模型的词典经过小写处理
    model_name = 'bert-base-uncased'
    # 读取模型对应的tokenizer
    prepath = '/p300/Graph_Probe-Experiments/src/pretrain_models/' + model_name
    tokenizer = BertTokenizer.from_pretrained(prepath)
    # 载入模型
    model = BertModel.from_pretrained(prepath)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        model.to(args.gpu)
    # 输入文本
    node_embedding = []
    i = 0
    for node in node_list:
        print("embedding_size:" + str(i) + '/' +str(len(node_list)))
        i = i + 1
        input_text = node
        # 通过tokenizer把文本变成 token_id
        input_ids = tokenizer.encode(input_text, add_special_tokens=False)
        # print(input_ids)
        # input_ids: [101, 2182, 2003, 2070, 3793, 2000, 4372, 16044, 102]
        input_ids = torch.tensor([input_ids])
        if use_cuda:
            input_ids = input_ids.to(args.gpu)
        # 获得BERT模型最后一个隐层结果
        with torch.no_grad():
            last_hidden_states = model(input_ids)[0]
            last_hidden_states = last_hidden_states.squeeze(0)
            last_hidden_states = torch.mean(last_hidden_states, dim=0)
        # print(last_hidden_states.shape)
        node_embedding.append(last_hidden_states)
    bert_embedding = torch.stack(node_embedding)
    return bert_embedding

