#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/1 3:40 上午
# @Author  : Gear
import torch
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import torch
from transformers import BertModel, BertTokenizer


def get_bert_embedding(lines, args):
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
    bert_embedding = []
    i = 0
    for line in tqdm(lines):
        i = i + 1
        input_text = line
        # 通过tokenizer把文本变成 token_id
        mask = []
        input_ids = tokenizer.encode(input_text, add_special_tokens=False)
        input_text = line.split(' ')
        for word in input_text:
            input_id = tokenizer.encode(word, add_special_tokens=False)
            if(len(input_id) != 0):
                mask.append(input_ids.index(input_id[0]))
        # print(input_ids)
        # input_ids: [101, 2182, 2003, 2070, 3793, 2000, 4372, 16044, 102]
        input_ids = torch.tensor([input_ids])
        if use_cuda:
            input_ids = input_ids.to(args.gpu)
        # 获得BERT模型最后一个隐层结果
        with torch.no_grad():
            last_hidden_states = model(input_ids)[0]
            last_hidden_states = last_hidden_states.squeeze(0)
            #last_hidden_states = torch.mean(last_hidden_states, dim=0)
        # print(last_hidden_states.shape)
            states = []
            for i in range(len(mask)):
                states.append(last_hidden_states[mask[i]])
        bert_embedding.append(states)
    return bert_embedding

