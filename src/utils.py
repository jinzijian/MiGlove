import numpy as np
import os
import torch
import random

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
import itertools
import math
import random

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import trange
from alias import alias_sample, create_alias_table
def partition_num(num, workers):
    if num % workers == 0:
        return [num//workers]*workers
    else:
        return [num//workers]*workers + [num % workers]

def random_walks(G, num_walks=100, walk_len=10, string_nid=False):
    paths = []
    # add self loop
    for nid in G.nodes(): G.add_edge(nid, nid)
    if not string_nid:
        for nid in G.nodes():
            if G.degree(nid) == 0: continue
            for i in range(num_walks):
                tmp_path = [str(nid)]
                for j in range(walk_len):
                    neighbors = [str(n) for n in G.neighbors(int(tmp_path[-1]))]
                    tmp_path.append(random.choice(neighbors))
                paths.append(tmp_path)
    else:
        for nid in G.nodes():
            if G.degree(nid) == 0: continue
            for i in range(num_walks):
                tmp_path = [nid]
                for j in range(walk_len):
                    neighbors = [n for n in G.neighbors(tmp_path[-1])]
                    tmp_path.append(random.choice(neighbors))
                paths.append(tmp_path)

    return paths


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True