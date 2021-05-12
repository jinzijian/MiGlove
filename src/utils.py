import numpy as np
import os
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