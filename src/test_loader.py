# coding:utf-8
import random
import sklearn
import dgl
import torch
import networkx as nx
import matplotlib.pyplot as plt

def ShowGraph(graph, nodeLabel, EdgeLabel):
    plt.figure(figsize=(8, 8))
    G = graph.to_networkx(node_attrs=nodeLabel.split(), edge_attrs=EdgeLabel.split())  # 转换 dgl graph to networks
    pos = nx.spring_layout(G)
    nx.draw(G, pos, edge_color="grey", node_size=500, with_labels=True)  # 画图，设置节点大小
    node_data = nx.get_node_attributes(G, nodeLabel)  # 获取节点的desc属性
    node_labels = {index: "N:" + str(data) for index, data in
                   enumerate(node_data)}  # 重新组合数据， 节点标签是dict, {nodeid:value,nodeid2,value2} 这样的形式
    pos_higher = {}

    for k, v in pos.items():  # 调整下顶点属性显示的位置，不要跟顶点的序号重复了
        if (v[1] > 0):
            pos_higher[k] = (v[0] - 0.04, v[1] + 0.04)
        else:
            pos_higher[k] = (v[0] - 0.04, v[1] - 0.04)
    nx.draw_networkx_labels(G, pos_higher, labels=node_labels, font_color="brown", font_size=12)  # 将desc属性，显示在节点上
    edge_labels = nx.get_edge_attributes(G, EdgeLabel)  # 获取边的weights属性，

    edge_labels = {(key[0], key[1]): "w:" + str(edge_labels[key].item()) for key in
                   edge_labels}  # 重新组合数据， 边的标签是dict, {(nodeid1,nodeid2):value,...} 这样的形式
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12)  # 将Weights属性，显示在边上

    print(G.edges.data())
    plt.show()


class NegativeSampler(object):
    def __init__(self, g, k, neg_share=False):
        self.weights = g.in_degrees().float() ** 0.75
        self.k = k
        self.neg_share = neg_share

    def __call__(self, g, eids):
        src, _ = g.find_edges(eids)
        n = len(src)
        if self.neg_share and n % self.k == 0:
            dst = self.weights.multinomial(n, replacement=True)
            dst = dst.view(-1, 1, self.k).expand(-1, self.k, -1).flatten()
        else:
            dst = self.weights.multinomial(n*self.k, replacement=True)
        src = src.repeat_interleave(self.k)
        return src, dst


u, v = torch.tensor([0, 0, 0, 1, 2, 3, 4, 5, 6]), torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
train_g = dgl.graph((u, v))
train_g.ndata['nfeats'] = torch.zeros(train_g.num_nodes(), 1)
edata = torch.tensor([[1],[2],[3],[4], [5], [6], [7], [8], [9]])
print(edata.size())
train_g.edata['efeats'] = edata
print(train_g)
#ShowGraph(train_g,'nfeats','efeats')

n_edges = train_g.num_edges()
train_seeds = torch.arange(n_edges)
sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
train_dataloader = dgl.dataloading.NodeDataLoader(
            train_g, torch.arange(train_g.number_of_nodes()), sampler,
            batch_size=3,
            shuffle=False,
            drop_last=False,
            num_workers=0)

count = 0
for input_nodes, output_nodes, blocks in train_dataloader:
    # feature copy from CPU to GPU takes place here
    count += 1
    print('loader %d times'%(count))
    print('input_nodes')
    print(input_nodes)
    print('output_nodes')
    print(output_nodes)























