import dgl
import pandas as pd
import csv
import numpy as np
import torch as th

# static variables
k = 10 # number of epochs is k, right?

def build_graph():
    g = dgl.DGLGraph()
    with open('./conn.log.labeled_formatted2.csv') as f: # for now convert to index list by hand
        network_data=[tuple(line) for line in csv.reader(f)]
    network_data=network_data[1:] # skip the labels

    # get the max index
    n_nodes = int(max(max(network_data))) + 1 # super clever :P . max returns str i guess
    #            ^     ^
    #    max of tpl    max tpl of list    ^ convert from index to len
    print(n_nodes)

    edges = []
    cnts = []
    mask = []
    for str_tup in network_data:
        edges.append(tuple((int(str_tup[0]), int(str_tup[1]))))
        if str_tup[2] == 'Malicious':
            mask.append(1)
        else:
            mask.append(0)
        cnts.append(int(str_tup[3]))
    g.add_nodes(n_nodes) # essentially the same
    src, dst = tuple(zip(*edges))
    g.add_edges(src, dst)

    # add features
    th_cnts = th.tensor(cnts)
    g.edata['w'] = th_cnts
    print(th_cnts)

    # get colors for a heatmap
    max_cnt = max(cnts)
    colors = [] # good

    # create heat map
    for amount in cnts:
        colors.append([amount/max_cnt, 0, 0]) # create a shade of red, based on how often this one appears relative to most common
        # ie if max_cnt is 30 and this is 15, then this color will be [127.5, 0, 0], 0<c<1
        # .: black < amount < red

    # synthetic node and edge features, as well as edge labels
    g.ndata['feature'] = th.ones(10)
    g.edata['label'] = th_cnts
    # synthetic train-validation-test splits
    g.edata['train_mask'] = th.tensor(mask) # th.zeros(15, dtype=th.bool).bernoulli(0.6)

    return g, colors, cnts

G, colors, weights = build_graph() # outside function
print('We have %d nodes.' % G.number_of_nodes())
print('We have %d edges.' % G.number_of_edges())

import networkx as nx
# Since the actual graph is undirected, we convert it for visualization
# purpose.
nx_G = G.to_networkx().to_undirected()
i = 0
for src, dst in nx_G.edges():
    nx_G[src][dst][0]['label'] = weights[i]
    i+=1 # good ol fashioned incrementor
# Kamada-Kawaii layout usually looks pretty for arbitrary graphs
pos = nx.kamada_kawai_layout(nx_G)
nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]], edge_color=colors)
# edge weight labels
# edge_labels = nx.get_edge_attributes(nx_G, 'weight') # CANT WORK WITH MULTIGRAPH
# nx.draw_networkx_edge_labels(nx_G, pos, edge_labels)
nx.drawing.nx_pydot.write_dot(nx_G, 'multi.dot')
import matplotlib.pyplot as plt
plt.show()

# Contruct a two-layer GNN model
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h

class DotProductPredictor(nn.Module):
    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return graph.edata['score']

class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = self.W(th.cat([h_u, h_v], 1))
        return {'score': score}

    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']

class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.sage = SAGE(in_features, hidden_features, out_features)
        self.pred = DotProductPredictor()
    def forward(self, g, x):
        h = self.sage(g, x)
        return self.pred(g, h)

node_features = G.ndata['feature']
edge_label = G.edata['label']
train_mask = G.edata['train_mask']
model = Model(10, 20, 10) # 300 hidden layers gave a good loss
opt = th.optim.Adam(model.parameters())
for epoch in range(150):
    pred = model(G, node_features)
    loss = ((pred[train_mask] - edge_label[train_mask]) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(loss.item())

# save and load and stuff model
import os
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info

def save(self):
    # save graphs and labels
    graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph.bin')
    save_graphs(graph_path, self.graphs, {'labels': self.labels})
    # save other information in python dict
    info_path = os.path.join(self.save_path, self.mode + '_info.pkl')
    save_info(info_path, {'num_classes': self.num_classes})

def load(self):
    # load processed data from directory `self.save_path`
    graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph.bin')
    self.graphs, label_dict = load_graphs(graph_path)
    self.labels = label_dict['labels']
    info_path = os.path.join(self.save_path, self.mode + '_info.pkl')
    self.num_classes = load_info(info_path)['num_classes']

def has_cache(self):
    # check whether there are processed data in `self.save_path`
    graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph.bin')
    info_path = os.path.join(self.save_path, self.mode + '_info.pkl')
    return os.path.exists(graph_path) and os.path.exists(info_path)