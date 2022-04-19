from cProfile import label
import enum
import pandas as pd
import numpy as np

import stellargraph as sg
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import GCNSupervisedGraphClassification
from stellargraph import StellarGraph

from stellargraph import datasets

from sklearn import model_selection
from IPython.display import display, HTML

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt

import dgl
import pandas as pd
import csv
import numpy as np
import torch as th
import networkx as nx
import os

def build_hetero_graph(file):
    g = nx.Graph() # would use DGL, but kinda bad
    with open(file) as f: # for now convert to index list by hand
        network_data=[tuple(line) for line in csv.reader(f)]
    network_data=network_data[1:] # skip data

    # # get the max index. THIS IS NOW WRONG, GETS some high number i think. BUT IT ACTUALLY WORKS
    # n_nodes = int(max(max(network_data))) + 1 # super clever :P . max returns str i guess
    # #            ^     ^
    # #    max of tpl    max tpl of list    ^ convert from index to len
    df = pd.read_csv(file)
    n_nodes = df[['id.orig_h', 'conn', 'id.resp_h']].max().max()+1 # use df, conn should be highest row. good for now
    print(n_nodes)

    # best way?
    for i in range(n_nodes):
        g.add_node(i, feature=[1])

    edges = []
    cnts = []
    mask = []
    node_labels = {}
    node_colors = [None] * n_nodes # do here?. best way i think
    node_features = {}
    print(len(node_colors))
    for str_tup in network_data: # pandas might have a way to do this
        edges.append(tuple((int(str_tup[0]), int(str_tup[1]))))
        edges.append(tuple((int(str_tup[1]), int(str_tup[2]))))
        if str_tup[3] == 1.0: # if malicious, change threshold
            mask.append(1)
            mask.append(1)
        else:
            mask.append(0)
            mask.append(0)
        cnts.append(int(str_tup[4]))
        cnts.append(int(str_tup[4])) # for each edge

        node_labels[int(str_tup[0])] = 'ip' # smart
        node_colors[int(str_tup[0])] = [[.7, .7, .7]]
        node_features[int(str_tup[0])] = [1]
        node_labels[int(str_tup[1])] = 'conn'
        node_colors[int(str_tup[1])] = [[0, 1, 0]]
        node_features[int(str_tup[1])] = [str_tup[3]]
        node_labels[int(str_tup[2])] = 'ip'
        node_colors[int(str_tup[2])] = [[.7, .7, .7]]
        node_features[int(str_tup[0])] = [1]
    src, dst = tuple(zip(*edges))

    # get colors for a heatmap
    max_cnt = max(cnts)
    heatmap = [] # good, has to be dict, somehow, after init, the edges get shuffled, best fix is dict
    weights = []
    edge_colors = {}

    # create heat map
    for i in range(len(edges)):
        heatmap.append([cnts[i]/max_cnt, 0, 0]) # create a shade of red, based on how often this one appears relative to most common
        weights.append(cnts[i]/max_cnt)
        # ie if max_cnt is 30 and this is 15, then this color will be [127.5, 0, 0], 0<c<1
        # .: black < amount < red
    
    for i, conn in enumerate(edges):
        edge_colors[conn] = heatmap[i]

    weighted_edges = zip(src, dst, weights)
    g.add_weighted_edges_from(weighted_edges)
    # nx.set_node_attributes(g, )
    nx.set_node_attributes(g, node_labels, "label")
    nx.set_node_attributes(g, node_features, "feature")
    nx.set_edge_attributes(g, edge_colors, "color") # for the heatmap

    return (g, heatmap, node_colors)

# ///////////// LOAD INTO NETWORKX ///////////////////
print('creating networks')
files = ['conn.log.labeled_reformatted2.csv']

nx_Gs = []
_heatmaps = [] # find way to make one list
_graph_node_colors = []
for file in files:
    nx_G, _heatmap, _node_colors = build_hetero_graph(file) # outside function
    nx_Gs.append(nx_G)
    _heatmaps.append(_heatmap)
    _graph_node_colors.append(_node_colors)

# Since the actual graph is undirected, we convert it for visualization
# purpose.
# Kamada-Kawaii layout usually looks pretty for arbitrary graphs
pos = nx.kamada_kawai_layout(nx_Gs[0])
edge_colors = nx.get_edge_attributes(nx_Gs[0], 'color').values() # smart i thinkk, can get rid of other stuff
nx.draw(nx_Gs[0], pos, with_labels=True, node_color=_graph_node_colors[0], edge_color=edge_colors)
# edge weight labels
# edge_labels = nx.get_edge_attributes(nx_G, 'weight') # CANT WORK WITH MULTIGRAPH
# nx.draw_networkx_edge_labels(nx_G, pos, edge_labels)
# nx.drawing.nx_pydot.write_dot(nx_G, 'multi.dot')
import matplotlib.pyplot as plt
plt.show()

# ///////////////// LOAD GRAPHS /////////////////
print('Loading graphs')

# /////////////// get features /////////////////
graphs = []
for nx_G in nx_Gs:
    g_feature_attr = nx_G.copy()
    for node_id, node_data in g_feature_attr.nodes(data=True):
        print(node_data)
    graphs.append(StellarGraph.from_networkx(g_feature_attr, node_features="feature"))

#graphs, graph_labels = dataset.load()
print(graphs[0].info())

summary = pd.DataFrame(
    [(g.number_of_nodes(), g.number_of_edges()) for g in graphs],
    columns=["nodes", "edges"],
)
summary.describe().round(1)