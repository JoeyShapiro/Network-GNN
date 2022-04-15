import pandas as pd
import numpy as np

import stellargraph as sg
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import DeepGraphCNN
from stellargraph import StellarGraph

from stellargraph import datasets

from sklearn import model_selection
from IPython.display import display, HTML

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten
from tensorflow.keras.losses import binary_crossentropy
import tensorflow as tf

import dgl
import pandas as pd
import csv
import numpy as np
import torch as th
import networkx as nx

# /////////////////////// CREATE GRAPHS FUNCTION ///////////////////
def build_graph(file):
    g = nx.Graph() # would use DGL, but kinda bad
    with open(file) as f: # for now convert to index list by hand
        network_data=[tuple(line) for line in csv.reader(f)]
    network_data=network_data[1:] # skip the labels

    edges = []
    cnts = []
    mask = []
    for str_tup in network_data:
        edges.append(tuple((int(str_tup[0]), int(str_tup[1]))))
        if str_tup[2] == 1.0: # if malicious, change threshold
            mask.append(1)
        else:
            mask.append(0)
        cnts.append(int(str_tup[3]))
    src, dst = tuple(zip(*edges))

    # get colors for a heatmap
    max_cnt = max(cnts)
    heatmap = [] # good
    weights = []

    # create heat map
    for i in range(len(edges)):
        heatmap.append([cnts[i]/max_cnt, 0, 0]) # create a shade of red, based on how often this one appears relative to most common
        weights.append(cnts[i]/max_cnt)
        # ie if max_cnt is 30 and this is 15, then this color will be [127.5, 0, 0], 0<c<1
        # .: black < amount < red

    weighted_edges = zip(src, dst, weights)
    g.add_weighted_edges_from(weighted_edges)

    return g, heatmap

# ///////////// LOAD INTO NETWORKX ///////////////////
nx_G, heatmap = build_graph('./conn.log.labeled_formatted2.csv') # outside function

# Since the actual graph is undirected, we convert it for visualization
# purpose.
# Kamada-Kawaii layout usually looks pretty for arbitrary graphs
pos = nx.kamada_kawai_layout(nx_G)
nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]], edge_color=heatmap)
# edge weight labels
# edge_labels = nx.get_edge_attributes(nx_G, 'weight') # CANT WORK WITH MULTIGRAPH
# nx.draw_networkx_edge_labels(nx_G, pos, edge_labels)
# nx.drawing.nx_pydot.write_dot(nx_G, 'multi.dot')
import matplotlib.pyplot as plt
plt.show()

# ///////////////// LOAD GRAPHS /////////////////
dataset = datasets.PROTEINS()
display(HTML(dataset.description))
nx_G2, _heatmap = build_graph('./conn.log.labeled_formatted2.csv')
graphs = [ StellarGraph.from_networkx(nx_G), StellarGraph.from_networkx(nx_G2) ]
graph_labels = pd.Series([ 2, 1 ], copy=False)
#graphs, graph_labels = dataset.load()

summary = pd.DataFrame(
    [(g.number_of_nodes(), g.number_of_edges()) for g in graphs],
    columns=["nodes", "edges"],
)
summary.describe().round(1)

graph_labels = pd.get_dummies(graph_labels, drop_first=True)
generator = PaddedGraphGenerator(graphs=graphs)

k = 35  # the number of rows for the output tensor
layer_sizes = [32, 32, 32, 1]

dgcnn_model = DeepGraphCNN(
    layer_sizes=layer_sizes,
    activations=["tanh", "tanh", "tanh", "tanh"],
    k=k,
    bias=False,
    generator=generator,
)
x_inp, x_out = dgcnn_model.in_out_tensors()

x_out = Conv1D(filters=16, kernel_size=sum(layer_sizes), strides=sum(layer_sizes))(x_out)
x_out = MaxPool1D(pool_size=2)(x_out)

x_out = Conv1D(filters=32, kernel_size=5, strides=1)(x_out)

x_out = Flatten()(x_out)

x_out = Dense(units=128, activation="relu")(x_out)
x_out = Dropout(rate=0.5)(x_out)

predictions = Dense(units=1, activation="sigmoid")(x_out)

model = Model(inputs=x_inp, outputs=predictions)

model.compile(
    optimizer=Adam(lr=0.0001), loss=binary_crossentropy, metrics=["acc"],
)

# ////////////// TRAIN //////////////////
train_graphs, test_graphs = model_selection.train_test_split(
    graph_labels, train_size=0.9, test_size=None, stratify=graph_labels,
)

gen = PaddedGraphGenerator(graphs=graphs)

train_gen = gen.flow(
    list(train_graphs.index - 1),
    targets=train_graphs.values,
    batch_size=50,
    symmetric_normalization=False,
)

test_gen = gen.flow(
    list(test_graphs.index - 1),
    targets=test_graphs.values,
    batch_size=1,
    symmetric_normalization=False,
)

epochs = 10

history = model.fit(
    train_gen, epochs=epochs, verbose=1, validation_data=test_gen, shuffle=True,
)

sg.utils.plot_history(history)

test_metrics = model.evaluate(test_gen)
print("\nTest Set Metrics:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))