from cProfile import label
import enum
import pandas as pd
import numpy as np

import stellargraph as sg
import stellargraph
from stellargraph.mapper import PaddedGraphGenerator
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
    print('Loading graph from', file)
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
        cnts.append(int(str_tup[5]))
        cnts.append(int(str_tup[5])) # for each edge

        node_labels[int(str_tup[0])] = 'node' # smart
        node_colors[int(str_tup[0])] = [[.7, .7, .7]]
        node_features[int(str_tup[0])] = {'type': 'ip', 'feature': [1]}
        node_labels[int(str_tup[1])] = 'node' #'conn'
        node_colors[int(str_tup[1])] = [[0, 1, 0]]
        node_features[int(str_tup[1])] = {'type': 'conn', 'feature': [str_tup[3]]} # 1 is ip 2 is conn
        node_labels[int(str_tup[2])] = 'node'
        node_colors[int(str_tup[2])] = [[.7, .7, .7]]
        node_features[int(str_tup[2])] = {'type': 'ip', 'feature': [1]}
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
    nx.set_node_attributes(g, node_features)
    nx.set_edge_attributes(g, edge_colors, "color") # for the heatmap

    return (g, heatmap, node_colors)

# ///////////// LOAD INTO NETWORKX ///////////////////
print('creating networks')
dataset_path = '/Volumes/T7 Touch/ITS472/project 2/dataset'
mal_set = dataset_path+'/mal/'
bon_set = dataset_path+'/bon/'
labels = []
files = []
for file in os.listdir(mal_set): # fine cleaner way
    files.append(mal_set+file) # keeps filename only
    labels.append(2)
for file in os.listdir(bon_set):
    files.append(bon_set+file)
    labels.append(1)

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
    print('Loading new graph from networkx')
    for node_id, node_data in g_feature_attr.nodes(data=True):
        print(node_data)
    graphs.append(StellarGraph.from_networkx(g_feature_attr, node_features="feature"))

graph_labels = pd.Series(labels, copy=False) # 2 = mal, 1 = good, i think
#graphs, graph_labels = dataset.load()
print(graphs[0].info())

summary = pd.DataFrame(
    [(g.number_of_nodes(), g.number_of_edges()) for g in graphs],
    columns=["nodes", "edges"],
)
summary.describe().round(1)

graph_labels = pd.get_dummies(graph_labels, drop_first=True)

# /////////// GENERATING ///////////////
print('Generating')
generator = PaddedGraphGenerator(graphs=graphs)

# ///////////////////////////////////////////// DGCNN //////////////////////////
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

# ////////////// KERAS MODEL ///////////////////
print('Creating keras model')
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
print('training')
train_graphs, test_graphs = model_selection.train_test_split(
    graph_labels, train_size=0.5, test_size=2, stratify=graph_labels,
)

gen = PaddedGraphGenerator(graphs=graphs)

train_gen = gen.flow( # should this be gen or generator
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

epochs = 100

history = model.fit( # changing verbose over 2 removes cool ui
    train_gen, epochs=epochs, verbose=2, validation_data=test_gen, shuffle=False,
)

sg.utils.plot_history(history)

test_metrics = model.evaluate(test_gen)
print("\nTest Set Metrics:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

from sklearn import preprocessing, model_selection
all_gen = generator.flow(graphs)
all_predictions = model.predict(all_gen)
target_encoding = preprocessing.LabelBinarizer()
target_encoding.fit(graph_labels)
graph_predictions = target_encoding.inverse_transform(all_predictions.squeeze())
df = pd.DataFrame({"Predicted": graph_predictions, "True": graph_labels[2].values.tolist()})
print(df.head(20))

# SAVING
print("saving model")
tf.saved_model.save(model, './first.mdl')

# load
print('loading model')
imported = tf.saved_model.load('./first.mdl')