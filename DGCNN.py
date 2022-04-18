from cProfile import label
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

# /////////////////////// CREATE GRAPHS FUNCTION ///////////////////
def build_graph(file):
    g = nx.Graph() # would use DGL, but kinda bad
    with open(file) as f: # for now convert to index list by hand
        network_data=[tuple(line) for line in csv.reader(f)]
    network_data=network_data[1:] # skip the labels

    # get the max index
    n_nodes = int(max(max(network_data))) + 1 # super clever :P . max returns str i guess
    #            ^     ^
    #    max of tpl    max tpl of list    ^ convert from index to len
    print(n_nodes)

    # best way?
    for i in range(n_nodes):
        g.add_node(i, feature=1)

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
    # nx.set_node_attributes(g, )

    return (g, heatmap)

# ///////////// LOAD INTO NETWORKX ///////////////////
print('creating networks')
dataset_path = '/Volumes/T7 Touch/ITS472/project 2/dataset'
mal_set = dataset_path+'/mal'
bon_set = dataset_path+'/bon'
labels = []
files = []
for file in os.listdir(mal_set): # fine cleaner way
    files.append(file)
    labels.append(2)
for file in os.listdir(bon_set):
    files.append(file)
    labels.append(1)

nx_Gs = []
_heatmaps = [] # find way to make one list
for file in files:
    nx_G, _heatmap = build_graph(file) # outside function
    nx_Gs.append(nx_G)
    _heatmaps.append(_heatmap)

# Since the actual graph is undirected, we convert it for visualization
# purpose.
# Kamada-Kawaii layout usually looks pretty for arbitrary graphs
pos = nx.kamada_kawai_layout(nx_Gs[0])
nx.draw(nx_Gs[0], pos, with_labels=True, node_color=[[.7, .7, .7]], edge_color=_heatmaps[0])
# edge weight labels
# edge_labels = nx.get_edge_attributes(nx_G, 'weight') # CANT WORK WITH MULTIGRAPH
# nx.draw_networkx_edge_labels(nx_G, pos, edge_labels)
# nx.drawing.nx_pydot.write_dot(nx_G, 'multi.dot')
import matplotlib.pyplot as plt
plt.show()

# ///////////////// LOAD GRAPHS /////////////////
print('Loading graphs')

def compute_features(node_id):
    # in general this could compute something based on other features, but for this example,
    # we don't have any other features, so we'll just do something basic with the node_id
    return [int(node_id), 1]

# /////////////// get features /////////////////
graphs = []
for nx_G in nx_Gs:
    g_feature_attr = nx_G.copy()
    for node_id, node_data in g_feature_attr.nodes(data=True):
        node_data["feature"] = compute_features(node_id)
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
# # ///////////////////////////////////////////////// GCN ///////////////////////////////
# # /////////////// CREATING MODEL ////////////////
# def create_graph_classification_model(generator):
#     gc_model = GCNSupervisedGraphClassification(
#         layer_sizes=[64, 64],
#         activations=["relu", "relu"],
#         generator=generator,
#         dropout=0.5,
#     )
#     x_inp, x_out = gc_model.in_out_tensors()
#     predictions = Dense(units=32, activation="relu")(x_out)
#     predictions = Dense(units=16, activation="relu")(predictions)
#     predictions = Dense(units=1, activation="sigmoid")(predictions)

#     # Let's create the Keras model and prepare it for training
#     model = Model(inputs=x_inp, outputs=predictions)
#     model.compile(optimizer=Adam(0.005), loss=binary_crossentropy, metrics=["acc"])

#     return model

# # //////////// TRAINING ///////////////
# epochs = 200  # maximum number of training epochs
# folds = 2  # the number of folds for k-fold cross validation
# n_repeats = 5  # the number of repeats for repeated k-fold cross validation

# es = EarlyStopping(
#     monitor="val_loss", min_delta=0, patience=25, restore_best_weights=True
# )

# def train_fold(model, train_gen, test_gen, es, epochs):
#     history = model.fit(
#         train_gen, epochs=epochs, validation_data=test_gen, verbose=0, callbacks=[es],
#     )
#     # calculate performance on the test data and return along with history
#     test_metrics = model.evaluate(test_gen, verbose=0)
#     test_acc = test_metrics[model.metrics_names.index("acc")]

#     return history, test_acc

# def get_generators(train_index, test_index, graph_labels, batch_size):
#     train_gen = generator.flow(
#         train_index, targets=graph_labels.iloc[train_index].values, batch_size=batch_size
#     )
#     test_gen = generator.flow(
#         test_index, targets=graph_labels.iloc[test_index].values, batch_size=batch_size
#     )

#     return train_gen, test_gen

# test_accs = []

# stratified_folds = model_selection.RepeatedStratifiedKFold(
#     n_splits=folds, n_repeats=n_repeats
# ).split(graph_labels, graph_labels)

# for i, (train_index, test_index) in enumerate(stratified_folds):
#     print(f"Training and evaluating on fold {i+1} out of {folds * n_repeats}...")
#     train_gen, test_gen = get_generators(
#         train_index, test_index, graph_labels, batch_size=30
#     )

#     model = create_graph_classification_model(generator)

#     history, acc = train_fold(model, train_gen, test_gen, es, epochs)

#     test_accs.append(acc)

# print(
#     f"Accuracy over all folds mean: {np.mean(test_accs)*100:.3}% and std: {np.std(test_accs)*100:.2}%"
# )

# plt.figure(figsize=(8, 6))
# plt.hist(test_accs)
# plt.xlabel("Accuracy")
# plt.ylabel("Count")
# plt.show()

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

from sklearn import preprocessing, model_selection
all_gen = generator.flow(graphs)
all_predictions = model.predict(all_gen)
target_encoding = preprocessing.LabelBinarizer()
target_encoding.fit(graph_labels)
graph_predictions = target_encoding.inverse_transform(all_predictions.squeeze())
df = pd.DataFrame({"Predicted": graph_predictions, "True": graph_labels[2].values.tolist()})
print(df.head(20))