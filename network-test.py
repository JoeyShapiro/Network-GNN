import dgl
import pandas as pd
import csv
import numpy as np
import torch as th

def build_graph():
    g = dgl.DGLGraph()
    with open('/Volumes/T7 Touch/ITS472/project 2/opt/Malware-Project/BigDataset/IoTScenarios/CTU-Honeypot-Capture-4-1/bro/conn_formatted.csv') as f: # for now convert to index list by hand
        network_data=[tuple(line) for line in csv.reader(f)]
    network_data=network_data[1:] # skip the labels

    # get the max index
    n_nodes = int(max(max(network_data))) + 1 # super clever :P . max returns str i guess
    #            ^     ^
    #    max of tpl    max tpl of list    ^ convert from index to len
    print(n_nodes)

    edges = []
    cnts = []
    for str_tup in network_data:
        edges.append(tuple((int(str_tup[0]), int(str_tup[1]))))
        cnts.append(int(str_tup[2]))
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