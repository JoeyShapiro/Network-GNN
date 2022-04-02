import dgl
import pandas as pd
import csv

def build_graph():
    g = dgl.DGLGraph()
    with open('./formatted_bigFlowsTest.csv') as f: # for now convert to index list by hand
        network_data=[tuple(line) for line in csv.reader(f)]
    network_data=network_data[1:]

    # get the max index
    n_nodes = int(max(max(network_data))) + 1 # super clever :P . max returns str i guess
    #            ^     ^
    #    max of tpl    max tpl of list    ^ convert from index to len
    print(n_nodes)

    edges = []
    for str_tup in network_data:
        edges.append(tuple((int(str_tup[0]), int(str_tup[1]))))
    g.add_nodes(n_nodes) # essentially the same
    src, dst = tuple(zip(*edges))
    g.add_edges(src, dst)
    g.add_edges(dst, src)

    return g

G = build_graph()
print('We have %d nodes.' % G.number_of_nodes())
print('We have %d edges.' % G.number_of_edges())

import networkx as nx
# Since the actual graph is undirected, we convert it for visualization
# purpose.
nx_G = G.to_networkx().to_undirected()
# Kamada-Kawaii layout usually looks pretty for arbitrary graphs
pos = nx.kamada_kawai_layout(nx_G)
nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
import matplotlib.pyplot as plt
plt.show()