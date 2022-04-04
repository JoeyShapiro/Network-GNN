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

    # get colors for a heatmap
    max_edges = 0
    best_node = edges[0]
    colors = [] # good
     
    # get node with most connections
    for i in edges:
        curr_frequency = edges.count(i)
        if(curr_frequency > max_edges):
            max_edges = curr_frequency
            best_node = i
        
    print("most common:", best_node, "; amount:", max_edges)
    
    # create heat map
    for e in edges:
        amount = edges.count(e)
        colors.append([amount/max_edges, 0, 0]) # create a shade of red, based on how often this one appears relative to most common
        # ie if max_edges is 30 and this is 15, then this color will be [127.5, 0, 0], 0<c<1
        # .: black < amount < red

    return g, colors

G, colors = build_graph() # outside function
print('We have %d nodes.' % G.number_of_nodes())
print('We have %d edges.' % G.number_of_edges())

import networkx as nx
# Since the actual graph is undirected, we convert it for visualization
# purpose.
nx_G = G.to_networkx().to_undirected()
# Kamada-Kawaii layout usually looks pretty for arbitrary graphs
pos = nx.kamada_kawai_layout(nx_G)
nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]], edge_color=colors)
import matplotlib.pyplot as plt
plt.show()