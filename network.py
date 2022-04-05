import dgl
import pandas as pd
import csv
import numpy as np
import torch as th

def build_graph():
    g = dgl.DGLGraph()
    with open('./formatted_bfwppTest.csv') as f: # for now convert to index list by hand
        network_data=[tuple(line) for line in csv.reader(f)]
    network_data=network_data[1:]

    # get the max index
    n_nodes = int(max(max(network_data))) + 1 # super clever :P . max returns str i guess
    #            ^     ^
    #    max of tpl    max tpl of list    ^ convert from index to len
    print(n_nodes)

    edges = []
    dst_ports = []
    src_ports = []
    for str_tup in network_data:
        edges.append(tuple((int(str_tup[0]), int(str_tup[1]))))
        # get dst port
        if (str_tup[2].isdigit()):
            dst_ports.append(int(str_tup[2])) # the dst port
        else: # if doesnt have a port
            dst_ports.append(-1) # dummy
        # get src port
        if (str_tup[3].isdigit()):
            src_ports.append(int(str_tup[3])) # the src port
        else: # if doesnt have a port
            src_ports.append(-1) # dummy
    g.add_nodes(n_nodes) # essentially the same
    src, dst = tuple(zip(*edges))
    g.add_edges(src, dst)
    g.add_edges(dst, src) # i dont think this needed

    # add features
    ports = dst_ports + src_ports # combined of ports, i do dst+src ports to match src+dst ips, is this right, always go to end edge port
    th_ports = th.tensor(ports)
    g.edata['port'] = th_ports
    print(g.edata['port'])

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