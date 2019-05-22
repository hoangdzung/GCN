import os
import networkx as nx
import numpy as np
import random
import pdb
import gen.feat as featgen

def add_weight(subgraph):
    for n1, n2 in subgraph.edges():
        subgraph[n1][n2]['weight'] = 1
    return subgraph

def load_data(data_path, read_feature=False):
    G = nx.read_edgelist(data_path+'_edgelist.txt', nodetype=int)
    # add_weight(G)
    G.graph['label'] = 0
    if read_feature and os.path.isfile(data_path+'_features.txt'):
        with open(data_path+'_features.txt') as fp:
            for line in fp:
                vec = line.split()
                G.nodes[int(vec[0])]['feat'] = np.array([float(x) for x in vec[1:]])
    else:
        for i in range(G.number_of_nodes()):
            G.nodes[i]['feat'] = np.array([1,1,1,1])

    return G