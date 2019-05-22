import networkx as nx 
import numpy as np 
import torch

def process_graph(G):
    adj = nx.adjacency_matrix(G)
    edge_index = np.array(adj.nonzero())
    X = []
    for node in range(len(G)):
        X.append(G.node[node]['feat'])
    X = np.array(X)
    return X, adj.toarray(), edge_index

def embed_arr_2_dict(embed_arr, G): ##keep
    embed_dict = {}
    for idx, node in enumerate(G.nodes()):
        embed_dict[str(node)] = embed_arr[idx]
    return embed_dict

def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index

if __name__ == "__main__":
    import gen.data as datagen
    G = datagen.load_data('./data/cora/cora',True)
    X, adj, edge_index = process_graph(G)
    import pdb
    pdb.set_trace() 