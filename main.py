
import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebCon

from models import GCNet
from loss import n2v_loss, edge_balance_loss
import gen.data as datagen
import argparse
import networkx as nx 

from openne.classify import Classifier, read_node_label
from utils import process_graph, embed_arr_2_dict
import sys 

def main(args):
    G = datagen.load_data(args.classifydir, True)
    X, Y = read_node_label(args.classifydir +'_labels.txt')
    attr_matrix, adj, edge_index = process_graph(G)
    attr_matrix = torch.FloatTensor(attr_matrix)
    adj = torch.FloatTensor(adj)
    edge_index = torch.FloatTensor(edge_index)

    model = GCNet(X.shape[1], args.embedding_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    if args.use_cuda:
        attr_matrix, adj, edge_index = attr_matrixX.cuda(), adj.cuda(), edge_index.cuda()
        model = model.cuda()
    if args.loss_type == 'n2v':
        loss_fn = n2v_loss()
    elif args.loss_type == "edge":
        loss_fn = edge_balance_loss()
    else:
        raise Exception, "must specify loss type"
    for i in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        emebedding = model(attr_matrix, edge_index)
        loss = loss_fn(emebedding, adj).backward()
        loss.backward()
        optimizer.step()
        if i %5 = 0:
            vectors = embed_arr_2_dict(emebedding.numpy(), G)
            clf = Classifier(vectors=vectors, clf=LogisticRegression(solver="lbfgs", max_iter=4000))
            scores = clf.split_train_evaluate(X, Y,0.5)
            print(scores)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true',
                        help='Using GPU or not')
    parser.add_argument('--classifydir', dest='classifydir',
            help='Directory containing graph classify data')
    parser.add_argument('--loss_type', ,
            help='n2v or edge')
    parser.add_argument('--embedding_dim', type=int,
                        help='Dimension of node embeddings, default=128', default=128)
    parser.add_argument('--n_epochs', type=int,
                        help='Number of training step, default = 3000', default=3000)
                    
    return parser.parse_args(argv)