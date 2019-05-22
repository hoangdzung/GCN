
import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, DeepGraphInfomax

from models import GCNNet, GATNet
from loss import n2v_loss, edge_balance_loss
import gen.data as datagen
import argparse
import networkx as nx 
from sklearn.linear_model import LogisticRegression

from openne.classify import Classifier, read_node_label
from utils import process_graph, embed_arr_2_dict, corruption
import sys 

torch.manual_seed(0)
def main(args):
    dataset = 'Cora'
    path = args.datapath
    dataset = Planetoid(path, dataset)
    G = datagen.load_data(args.classifydir, True)
    X, Y = read_node_label(args.classifydir +'_labels.txt')
    # attr_matrix, adj, edge_index = process_graph(G)
    # attr_matrix = torch.LongTensor(attr_matrix)
    # adj = torch.LongTensor(adj)
    # edge_index = torch.LongTensor(edge_index)

    attr_matrix = dataset[0].x 
    edge_index = dataset[0].edge_index

    adj = torch.zeros((attr_matrix.shape[0],attr_matrix.shape[0]))
    adj[edge_index] = 1
    if args.net_type == 'gcn':
        model = GCNNet(attr_matrix.shape[1], args.embedding_size)
    elif args.net_type == 'gat':
        model = GATNet(attr_matrix.shape[1], args.embedding_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)
    if args.use_cuda:
        attr_matrix, adj, edge_index = attr_matrix.cuda(), adj.cuda(), edge_index.cuda()
        model = model.cuda()
    if args.loss_type == 'n2v':
        loss_fn = n2v_loss
    elif args.loss_type == "edge":
        loss_fn = edge_balance_loss

    for i in range(args.n_epochs):
        model.train()
        optimizer.zero_grad()
        emebedding = model(attr_matrix, edge_index)
        loss = loss_fn(emebedding, adj)
        loss.backward()
        optimizer.step()
        if i %5 == 0:
            vectors = embed_arr_2_dict(emebedding.detach().numpy(), G)
            clf = Classifier(vectors=vectors, clf=LogisticRegression(solver="lbfgs", max_iter=4000))
            scores = clf.split_train_evaluate(X, Y,0.5)
            print(loss, scores)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true',
                        help='Using GPU or not')
    parser.add_argument('--datapath',)
    parser.add_argument('--classifydir', dest='classifydir',
            help='Directory containing graph classify data')
    parser.add_argument('--loss_type', 
            help='n2v or edge')
    parser.add_argument('--net_type', 
            help='gcn or eat')
    parser.add_argument('--embedding_size', type=int,
                        help='Dimension of node embeddings, default=128', default=128)
    parser.add_argument('--n_epochs', type=int,
                        help='Number of training step, default = 3000', default=3000)
                    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
