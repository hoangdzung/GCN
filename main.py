
import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, DeepGraphInfomax

from models import GCNNet, GATNet, SGNet, SAGENet
from loss import n2v_loss, edge_balance_loss
import gen.data as datagen
import argparse
import networkx as nx 
from sklearn.linear_model import LogisticRegression

from openne.classify import Classifier, read_node_label
from utils import process_graph, embed_arr_2_dict, corruption
import sys 
import pdb
from tqdm import tqdm


torch.manual_seed(0)
def main(args):
    dataset = 'Cora'
    path = args.datapath
    dataset = Planetoid(path, dataset)
    dd = dataset[0]
    # G = datagen.load_data(args.classifydir, True)
    # X, Y = read_node_label(args.classifydir +'_labels.txt')
    
    # attr_matrix, adj, edge_index = process_graph(G)
    # attr_matrix = torch.LongTensor(attr_matrix)
    # adj = torch.LongTensor(adj)
    # edge_index = torch.LongTensor(edge_index)

    attr_matrix = dataset[0].x 
    edge_index = dataset[0].edge_index
    

    adj = torch.zeros((attr_matrix.shape[0],attr_matrix.shape[0]))
    G = nx.from_numpy_matrix(adj.detach().cpu().numpy())
    nodes = torch.Tensor(list(G.nodes()))
    
    adj[edge_index] = 1
    if args.net_type == 'gcn':
        model = GCNNet(attr_matrix.shape[1], args.embedding_size)
    elif args.net_type == 'gat':
        model = GATNet(attr_matrix.shape[1], args.embedding_size)
    elif args.net_type == 'sage':
        model = SAGENet(attr_matrix.shape[1], args.embedding_size)
    elif args.net_type == 'sg':
        model = SGNet(attr_matrix.shape[1], args.embedding_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)
    if args.use_cuda:
        attr_matrix, adj, edge_index = attr_matrix.cuda(), adj.cuda(), edge_index.cuda()
        model = model.cuda()
    if args.loss_type == 'n2v':
        loss_fn = n2v_loss
    elif args.loss_type == "edge":
        loss_fn = edge_balance_loss


    best_score = 0
    best_clf = None

    for i in tqdm(range(args.n_epochs)):

        model.train()
        optimizer.zero_grad()
        emebedding = model(attr_matrix, edge_index)
        loss = loss_fn(emebedding, adj)
        loss.backward()
        optimizer.step()
        if i %100 == 0:
            X_train = list(map(str, map(int, nodes[dd.test_mask].detach().cpu().numpy())))
            y_train = list(map(str, map(int,dd.y[dd.test_mask].detach().cpu().numpy())))
            X_test = list(map(str, map(int,nodes[dd.val_mask].detach().cpu().numpy())))
            y_test = list(map(str, map(int,dd.y[dd.val_mask].detach().cpu().numpy())))
            Y = list(map(str, map(int, dd.y.detach().cpu().numpy().tolist())))

            vectors = embed_arr_2_dict(emebedding.detach().numpy(), G)
            clf = Classifier(vectors=vectors, clf=LogisticRegression(solver="lbfgs", max_iter=4000))
            scores = clf.train_evaluate(X_train, y_train, X_test, y_test, Y)
            print(i, loss.detach().cpu().numpy(), scores)

            if scores['micro'] > best_score:
                best_score = scores['micro']
                best_clf = clf
    # pdb.set_trace()
    X_val = list(map(str, map(int,nodes[dd.train_mask].detach().cpu().numpy())))
    y_val = list(map(str, map(int,dd.y[dd.train_mask].detach().cpu().numpy())))
    print("Testing ", best_clf.evaluate(X_val, y_val))


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
