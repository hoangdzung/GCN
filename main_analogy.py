
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, DeepGraphInfomax

from models import GCNNet, GATNet, SGNet, SAGENet
from loss import n2v_loss, edge_balance_loss
import argparse
import networkx as nx 
import numpy as np
import sys 
import pdb
from tqdm import tqdm
from scipy import spatial
from sklearn.metrics import roc_auc_score

torch.manual_seed(0)
def cos_sim(embedding, x):
    return 1 - spatial.distance.cosine(embedding[x[0]]+embedding[x[1]]+embedding[x[2]],\
        embedding[x[3]]+embedding[x[4]]+embedding[x[5]])
        
def main(args):

    G = nx.read_edgelist(args.edgelist_path, nodetype=int)
    n_nodes = len(G.nodes())
    
    combinations = np.load(args.combination_file)
    linear_size = combinations.shape[1]
    combinations = combinations.tolist()
    combinations_list =  [(set(combination[:3]), set(combination[3:])) for combination in combinations]
    non_combinations = []
    while (len(non_combinations) < len(combinations)):
        non_combination = np.random.choice(np.arange(n_nodes), size=(linear_size), replace=False)
        if (set(non_combination[:3]), set(non_combination[3:])) in combinations_list \
            or (set(non_combination[3:]), set(non_combination[:3])) in combinations_list:
            continue

        non_combinations.append(non_combination.tolist())

    attr_matrix = torch.ones(n_nodes, args.attr_dim)    
    edge_index = torch.LongTensor(np.array([list(edge) for edge in G.edges()]).T)
    adj = torch.FloatTensor(nx.adjacency_matrix(G).toarray())
    
    model = SGNet(args.attr_dim, args.embedding_size, args.slope, args.temp)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    if args.use_cuda:
        attr_matrix, adj, edge_index = attr_matrix.cuda(), adj.cuda(), edge_index.cuda()
        model = model.cuda()
    if args.loss_type == 'n2v':
        loss_fn = n2v_loss
    elif args.loss_type == "edge":
        loss_fn = edge_balance_loss
    # best_loss = 1e100
    # embedd_np = None
    for i in tqdm(range(args.n_epochs)):
        model.train()
        optimizer.zero_grad()
        embedding = model(attr_matrix, edge_index)
        loss = loss_fn(embedding, adj)
        # if loss.cpu().item() < best_loss:
        #     best_loss = loss.cpu().item()
        embedd_np = embedding.detach().cpu().numpy()
        loss.backward()
        optimizer.step()
        if i % 100==0: 
            sims = []
            labels = []

            for combination in combinations:
                labels.append(1)
                sims.append(cos_sim(embedd_np, combination))

            for non_combination in non_combinations:
                labels.append(0)
                sims.append(cos_sim(embedd_np, non_combination))

            print(roc_auc_score(labels, sims))
            print(loss.cpu().item())

    np.save(args.output_file, embedd_np)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true',
                        help='Using GPU or not')
    parser.add_argument('--loss_type', 
            help='n2v or edge')
    parser.add_argument('--net_type', 
            help='gcn or eat')
    parser.add_argument('--edgelist_path', 
            help='')
    parser.add_argument('--slope',type=float, default = 0,
            help='Slope of leaky_relu')
    parser.add_argument('--temp',type=float, default = 0.1,
            help='Temp of gumbel softmax')
    parser.add_argument('--lr',type=float, default = 0.001,
            help='')
    parser.add_argument('--attr_dim', type=int,
                        help='Dimension of node attribute, default=128', default=128)
    parser.add_argument('--embedding_size', type=int,
                        help='Dimension of node embeddings, default=128', default=128)
    parser.add_argument('--n_epochs', type=int,
                        help='Number of training step, default = 3000', default=3000)
    parser.add_argument('--combination_file')                    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
