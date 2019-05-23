import os.path as osp

import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, DeepGraphInfomax
import gen.data as datagen 
from utils import embed_arr_2_dict
from openne.classify import Classifier, read_node_label
from sklearn.linear_model import LogisticRegression
import sys
dataset = 'Cora'
path = sys.argv[1]
classifydir = sys.argv[2]
dataset = Planetoid(path, dataset)


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=True)
        self.prelu = nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        return x


def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeepGraphInfomax(
    hidden_channels=128,
    encoder=Encoder(dataset.num_features, 128),
    summary=lambda z, *args, **kwargs: z.mean(dim=0),
    corruption=corruption).to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train():
    model.train()
    optimizer.zero_grad()
    pos_z, neg_z, summary = model(data.x, data.edge_index)
    loss = model.loss(pos_z, neg_z, summary)
    loss.backward()
    optimizer.step()
    return loss.item()


# def test():
#     model.eval()
#     z, _, _ = model(data.x, data.edge_index)
#     acc = model.test(
#         z[data.train_mask],
#         data.y[data.train_mask],
#         z[data.test_mask],
#         data.y[data.test_mask],
#         max_iter=150)
#     return acc


for epoch in range(1, 301):
    loss = train()
    X, Y = read_node_label(classifydir +'_labels.txt')
    embedding, _, _ = model(data.x, data.edge_index)
    vectors = embed_arr_2_dict(embedding.detach().numpy(), G)
            clf = Classifier(vectors=vectors, clf=LogisticRegression(solver="lbfgs", max_iter=4000))
            scores = clf.split_train_evaluate(X, Y,0.5)
    print(loss, scores)
