import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, ChebConv  

class GCNet(torch.nn.Module):
    def __init__(self, num_features, embedding_size=128):
        super(Net, self).__init__()
        self.conv1 = GCNConv(num_features, embedding_size*2, cached=True)
        self.conv2 = GCNConv(embedding_size*2, embedding_size, cached=True)

    def forward(self,x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x 

