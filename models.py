import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, ChebConv, GATConv, SGConv, SAGEConv
from torch.autograd import Variable

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    if torch.cuda.is_available():
        U = U.cuda()
    return -Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    if torch.cuda.is_available():
        y_hard = y_hard.cuda()
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y

class GCNNet(torch.nn.Module):
    def __init__(self, num_features, embedding_size=128, slope=1.0, temp=0.6):
        super(GCNNet, self).__init__()
        self.slope = slope
        self.temp  = temp
        self.conv1 = GCNConv(num_features, embedding_size*2, cached=True)
        self.conv2 = GCNConv(embedding_size*2, embedding_size, cached=True)

    def forward(self,x, edge_index):
        x = F.leaky_relu(self.conv1(x, edge_index),self.slope)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        self.embedding = x
        return gumbel_softmax(x, self.temp) 

class SAGENet(torch.nn.Module):
    def __init__(self, num_features, embedding_size=128, slope=1.0, temp=0.6):
        super(SAGENet, self).__init__()
        self.slope = slope
        self.temp = temp
        self.conv1 = SAGEConv(num_features, embedding_size*2)
        self.conv2 = SAGEConv(embedding_size*2, embedding_size)

    def forward(self,x, edge_index):
        x = F.leaky_relu(self.conv1(x, edge_index), self.slope)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        self.embedding = x
        return gumbel_softmax(x, self.temp)          

class GATNet(torch.nn.Module):
    def __init__(self,num_features, embedding_size=128, slope=1.0, temp=0.6):
        super(GATNet, self).__init__()
        self.slope = slope
        self.temp = temp
        self.conv1 = GATConv(num_features, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 8, embedding_size, dropout=0.6)

    def forward(self,x, edge_index):
        # x = F.dropout(x, p=0.6, training=self.training)
        x = F.leaky_relu(self.conv1(x, edge_index), self.slope)
        # x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        self.embedding = x
        return gumbel_softmax(x, self.temp)  

class SGNet(torch.nn.Module):
    def __init__(self, num_features, embedding_size=128, slope=None, temp= None):
        super(SGNet, self).__init__()
        self.conv1 = SGConv(
            num_features, embedding_size, K=2, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=True)
        self.prelu = nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        return x


# model = DeepGraphInfomax(
#     hidden_channels=512,
#     encoder=Encoder(dataset.num_features, 512),
#     summary=lambda z, *args, **kwargs: z.mean(dim=0),
#     corruption=corruption)
