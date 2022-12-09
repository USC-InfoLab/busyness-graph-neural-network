import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN, DCRNN, GConvGRU

class A3TGCN_Temporal(torch.nn.Module):
    def __init__(self, node_features, periods, out_channels=128):
        super(A3TGCN_Temporal, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN(in_channels=node_features, 
                           out_channels=out_channels, 
                           periods=periods)
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(out_channels, periods)

    def forward(self, x, edge_index, edge_weight):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h
    
    
class DCRNN_Temporal(torch.nn.Module):
    def __init__(self, node_features, periods, filter_size=3, out_channels=64):
        super(DCRNN_Temporal, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.dcrnn = DCRNN(in_channels=node_features, 
                           out_channels=out_channels, 
                           K=filter_size)
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(out_channels, periods)

    def forward(self, x, edge_index, edge_weight):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.dcrnn(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h
    
    
class GConvGRU_Temporal(torch.nn.Module):

    def __init__(self, node_features, periods):
        super(GConvGRU_Temporal, self).__init__()
        self.recurrent_1 = GConvGRU(node_features, 32, 5)
        self.recurrent_2 = GConvGRU(32, 16, 5)
        self.linear = torch.nn.Linear(16, periods)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent_1(x, edge_index, edge_weight)
        h = F.relu(h)
        h = F.dropout(h, training=self.training)
        h = self.recurrent_2(h, edge_index, edge_weight)
        h = F.relu(h)
        h = F.dropout(h, training=self.training)
        h = self.linear(h)
        return h