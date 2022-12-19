import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN, DCRNN, GConvGRU, GConvLSTM

class A3TGCN_Temporal(torch.nn.Module):
    def __init__(self, node_features, periods, horizon, out_channels=32):
        super(A3TGCN_Temporal, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN(in_channels=node_features, 
                           out_channels=out_channels, 
                           periods=periods)
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(out_channels, horizon)

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
    def __init__(self, node_features, horizon, filter_size=3, out_channels=64):
        super(DCRNN_Temporal, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.dcrnn = DCRNN(in_channels=node_features, 
                           out_channels=out_channels, 
                           K=filter_size)
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(out_channels, horizon)

    def forward(self, x, edge_index, edge_weight):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.dcrnn(x.squeeze(1), edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h
    
    
class GConvGRU_Temporal(torch.nn.Module):

    def __init__(self, node_features, horizon):
        super(GConvGRU_Temporal, self).__init__()
        self.recurrent_1 = GConvGRU(node_features, 32, 2)
        self.linear = torch.nn.Linear(32, horizon)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent_1(x.squeeze(1), edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h
    
    
class GConvLSTM_Temporal(torch.nn.Module):

    def __init__(self, node_features, horizon):
        super(GConvLSTM_Temporal, self).__init__()
        self.recurrent_1 = GConvLSTM(node_features, 32, 2)
        self.linear = torch.nn.Linear(32, horizon)

    def forward(self, x, edge_index, edge_weight, hidden=None, context=None):
        h_0, c_0 = self.recurrent_1(x.squeeze(1), edge_index, edge_weight, hidden, context)
        h = F.relu(h_0)
        h = self.linear(h)
        return h, h_0, c_0