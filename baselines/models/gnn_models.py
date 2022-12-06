import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN

class A3TGCN_Temporal(torch.nn.Module):
    def __init__(self, node_features, periods):
        super(A3TGCN_Temporal, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN(in_channels=node_features, 
                           out_channels=24, 
                           periods=periods)
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(24, periods)

    def forward(self, x, edge_index, edge_weight):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h