# rgcn_model.py
import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv

class RGCN(nn.Module):
    def __init__(self, num_nodes, num_relations, hidden_dim=128, num_layers=2):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, hidden_dim)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim
            out_dim = hidden_dim
            self.layers.append(
                RGCNConv(in_dim, out_dim, num_relations)
            )

    def forward(self, x, edge_index, edge_type):
        h = self.emb(x)
        for layer in self.layers:
            h = layer(h, edge_index, edge_type)
            h = torch.relu(h)
        return h
