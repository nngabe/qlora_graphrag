import torch
import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.models import GAT,MLP
from torch_geometric.nn.conv import SAGEConv, GATConv
import torch.nn.functional as F


class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr=['max'], dropout=0.5, ffw_dim=2):
        super().__init__(aggr=aggr)
        self.phi = MLP(in_channels=2*in_channels, hidden_channels=ffw_dim*in_channels, out_channels=out_channels, num_layers=2, dropout=dropout)
        self.psi = MLP(in_channels=len(aggr)*out_channels, hidden_channels=ffw_dim*in_channels, out_channels=out_channels, num_layers=2, dropout=dropout)
        self.in_features=in_channels
        self.out_features=out_channels
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr=None):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.psi(self.propagate(edge_index, x=x))

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        tmp = torch.cat([x_i, x_j - x_i], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.phi(tmp)

    def __repr__(self):
        return f"EdgeConv(in_features={self.in_features}, out_features={self.out_features}, aggr={self.aggr})"

class MPNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, aggr=['mean'], residual=True, dropout=0.5, ffw_dim=2):
        super().__init__()

        self.residual = residual
        self.dropout = dropout
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        conv = EdgeConv

        self.conv = torch.nn.ModuleList() # edge convolution message passing layers
        self.project = torch.nn.ModuleList() # linear projection for residual connections

        # first layer
        self.conv.append( conv(in_channels, hidden_channels, aggr=aggr, dropout=dropout, ffw_dim=ffw_dim))
        self.project.append( torch.nn.Linear(in_channels, hidden_channels))

        # middle layers
        for _ in range(num_layers-2):
            self.conv.append( conv(hidden_channels, hidden_channels, aggr=aggr, dropout=dropout, ffw_dim=ffw_dim))
            #self.project.append( torch.nn.Linear(hidden_channels, hidden_channels))
            self.project.append( torch.nn.Identity())

        # output layers
        self.conv.append( conv(hidden_channels, out_channels, aggr=aggr, dropout=dropout, ffw_dim=ffw_dim))
        self.project.append( torch.nn.Linear(hidden_channels, out_channels))

    def forward(self, x, edge_index, edge_attr=None):
        for conv,proj in zip(self.conv[:-1],self.project[:-1]):
            pr = proj(x)
            x = conv(x, edge_index)
            x = x.relu()
            if self.residual:
                x = x + pr
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv[-1](x, edge_index)
        return x

class Proj(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, out_tokens, dropout = 0.):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, out_dim*out_tokens),
            torch.nn.Dropout(dropout),
            torch.nn.Unflatten(-1, (out_tokens, out_dim))
        )

    def forward(self, x):
        return self.net(x)

if __name__ == '__main__':

    conv = EdgeConv(1536,1536,aggr=['mean','max','sum','var','min'])
    x = torch.rand((100,1536))
    edge_index = torch.triu_indices(100,100)
    
    mpnn = MPNN(1536, 2048, 1536, 4)
    
    gnn = GAT(
        in_channels=1536,
        hidden_channels=2048,
        out_channels=1536,
        num_layers=4,
        heads=4,
    )

    def count_parameters(model):
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return f'num_params: {str(count//1e+6)}M'
