import torch
import torch_geometric
from torch_geometric.nn.models import GAT,MLP
from torch_geometric.nn.conv import SAGEConv, GATConv, EdgeConv
import torch.nn.functional as F

edge_conv = lambda in_, out_, aggr: EdgeConv(MLP(in_channels=in_, hidden_channels=4*in_, out_channels=in_, num_layers=2), aggr=aggr)

class MPNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, conv='EdgeConv', aggr='mean', residual = True):
        super().__init__()

        Conv = getattr(torch_geometric.nn.conv,conv)
        if conv=='EdgeConv':
            Conv = edge_conv

        self.conv = torch.nn.ModuleList()
        #hidden_channels = hidden_channels // heads
        self.conv.append( Conv(in_channels, hidden_channels, aggr=aggr))
        for _ in range(num_layers-2):
            self.conv.append( Conv(hidden_channels, hidden_channels, aggr=aggr))
        self.conv.append( Conv(hidden_channels, out_channels, aggr=aggr))

    def forward(self, x, edge_index):
        for conv in self.conv[:-1]:
            x = conv(x, edge_index)
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv[-1](x, edge_index)
        return x

if __name__ == '__main__':

    x = torch.rand((100,1536))
    edge_index = torch.triu_indices(100,100)

    gnn = GAT(
        in_channels=1536,
        hidden_channels=2048,
        out_channels=1536,
        num_layers=4,
        heads=4,
    )

    mpnn = MPNN(
        in_channels=1536,
        hidden_channels=2048,
        out_channels=1536,
        num_layers=4,
    )
