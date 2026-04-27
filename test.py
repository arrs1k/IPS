import torch
from sklearn.metrics import roc_auc_score
import torch_geometric as tg
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.transforms import RandomLinkSplit
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from btc_link_predictor import LinkPredictor
class SymmetricNormConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return self.lin(aggr_out)
class SymmetricNormLinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = SymmetricNormConv(in_channels, hidden_channels)
        self.conv2 = SymmetricNormConv(hidden_channels, hidden_channels)
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, edge_index, edge_label_index, edge_attr=None):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        src, dst = edge_label_index
        h = torch.cat([x[src], x[dst]], dim=1)
        return self.classifier(h)

device = 'cpu'
data = torch.load('bitcoin_adress_split.pt', weights_only=False)
data = data.to(device)
transform = RandomLinkSplit(
    is_undirected=False,
    num_val=0.1,
    num_test=0.1,
    neg_sampling_ratio=1.0,
)
train_data, val_data, test_data = transform(data)
model = SymmetricNormLinkPredictor(
    in_channels=train_data.num_node_features,
    hidden_channels=64
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()

trainer = LinkPredictor(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    batch_size=1024,
    num_neighbors=[10, 5]
)

trainer.fit(train_data, val_data, epochs=50, verbose=True)
probs = trainer.predict(test_data)
true = test_data.edge_label.cpu().numpy()
auc = roc_auc_score(true, probs)
print(f'Test AUC: {auc:.4f}')