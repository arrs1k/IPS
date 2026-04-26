import torch
from torch_geometric.data import Data

data = torch.load('address_graph.pt', weights_only=False)
print(data)