import torch
from torch_geometric.data import Data

data = torch.load('bitcoin_sample_split.pt', weights_only=False)
print(data)