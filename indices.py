import torch
import numpy as np
from scipy.sparse import csr_matrix
from torch_geometric.utils import to_undirected


def jaccard_scores(data, edge_label_index=None):
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    if edge_label_index is None:
        edge_label_index = data.edge_label_index

    edge_index_und = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index_und = torch.unique(edge_index_und, dim=1)

    row, col = edge_index_und.cpu().numpy()
    values = np.ones(len(row), dtype=np.float32)
    adj = csr_matrix((values, (row, col)), shape=(num_nodes, num_nodes))

    u = edge_label_index[0].cpu().numpy()
    v = edge_label_index[1].cpu().numpy()

    inter = np.array(adj[u].multiply(adj[v]).sum(axis=1)).flatten()
    deg_u = np.array(adj[u].sum(axis=1)).flatten()
    deg_v = np.array(adj[v].sum(axis=1)).flatten()
    union = deg_u + deg_v - inter

    with np.errstate(divide='ignore', invalid='ignore'):
        scores = inter / union
        scores[union == 0] = 0.0

    return torch.tensor(scores, dtype=torch.float)