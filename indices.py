import torch
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, eye, diags
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

def katz_scores(data, beta=0.1, edge_label_index=None):
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    if edge_label_index is None:
        edge_label_index = data.edge_label_index

    edge_index_und = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index_und = torch.unique(edge_index_und, dim=1)

    row, col = edge_index_und.cpu().numpy()
    data_vals = np.ones(len(row), dtype=np.float64)
    A = csr_matrix((data_vals, (row, col)), shape=(num_nodes, num_nodes))

    M = eye(num_nodes, dtype=np.float64, format='csr') - beta * A

    u = edge_label_index[0].cpu().numpy()
    v = edge_label_index[1].cpu().numpy()

    unique_v, inv_map = np.unique(v, return_inverse=True)

    E = np.zeros((num_nodes, len(unique_v)), dtype=np.float64)
    E[unique_v, np.arange(len(unique_v))] = 1.0

    E_csr = csr_matrix(E)
    X = spsolve(M, E_csr)

    katz_values = X[u, inv_map]

    scores = 1.0 / (1.0 + np.exp(-katz_values))

    return torch.tensor(scores, dtype=torch.float)


def adamic_adar_scores(data, edge_label_index=None):
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    if edge_label_index is None:
        edge_label_index = data.edge_label_index

    edge_index_und = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index_und = torch.unique(edge_index_und, dim=1)

    row, col = edge_index_und.cpu().numpy()
    values = np.ones(len(row), dtype=np.float32)
    adj = csr_matrix((values, (row, col)), shape=(num_nodes, num_nodes))

    degrees = np.array(adj.sum(axis=1)).flatten()
    with np.errstate(divide='ignore', invalid='ignore'):
        weights = np.zeros_like(degrees, dtype=np.float32)
        mask = degrees > 1
        weights[mask] = 1.0 / np.log(degrees[mask])
    
    W = diags(weights, format='csr')
    adj_weighted = adj @ W

    u = edge_label_index[0].cpu().numpy()
    v = edge_label_index[1].cpu().numpy()
    
    scores = np.array(adj_weighted[u].multiply(adj[v]).sum(axis=1)).flatten()
    
    return torch.tensor(scores, dtype=torch.float)


def personalized_pagerank_scores(data, edge_label_index=None, alpha=0.85, K=10):
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    if edge_label_index is None:
        edge_label_index = data.edge_label_index

    edge_index_und = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index_und = torch.unique(edge_index_und, dim=1)

    u = edge_label_index[0]
    v = edge_label_index[1]
    
    appnp = APPNP(K=K, alpha=alpha, cached=True, normalize=True)
    
    x = torch.eye(num_nodes, dtype=torch.float)
    
    with torch.no_grad():
        ppr_matrix = appnp(x, edge_index_und)
    
    scores = torch.zeros(len(u), dtype=torch.float)
    
    unique_u = torch.unique(u)
    for ui in unique_u:
        mask = (u == ui)
        v_indices = v[mask]
        scores[mask] = ppr_matrix[ui][v_indices]
    
    return scores
