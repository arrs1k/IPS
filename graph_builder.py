import argparse
import torch
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.utils import degree
from chartalist.common.bitcoin_graph_maker import BitcoinGraphMaker


def load_raw_data(in_path, out_path):
    with open(in_path, 'r') as f:
        lines_in = [line.strip() for line in f if line.strip()]
    with open(out_path, 'r') as f:
        lines_out = [line.strip() for line in f if line.strip()]

    df_in = pd.DataFrame({'trans': lines_in})
    df_out = pd.DataFrame({'trans': lines_out})
    return df_in, df_out

def build_networkx_graph(df_in, df_out):
    bgm = BitcoinGraphMaker()
    G_nx = bgm.make_graph(df_in, df_out)
    return G_nx

def convert_to_pyg_data(G_nx):
    nodes = list(G_nx.nodes())
    addr2idx = {addr: i for i, addr in enumerate(nodes)}

    edges = list(G_nx.edges(data='value'))
    src = [addr2idx[u] for u, v, w in edges]
    dst = [addr2idx[v] for u, v, w in edges]
    weights = [w for u, v, w in edges]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
    data = Data(edge_index=edge_index, edge_attr=edge_attr)
    return data

def add_transaction_weights(G, df_in, df_out):
    for _, row in df_in.iterrows():
        parts = row['trans'].strip().split('\t')
        if len(parts) < 4:
            continue
        txHash = parts[1]
        num_inputs = int(parts[2])
        for i in range(num_inputs):
            addr = parts[3 + 2*i]
            amount = float(parts[4 + 2*i])
            if G.has_edge(addr, txHash):
                G[addr][txHash]['value'] = amount

    for _, row in df_out.iterrows():
        parts = row['trans'].strip().split('\t')
        if len(parts) < 4:
            continue
        txHash = parts[1]
        num_outputs = int(parts[2])
        for i in range(num_outputs):
            addr = parts[3 + 2*i]
            amount = float(parts[4 + 2*i])
            if G.has_edge(txHash, addr):
                G[txHash][addr]['value'] = amount

    return G

def add_node_features(data):
    num_nodes = data.num_nodes
    in_deg = degree(data.edge_index[1], num_nodes=num_nodes, dtype=torch.float32)
    out_deg = degree(data.edge_index[0], num_nodes=num_nodes, dtype=torch.float32)
    data.x = torch.stack([in_deg, out_deg], dim=1)
    return data

def save_data(data, path):
    torch.save(data, path)

def prepare_link_prediction_loaders(data, num_neighbors=[10, 5], batch_size=1024,
                                    num_val=0.1, num_test=0.1, neg_sampling_ratio=1.0):
    transform = RandomLinkSplit(
        is_undirected=False,
        num_val=num_val,
        num_test=num_test,
        neg_sampling_ratio=neg_sampling_ratio,
    )
    train_data, val_data, test_data = transform(data)

    train_loader = LinkNeighborLoader(
        train_data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        edge_label_index=train_data.edge_label_index,
        edge_label=train_data.edge_label,
    )
    return train_loader, val_data, test_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Построение графа транзакций Bitcoin и подготовка данных для link prediction")
    parser.add_argument("in_file", help="Путь к объединённому CSV-файлу входов")
    parser.add_argument("out_file", help="Путь к объединённому CSV-файлу выходов")
    parser.add_argument("--save", default="bitcoin_split.pt", help="Имя файла для сохранения объекта Data (по умолчанию bitcoin_split.pt)")
    args = parser.parse_args()

    df_in, df_out = load_raw_data(args.in_file, args.out_file)
    G_nx = build_networkx_graph(df_in, df_out)
    G_nx = add_transaction_weights(G_nx, df_in, df_out)
    data = convert_to_pyg_data(G_nx)
    data = add_node_features(data)
    save_data(data, args.save)

    train_loader, val_data, test_data = prepare_link_prediction_loaders(data)

    print(f"Граф загружен: {data.num_nodes} вершин, {data.num_edges} рёбер")

