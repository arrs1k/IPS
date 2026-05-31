import argparse
import torch
import pandas as pd
import numpy as np
import networkx as nx
import os
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.utils import degree, remove_self_loops
from indices import jaccard_scores, adamic_adar_scores, katz_scores, personalized_pagerank_scores


def load_ethereum_data(csv_path):
    df = pd.read_csv(csv_path)
    if df['value'].dtype == 'object':
        df['value'] = pd.to_numeric(df['value'].astype(str).str.replace(',', ''), errors='coerce')

    df = df.dropna(subset=['from_address', 'to_address', 'value'])
    df = df[df['value'] > 0]
    return df


def build_networkx_graph(df):
    G = nx.DiGraph()
    grouped = df.groupby(['from_address', 'to_address']).agg({
        'value': ['sum', 'count']
    }).reset_index()

    grouped.columns = ['from_address', 'to_address', 'total_value', 'tx_count']

    for _, row in grouped.iterrows():
        G.add_edge(
            row['from_address'],
            row['to_address'],
            value=row['total_value'],
            count=row['tx_count']
        )
    return G


def convert_to_pyg_data(G_nx):
    nodes = list(G_nx.nodes())
    addr2idx = {addr: i for i, addr in enumerate(nodes)}

    edges = list(G_nx.edges(data='value'))
    src = [addr2idx[u] for u, v, w in edges]
    dst = [addr2idx[v] for u, v, w in edges]
    weights = [w if w is not None else 0.0 for u, v, w in edges]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
    data = Data(edge_index=edge_index, edge_attr=edge_attr)
    return data


def add_heuristic_node_features(data):
    num_nodes = data.num_nodes
    edges = data.edge_index

    in_deg = degree(edges[1], num_nodes=num_nodes, dtype=torch.float32)
    out_deg = degree(edges[0], num_nodes=num_nodes, dtype=torch.float32)

    def mean_score_per_node(score_func):
        scores = score_func(Data(edge_index=edges, num_nodes=num_nodes), edge_label_index=edges)
        aggregated = torch.zeros(num_nodes, dtype=torch.float32)
        count = torch.zeros(num_nodes, dtype=torch.float32)
        aggregated = aggregated.scatter_add(0, edges[0], scores)
        count = count.scatter_add(0, edges[0], torch.ones_like(scores))
        count[count == 0] = 1.0
        return (aggregated / count).unsqueeze(1)

    features = [torch.stack([in_deg, out_deg], dim=1)]
    features.append(mean_score_per_node(jaccard_scores))
    features.append(mean_score_per_node(adamic_adar_scores))

    data.x = torch.cat(features, dim=1)
    return data


def prepare_and_save_loaders(train_data, val_data, test_data, save_dir,
                             num_neighbors, batch_size):
    train_loader = LinkNeighborLoader(
        train_data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        edge_label_index=train_data.edge_label_index,
        edge_label=train_data.edge_label,
        shuffle=True,
    )
    val_loader = LinkNeighborLoader(
        train_data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        edge_label_index=val_data.edge_label_index,
        edge_label=val_data.edge_label,
        shuffle=False,
    )
    test_loader = LinkNeighborLoader(
        train_data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        edge_label_index=test_data.edge_label_index,
        edge_label=test_data.edge_label,
        shuffle=False,
    )

    save_path = os.path.join(save_dir, 'ethereum_link_pred_data.pt')
    torch.save({
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data,
        'num_neighbors': num_neighbors,
        'batch_size': batch_size,
    }, save_path)
    print(f"Данные и конфигурация загрузчиков сохранены в {save_path}")


def main(args):
    df = load_ethereum_data(args.csv_file)
    print(f"Загружено {len(df)} транзакций")

    G_nx = build_networkx_graph(df)
    print(f"Граф NetworkX: {G_nx.number_of_nodes()} вершин, {G_nx.number_of_edges()} рёбер")

    data = convert_to_pyg_data(G_nx)

    data.edge_index, data.edge_attr = remove_self_loops(data.edge_index, data.edge_attr)

    transform = RandomLinkSplit(
        is_undirected=False,
        num_val=args.num_val,
        num_test=args.num_test,
        neg_sampling_ratio=args.neg_sampling_ratio,
    )
    train_data, val_data, test_data = transform(data)

    train_data = add_heuristic_node_features(train_data)
    val_data.x = train_data.x
    test_data.x = train_data.x

    prepare_and_save_loaders(train_data, val_data, test_data,
                             save_dir=args.save_dir,
                             num_neighbors=args.num_neighbors,
                             batch_size=args.batch_size)

    print(f"Граф загружен: {train_data.num_nodes} вершин, "
          f"{train_data.edge_index.size(1)} тренировочных рёбер")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Подготовка данных Ethereum для link prediction")
    parser.add_argument("csv_file", help="Путь к CSV файлу с транзакциями Ethereum")
    parser.add_argument("--save-dir", default=".", help="Директория для сохранения")
    parser.add_argument("--num-val", type=float, default=0.1)
    parser.add_argument("--num-test", type=float, default=0.1)
    parser.add_argument("--neg-sampling-ratio", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-neighbors", type=int, nargs='+', default=[10, 5])

    args = parser.parse_args()
    main(args)