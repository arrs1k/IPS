import argparse
import torch
import pandas as pd
import os
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.utils import degree, remove_self_loops
from chartalist.common.bitcoin_graph_maker import BitcoinGraphMaker
from indices import jaccard_scores, katz_scores, adamic_adar_scores, personalized_pagerank_scores


def load_raw_data(in_path, out_path):
    with open(in_path, 'r') as f:
        lines_in = [line.strip() for line in f if line.strip()]
    with open(out_path, 'r') as f:
        lines_out = [line.strip() for line in f if line.strip()]

    df_in = pd.DataFrame({'trans': lines_in})
    df_out = pd.DataFrame({'trans': lines_out})
    return df_in, df_out


def collapse_transaction_vertices(G):
    tx_nodes = [node for node, attr in G.nodes(data=True) if attr.get('type') == 'trans']
    for tx in tx_nodes:
        in_edges = list(G.in_edges(tx))
        out_edges = list(G.out_edges(tx))
        for u, _ in in_edges:
            for _, v in out_edges:
                if not G.has_edge(u, v):
                    G.add_edge(u, v, value=1.0)
        G.remove_node(tx)
    return G


def build_networkx_graph(df_in, df_out, collapse=True):
    bgm = BitcoinGraphMaker()
    G_nx = bgm.make_graph(df_in, df_out)
    if collapse:
        G_nx = collapse_transaction_vertices(G_nx)
    return G_nx


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

    save_path = os.path.join(save_dir, 'bitcoin_link_pred_data.pt')
    torch.save({
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data,
        'num_neighbors': num_neighbors,
        'batch_size': batch_size,
    }, save_path)
    print(f"Данные и конфигурация загрузчиков сохранены в {save_path}")


def main(args):
    df_in, df_out = load_raw_data(args.in_file, args.out_file)
    G_nx = build_networkx_graph(df_in, df_out, collapse=True)
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
    parser = argparse.ArgumentParser(description="Подготовка данных Bitcoin для link prediction")
    parser.add_argument("in_file")
    parser.add_argument("out_file")
    parser.add_argument("--save-dir", default=".", help="Директория для сохранения")
    parser.add_argument("--num-val", type=float, default=0.1)
    parser.add_argument("--num-test", type=float, default=0.1)
    parser.add_argument("--neg-sampling-ratio", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-neighbors", type=int, nargs='+', default=[10, 5])

    args = parser.parse_args()
    main(args)