import argparse
import torch
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.utils import degree


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

    if not edges:
        print("Предупреждение: граф не содержит рёбер")
        return Data(edge_index=torch.tensor([[], []], dtype=torch.long))

    src = [addr2idx[u] for u, v, w in edges]
    dst = [addr2idx[v] for u, v, w in edges]
    weights = [w for u, v, w in edges]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
    data = Data(edge_index=edge_index, edge_attr=edge_attr)
    return data


def add_node_features(data):
    num_nodes = data.num_nodes

    if data.num_edges == 0:
        data.x = torch.zeros((num_nodes, 4), dtype=torch.float32)
        return data

    in_deg = degree(data.edge_index[1], num_nodes=num_nodes, dtype=torch.float32)
    out_deg = degree(data.edge_index[0], num_nodes=num_nodes, dtype=torch.float32)

    max_deg = max(in_deg.max(), out_deg.max()).item()
    if max_deg > 0:
        in_deg_norm = in_deg / max_deg
        out_deg_norm = out_deg / max_deg
    else:
        in_deg_norm = in_deg
        out_deg_norm = out_deg

    data.x = torch.stack([in_deg_norm, out_deg_norm, torch.log1p(in_deg), torch.log1p(out_deg)], dim=1)
    return data


def save_data(data, path):
    torch.save(data, path)
    print(f"Данные сохранены в {path}")


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
        shuffle=True
    )

    return train_loader, val_data, test_data


def visualize_top_nodes_with_edges(G, top_n=50, max_edges=500, figsize=(14, 10)):
    """
    Визуализирует топ N узлов по степени и их связи (оптимизированная версия)
    """
    degrees = dict(G.degree())
    top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_nodes_list = [node for node, deg in top_nodes]

    print(f"\nТоп-5 узлов по степени:")
    for i, (node, deg) in enumerate(top_nodes[:5]):
        print(f"  {i + 1}. {str(node)[:20]}... степень: {deg}")

    print("Создание подграфа...")
    subgraph = G.subgraph(top_nodes_list).copy()

    print(f"Исходный подграф: {subgraph.number_of_nodes()} узлов, {subgraph.number_of_edges()} рёбер")

    if subgraph.number_of_edges() > max_edges:
        print(f"Слишком много рёбер ({subgraph.number_of_edges()}), оставляем топ-{max_edges} по весу")
        edges_with_weights = [(u, v, data.get('value', 0)) for u, v, data in subgraph.edges(data=True)]
        edges_with_weights.sort(key=lambda x: x[2], reverse=True)
        edges_to_keep = edges_with_weights[:max_edges]
        new_graph = nx.DiGraph()
        new_graph.add_nodes_from(subgraph.nodes())
        for u, v, w in edges_to_keep:
            new_graph.add_edge(u, v, value=w)
        subgraph = new_graph

    isolated = [n for n in subgraph.nodes() if subgraph.degree(n) == 0]
    subgraph.remove_nodes_from(isolated)

    if subgraph.number_of_edges() == 0:
        print("Не удалось создать подграф с рёбрами")
        return

    print("Вычисление позиций узлов...")
    pos = nx.spring_layout(subgraph, k=2.0, iterations=20, seed=42)

    print("Отрисовка графа...")
    plt.figure(figsize=figsize)

    deg_sub = dict(subgraph.degree())
    if deg_sub:
        max_deg_sub = max(deg_sub.values())
        if max_deg_sub > 0:
            node_sizes = [500 + (deg_sub[n] / max_deg_sub) * 1500 for n in subgraph.nodes()]
        else:
            node_sizes = [800] * subgraph.number_of_nodes()
    else:
        node_sizes = [800] * subgraph.number_of_nodes()

    in_deg = dict(subgraph.in_degree())
    out_deg = dict(subgraph.out_degree())
    node_balance = [in_deg.get(n, 0) - out_deg.get(n, 0) for n in subgraph.nodes()]

    nodes_draw = nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes,
                                        node_color=node_balance, cmap='RdBu', alpha=0.8)

    if nodes_draw:
        plt.colorbar(nodes_draw, label='Баланс транзакций (получено - отправлено)')

    nx.draw_networkx_edges(subgraph, pos, alpha=0.3, edge_color='gray',
                           arrows=True, arrowsize=8, width=0.8,
                           arrowstyle='->', connectionstyle='arc3,rad=0.1')

    labels = {n: str(n)[:8] + '..' for n in subgraph.nodes()}
    nx.draw_networkx_labels(subgraph, pos, labels, font_size=8, font_weight='bold')

    plt.title(f"Транзакционный граф Ethereum\n"
              f"{subgraph.number_of_nodes()} узлов, {subgraph.number_of_edges()} транзакций",
              fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    print(f"\nВизуализировано:")
    print(f"  - Узлов: {subgraph.number_of_nodes()}")
    print(f"  - Рёбер: {subgraph.number_of_edges()}")
    if subgraph.number_of_nodes() > 0:
        print(f"  - Средняя степень: {2 * subgraph.number_of_edges() / subgraph.number_of_nodes():.2f}")

    return subgraph


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Построение графа транзакций Ethereum и подготовка данных для link prediction")
    parser.add_argument("csv_file", help="Путь к CSV файлу с транзакциями Ethereum")
    parser.add_argument("--save", default="ethereum_graph.pt", help="Имя файла для сохранения объекта Data")
    parser.add_argument("--batch_size", type=int, default=1024, help="Размер батча")
    parser.add_argument("--visualize", action="store_true", help="Визуализировать граф после построения")
    parser.add_argument("--top_n", type=int, default=50, help="Количество топ узлов для визуализации")
    parser.add_argument("--max_edges_viz", type=int, default=500, help="Максимальное количество рёбер на визуализации")

    args = parser.parse_args()

    print(f"Загрузка данных из {args.csv_file}...")
    df = load_ethereum_data(args.csv_file)
    print(f"Загружено {len(df)} транзакций")

    print("Построение NetworkX графа...")
    G_nx = build_networkx_graph(df)
    print(f"Граф: {G_nx.number_of_nodes()} вершин, {G_nx.number_of_edges()} рёбер")

    if args.visualize:
        visualize_top_nodes_with_edges(G_nx, top_n=args.top_n, max_edges=args.max_edges_viz)

    print("\nКонвертация в PyG формат...")
    data = convert_to_pyg_data(G_nx)

    print("Добавление признаков вершин...")
    data = add_node_features(data)

    print("Подготовка данных для link prediction...")
    train_loader, val_data, test_data = prepare_link_prediction_loaders(data, batch_size=args.batch_size)

    save_data(data, args.save)

    print(f"\nИтоговый граф: {data.num_nodes} вершин, {data.num_edges} рёбер")
    if hasattr(data, 'x') and data.x is not None:
        print(f"Размерность признаков вершин: {data.x.shape[1]}")
    print(f"Train батчей: {len(train_loader)}")
    print(f"Validation рёбер: {val_data.edge_index.shape[1]}")
    print(f"Test рёбер: {test_data.edge_index.shape[1]}")