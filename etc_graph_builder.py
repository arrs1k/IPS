# etc_graph_builder.py
import argparse
import torch
import pandas as pd
import numpy as np
import networkx as nx
import os
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.utils import degree


def load_ethereum_data(csv_path):
    """Загружает и предобрабатывает данные транзакций Ethereum."""
    df = pd.read_csv(csv_path)
    if df['value'].dtype == 'object':
        df['value'] = pd.to_numeric(df['value'].astype(str).str.replace(',', ''), errors='coerce')

    df = df.dropna(subset=['from_address', 'to_address', 'value'])
    df = df[df['value'] > 0]
    return df


def build_networkx_graph(df):
    """Строит ориентированный граф NetworkX из DataFrame."""
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
    """Конвертирует NetworkX граф в формат PyTorch Geometric."""
    nodes = list(G_nx.nodes())
    addr2idx = {addr: i for i, addr in enumerate(nodes)}
    edges = list(G_nx.edges(data='value'))

    if not edges:
        print("Предупреждение: граф не содержит рёбер")
        return Data(edge_index=torch.tensor([[], []], dtype=torch.long), num_nodes=0)

    src = [addr2idx[u] for u, v, w in edges]
    dst = [addr2idx[v] for u, v, w in edges]
    weights = [w if w is not None else 0.0 for u, v, w in edges]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)

    data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(nodes))
    return data


def add_node_features(data):
    """Добавляет признаки вершин на основе степеней."""
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


def prepare_and_save_loaders(train_data, val_data, test_data, save_dir,
                             num_neighbors, batch_size):
    """
    Создаёт загрузчики для обучения, валидации и тестирования,
    и сохраняет их вместе с данными.
    """
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
    
    return train_loader, val_loader, test_loader


def visualize_top_nodes_with_edges(G, top_n=50, max_edges=500, figsize=(14, 10)):
    """Визуализирует топ узлов графа."""
    import matplotlib.pyplot as plt
    
    degrees = dict(G.degree())
    top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_nodes_list = [node for node, deg in top_nodes]

    print(f"\nТоп-5 узлов по степени:")
    for i, (node, deg) in enumerate(top_nodes[:5]):
        print(f"  {i + 1}. {str(node)[:20]}... степень: {deg}")

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

    plt.figure(figsize=figsize)
    pos = nx.spring_layout(subgraph, k=2.0, iterations=20, seed=42)

    deg_sub = dict(subgraph.degree())
    if deg_sub:
        max_deg_sub = max(deg_sub.values())
        if max_deg_sub > 0:
            node_sizes = [300 + (deg_sub[n] / max_deg_sub) * 1700 for n in subgraph.nodes()]
        else:
            node_sizes = [500] * subgraph.number_of_nodes()
    else:
        node_sizes = [500] * subgraph.number_of_nodes()

    node_colors = [deg_sub[n] for n in subgraph.nodes()]

    nodes_draw = nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes,
                                        node_color=node_colors, cmap='viridis', alpha=0.8)

    if nodes_draw:
        plt.colorbar(nodes_draw, label='Количество транзакций')

    nx.draw_networkx_edges(subgraph, pos, alpha=0.4, edge_color='gray',
                           arrows=True, arrowsize=8, width=0.8,
                           arrowstyle='->', connectionstyle='arc3,rad=0.1')

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


def main(args):
    """Основная функция для подготовки данных Ethereum."""
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
    print(f"Размерность признаков: {data.x.shape[1]}")

    # Разделение данных на train/val/test
    print("\nРазделение данных на train/val/test...")
    transform = RandomLinkSplit(
        is_undirected=False,
        num_val=args.num_val,
        num_test=args.num_test,
        neg_sampling_ratio=args.neg_sampling_ratio,
    )
    train_data, val_data, test_data = transform(data)

    print(f"Train: {train_data.edge_index.shape[1]} рёбер")
    print(f"Val: {val_data.edge_label_index.shape[1]} пар")
    print(f"Test: {test_data.edge_label_index.shape[1]} пар")

    # Создаём и сохраняем загрузчики
    print("\nСоздание загрузчиков...")
    train_loader, val_loader, test_loader = prepare_and_save_loaders(
        train_data, val_data, test_data,
        save_dir=args.save_dir,
        num_neighbors=args.num_neighbors,
        batch_size=args.batch_size
    )

    print(f"\nИтоговый граф: {train_data.num_nodes} вершин, "
          f"{train_data.edge_index.shape[1]} тренировочных рёбер")
    print(f"Train батчей: {len(train_loader)}")
    print(f"Val батчей: {len(val_loader)}")
    print(f"Test батчей: {len(test_loader)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Построение графа транзакций Ethereum и подготовка данных для link prediction"
    )
    parser.add_argument("csv_file", help="Путь к CSV файлу с транзакциями Ethereum")
    parser.add_argument("--save-dir", default=".", help="Директория для сохранения данных")
    parser.add_argument("--num-val", type=float, default=0.1, help="Доля рёбер для валидации")
    parser.add_argument("--num-test", type=float, default=0.1, help="Доля рёбер для теста")
    parser.add_argument("--neg-sampling-ratio", type=float, default=1.0, 
                        help="Отношение негативных примеров к позитивным")
    parser.add_argument("--batch-size", type=int, default=1024, help="Размер батча")
    parser.add_argument("--num-neighbors", type=int, nargs='+', default=[10, 5],
                        help="Количество соседей для каждого уровня сэмплирования")
    parser.add_argument("--visualize", action="store_true", help="Визуализировать граф после построения")
    parser.add_argument("--top_n", type=int, default=50, help="Количество топ узлов для визуализации")
    parser.add_argument("--max_edges_viz", type=int, default=500, 
                        help="Максимальное количество рёбер на визуализации")

    args = parser.parse_args()
    main(args)