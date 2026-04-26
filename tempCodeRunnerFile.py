import argparse
import torch
import pandas as pd
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.utils import degree

def load_ethereum_data(csv_path):
    """
    Загрузка данных из CSV файла с транзакциями Ethereum
    Ожидаемые колонки: from_address, to_address, value, (опционально: token_address, transaction_hash, block_number)
    """
    df = pd.read_csv(csv_path)
    
    # Очистка данных
    if df['value'].dtype == 'object':
        df['value'] = pd.to_numeric(df['value'].astype(str).str.replace(',', ''), errors='coerce')
    
    # Удаляем строки с некорректными значениями
    df = df.dropna(subset=['from_address', 'to_address', 'value'])
    df = df[df['value'] > 0]
    
    return df

def build_networkx_graph(df):
    """
    Построение NetworkX графа из DataFrame с транзакциями
    """
    G = nx.DiGraph()
    
    # Группируем транзакции по парам для агрегации весов
    grouped = df.groupby(['from_address', 'to_address']).agg({
        'value': ['sum', 'count']
    }).reset_index()
    
    grouped.columns = ['from_address', 'to_address', 'total_value', 'tx_count']
    
    # Добавляем рёбра в граф
    for _, row in grouped.iterrows():
        G.add_edge(
            row['from_address'], 
            row['to_address'], 
            value=row['total_value'],
            count=row['tx_count']
        )
    
    return G

def convert_to_pyg_data(G_nx):
    """
    Конвертация NetworkX графа в PyG Data объект
    """
    nodes = list(G_nx.nodes())
    addr2idx = {addr: i for i, addr in enumerate(nodes)}
    
    # Собираем рёбра и их атрибуты
    edges = list(G_nx.edges(data=True))
    src = [addr2idx[u] for u, v, _ in edges]
    dst = [addr2idx[v] for u, v, _ in edges]
    weights = [data.get('value', 0) for _, _, data in edges]
    
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    # Логарифмическая трансформация весов для нормализации
    log_weights = np.log1p(weights)
    max_weight = max(log_weights) if log_weights else 1
    normalized_weights = [w / max_weight for w in log_weights]
    edge_attr = torch.tensor(normalized_weights, dtype=torch.float32).unsqueeze(1)
    
    data = Data(edge_index=edge_index, edge_attr=edge_attr)
    return data

def add_node_features(data):
    """
    Добавление признаков для вершин графа
    """
    num_nodes = data.num_nodes
    in_deg = degree(data.edge_index[1], num_nodes=num_nodes, dtype=torch.float32)
    out_deg = degree(data.edge_index[0], num_nodes=num_nodes, dtype=torch.float32)
    
    # Нормализуем степени
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
    """
    Сохранение PyG Data объекта
    """
    torch.save(data, path)
    print(f"Данные сохранены в {path}")

def prepare_link_prediction_loaders(data, num_neighbors=[10, 5], batch_size=1024,
                                    num_val=0.1, num_test=0.1, neg_sampling_ratio=1.0):
    """
    Подготовка данных для задачи link prediction
    """
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Построение графа транзакций Ethereum и подготовка данных для link prediction")
    parser.add_argument("csv_file", help="Путь к CSV файлу с транзакциями Ethereum")
    parser.add_argument("--save", default="ethereum_graph.pt", help="Имя файла для сохранения объекта Data")
    parser.add_argument("--batch_size", type=int, default=1024, help="Размер батча")
    
    args = parser.parse_args()
    
    print(f"Загрузка данных из {args.csv_file}...")
    df = load_ethereum_data(args.csv_file)
    print(f"Загружено {len(df)} транзакций")
    
    print("Построение NetworkX графа...")
    G_nx = build_networkx_graph(df)
    print(f"Граф: {G_nx.number_of_nodes()} вершин, {G_nx.number_of_edges()} рёбер")
    
    print("Конвертация в PyG формат...")
    data = convert_to_pyg_data(G_nx)
    
    print("Добавление признаков вершин...")
    data = add_node_features(data)
    
    print("Подготовка данных для link prediction...")
    train_loader, val_data, test_data = prepare_link_prediction_loaders(data, batch_size=args.batch_size)
    
    save_data(data, args.save)
    
    print(f"\nИтоговый граф: {data.num_nodes} вершин, {data.num_edges} рёбер")
    print(f"Размерность признаков вершин: {data.x.shape[1]}")
    print(f"Train батчей: {len(train_loader)}")
    print(f"Validation рёбер: {val_data.edge_index.shape[1]}")
    print(f"Test рёбер: {test_data.edge_index.shape[1]}")