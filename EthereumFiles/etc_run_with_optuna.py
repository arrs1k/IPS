import torch
from aggregate_update_methods import run_with_optuna
from torch_geometric.loader import LinkNeighborLoader
print("Загрузка данных...")
checkpoint = torch.load('ethereum_link_pred_data.pt', weights_only=False)

train_data = checkpoint['train_data']
val_data = checkpoint['val_data']
test_data = checkpoint['test_data']
num_neighbors = checkpoint['num_neighbors']
batch_size = checkpoint['batch_size']

print(f"Train: {train_data.num_nodes} вершин, {train_data.edge_index.shape[1]} рёбер")
print(f"Val: {val_data.edge_label_index.shape[1]} пар")
print(f"Test: {test_data.edge_label_index.shape[1]} пар")
print(f"Размерность признаков: {train_data.num_features}")

train_loader = LinkNeighborLoader(
    train_data, num_neighbors=num_neighbors, batch_size=batch_size,
    edge_label_index=train_data.edge_label_index,
    edge_label=train_data.edge_label, shuffle=True
)

val_loader = LinkNeighborLoader(
    train_data, num_neighbors=num_neighbors, batch_size=batch_size,
    edge_label_index=val_data.edge_label_index,
    edge_label=val_data.edge_label, shuffle=False
)

test_loader = LinkNeighborLoader(
    train_data, num_neighbors=num_neighbors, batch_size=batch_size,
    edge_label_index=test_data.edge_label_index,
    edge_label=test_data.edge_label, shuffle=False
)

best_params, final_results, final_histories = run_with_optuna(
    train_loader, val_loader, test_loader,
    train_data, val_data, test_data,
    n_trials=20,
    save_dir='ethereum_optuna_results',
    final_epochs=50
)

print("\n" + "=" * 80)
print("ЭКСПЕРИМЕНТ ЗАВЕРШЁН!")
print("=" * 80)
print("Результаты сохранены в папке ethereum_optuna_results")