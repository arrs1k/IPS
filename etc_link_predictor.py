import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np


class EthereumLinkPredictor(nn.Module):
    """
    GNN модель для предсказания ссылок на графе Ethereum транзакций
    """

    def __init__(self, in_channels, hidden_channels=128, out_channels=64, num_layers=2, dropout=0.3):
        super().__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()  # batch normalization

        # Входной слой
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Скрытые слои
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Выходной слой для эмбеддингов
        self.lin = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def encode(self, x, edge_index):
        """
        Кодирование узлов в эмбеддинги
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lin(x)
        return x

    def decode(self, z, edge_label_index):
        """
        Декодирование эмбеддингов в предсказания рёбер
        """
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def forward(self, x, edge_index, edge_label_index):
        """
        Полный forward pass
        """
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)


class EthereumLinkPredictionTrainer:
    """
    Трейнер для задачи link prediction на графе Ethereum транзакций

    Параметры:
        model: nn.Module с методом forward(x, edge_index, edge_label_index) -> Tensor
        optimizer: torch.optim.Optimizer
        criterion: функция потерь (например, BCEWithLogitsLoss)
        device: torch.device
        loader_kwargs: dict с параметрами для LinkNeighborLoader (batch_size, num_neighbors и т.д.)
    """

    def __init__(self, model, optimizer, criterion, device=None, **loader_kwargs):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.loader_kwargs = loader_kwargs
        self.history = {'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': []}

    def fit(self, train_data, val_data=None, epochs=100, verbose=True, eval_metrics=True):
        """
        Обучение модели

        Параметры:
            train_data: данные для обучения (после RandomLinkSplit)
            val_data: данные для валидации
            epochs: количество эпох
            verbose: печатать ли прогресс
            eval_metrics: вычислять ли AUC метрики
        """
        train_loader = LinkNeighborLoader(
            train_data,
            edge_label_index=train_data.edge_label_index,
            edge_label=train_data.edge_label,
            shuffle=True,
            **self.loader_kwargs,
        )

        for epoch in range(1, epochs + 1):
            # Обучение
            self.model.train()
            total_loss = 0.0
            for batch in train_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(batch.x, batch.edge_index, batch.edge_label_index)
                loss = self.criterion(out.flatten(), batch.edge_label.float())
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * batch.num_graphs

            avg_train_loss = total_loss / len(train_data)
            self.history['train_loss'].append(avg_train_loss)

            # Валидация
            if val_data is not None:
                val_loss = self.evaluate_loss(val_data)
                self.history['val_loss'].append(val_loss)

                if eval_metrics:
                    train_auc = self.evaluate_auc(train_data)
                    val_auc = self.evaluate_auc(val_data)
                    self.history['train_auc'].append(train_auc)
                    self.history['val_auc'].append(val_auc)

                    if verbose:
                        print(f'Epoch {epoch:03d}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                              f'Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}')
                elif verbose:
                    print(f'Epoch {epoch:03d}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')
            elif verbose:
                print(f'Epoch {epoch:03d}: Train Loss: {avg_train_loss:.4f}')

        return self.history

    def evaluate_loss(self, data):
        """
        Вычисляет средний лосс на наборе данных (без градиентов)
        """
        self.model.eval()
        with torch.no_grad():
            loader = LinkNeighborLoader(
                data,
                edge_label_index=data.edge_label_index,
                edge_label=data.edge_label,
                shuffle=False,
                **{k: v for k, v in self.loader_kwargs.items() if k != 'shuffle'}
            )
            total_loss = 0.0
            for batch in loader:
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.edge_index, batch.edge_label_index)
                loss = self.criterion(out.flatten(), batch.edge_label.float())
                total_loss += loss.item() * batch.num_graphs
        return total_loss / len(data)

    def evaluate_auc(self, data):
        """
        Вычисляет AUC-ROC для бинарной классификации рёбер
        """
        self.model.eval()
        with torch.no_grad():
            loader = LinkNeighborLoader(
                data,
                edge_label_index=data.edge_label_index,
                edge_label=data.edge_label,
                shuffle=False,
                **{k: v for k, v in self.loader_kwargs.items() if k != 'shuffle'}
            )

            all_preds = []
            all_labels = []

            for batch in loader:
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.edge_index, batch.edge_label_index)
                probs = torch.sigmoid(out).flatten().cpu().numpy()
                labels = batch.edge_label.cpu().numpy()

                all_preds.extend(probs)
                all_labels.extend(labels)

            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)

            if len(np.unique(all_labels)) > 1:
                auc_roc = roc_auc_score(all_labels, all_preds)
            else:
                auc_roc = 0.5

        return auc_roc

    def predict(self, data, edge_label_index=None):
        """
        Возвращает вероятности для заданных пар вершин
        """
        if edge_label_index is None:
            edge_label_index = data.edge_label_index

        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            out = self.model(data.x, data.edge_index, edge_label_index.to(self.device))
            probs = torch.sigmoid(out).cpu()
        return probs.numpy().flatten()

    def save_model(self, path):
        """Сохраняет модель и историю обучения"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
        }, path)
        print(f"Модель сохранена в {path}")

    def load_model(self, path):
        """Загружает модель"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        print(f"Модель загружена из {path}")