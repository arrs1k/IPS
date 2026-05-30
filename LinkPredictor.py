import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import copy
from copy import deepcopy


class CustomMessagePassingLayer(MessagePassing):
    """
    Гибкий слой, который параметризуется функциями:
      - message_fn: (x_j, x_i, edge_attr, **kwargs) -> сообщение
      - aggr: объект Aggregation или строка ('add', 'mean', 'max')
      - update_fn: (aggr_out, x) -> новый вектор узла
    """

    def __init__(self, in_dim, out_dim, message_fn=None, aggr='add', update_fn=None):
        super().__init__(aggr=aggr)
        self.in_dim = in_dim
        self.out_dim = out_dim

        if message_fn is None:
            self.message_lin = nn.Linear(in_dim, out_dim, bias=False)
            self._custom_message = False
        else:
            self._custom_message = True
            self.message_fn = message_fn

        if update_fn is None:
            self._custom_update = False
        else:
            self._custom_update = True
            self.update_fn = update_fn

    def forward(self, x, edge_index, edge_attr=None):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr=None):
        if self._custom_message:
            return self.message_fn(x_j, x_i, edge_attr)
        else:
            return self.message_lin(x_j)

    def update(self, aggr_out, x):
        if self._custom_update:
            return self.update_fn(aggr_out, x)
        else:
            return aggr_out


class LinkPredictionMessagePassingModel(nn.Module):
    """
    Параметры:
      - in_channels: размерность входных признаков узлов
      - hidden_channels: размерность скрытых слоёв (можно int или список)
      - out_channels: размерность эмбеддингов узлов перед декодером
      - num_layers: количество message-passing слоёв
      - message_fn: опциональная функция для сообщений
      - aggr: агрегация (строка или Aggregation)
      - update_fn: опциональная функция обновления (aggr_out, x) -> new_x
      - decoder_fn: функция (z, edge_label_index) -> логиты (по умолчанию скалярное произведение)
      - dropout: вероятность dropout между слоями
    """

    def __init__(self, in_channels, hidden_channels=64, out_channels=64, num_layers=2,
                 message_fn=None, aggr='add', update_fn=None,
                 decoder_fn=None, dropout=0.0):
        super().__init__()

        if decoder_fn is None:
            self.decoder_fn = self.default_decoder
        else:
            self.decoder_fn = decoder_fn

        self.dropout = dropout
        self.num_layers = num_layers

        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels] * num_layers

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.layers.append(
            CustomMessagePassingLayer(in_channels, hidden_channels[0],
                                      message_fn=message_fn, aggr=aggr,
                                      update_fn=update_fn)
        )
        self.norms.append(nn.LayerNorm(hidden_channels[0]))

        for i in range(num_layers - 1):
            self.layers.append(
                CustomMessagePassingLayer(hidden_channels[i], hidden_channels[i + 1],
                                          message_fn=message_fn, aggr=aggr,
                                          update_fn=update_fn)
            )
            self.norms.append(nn.LayerNorm(hidden_channels[i + 1]))

        self.proj = nn.Linear(hidden_channels[-1], out_channels)

    def default_decoder(self, z, edge_label_index):
        """Скалярное произведение эмбеддингов двух узлов."""
        row, col = edge_label_index
        return (z[row] * z[col]).sum(dim=-1)

    def encode(self, x, edge_index, edge_attr=None):
        """Получение эмбеддингов всех узлов."""
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = layer(x, edge_index, edge_attr)
            x = norm(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        z = self.proj(x)
        return z

    def decode(self, z, edge_label_index):
        """Вычисление логитов для заданных пар узлов."""
        return self.decoder_fn(z, edge_label_index)

    def forward(self, x, edge_index, edge_label_index, edge_attr=None):
        """Прямой проход для предсказания связей."""
        z = self.encode(x, edge_index, edge_attr)
        return self.decode(z, edge_label_index)


class LinkPredictor:
    """
    Полный пайплайн для задачи link prediction с использованием LinkNeighborLoader.

    Параметры:
        data: граф (torch_geometric.data.Data)
        model_class: класс модели
        model_params: словарь параметров для модели
        val_ratio: доля рёбер для валидации
        test_ratio: доля рёбер для теста
        num_neighbors: список числа соседей на каждом уровне сэмплирования
        batch_size: размер мини-батча
        neg_sampling_ratio: отношение негативных примеров к позитивным
        lr: learning rate
        weight_decay: коэффициент регуляризации
        epochs: количество эпох
        patience: терпение для ранней остановки
    """

    def __init__(self, data, model_class, model_params,
                 val_ratio=0.1, test_ratio=0.1,
                 num_neighbors=[10, 5], batch_size=128,
                 neg_sampling_ratio=1.0,
                 lr=0.01, weight_decay=5e-4,
                 epochs=200, patience=20):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._prepare_loaders(data, val_ratio, test_ratio, num_neighbors, batch_size, neg_sampling_ratio)

        self.model = model_class(**model_params).to(self.device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience

        self.history = {'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': []}

    def _prepare_loaders(self, data, val_ratio, test_ratio, num_neighbors, batch_size, neg_sampling_ratio):
        """Генерирует разбиение рёбер и создаёт загрузчики."""
        transform = RandomLinkSplit(
            num_val=val_ratio,
            num_test=test_ratio,
            is_undirected=data.is_undirected(),
            add_negative_train_samples=True,
            neg_sampling_ratio=neg_sampling_ratio
        )
        train_data, val_data, test_data = transform(data)

        self.train_loader = LinkNeighborLoader(
            data,  # исходный граф (полный) для message passing
            num_neighbors=num_neighbors,
            edge_label_index=train_data.edge_label_index,
            edge_label=train_data.edge_label,
            batch_size=batch_size,
            shuffle=True,
            neg_sampling_ratio=0.0  # негативные примеры уже добавлены
        )

        self.val_loader = LinkNeighborLoader(
            data,
            num_neighbors=num_neighbors,
            edge_label_index=val_data.edge_label_index,
            edge_label=val_data.edge_label,
            batch_size=batch_size,
            shuffle=False,
            neg_sampling_ratio=0.0
        )

        self.test_loader = LinkNeighborLoader(
            data,
            num_neighbors=num_neighbors,
            edge_label_index=test_data.edge_label_index,
            edge_label=test_data.edge_label,
            batch_size=batch_size,
            shuffle=False,
            neg_sampling_ratio=0.0
        )

    def _eval_loader(self, loader):
        """Вычисляет AUC и loss для всех батчей загрузчика."""
        self.model.eval()
        all_probs = []
        all_labels = []
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                logits = self.model(batch.x, batch.edge_index, batch.edge_label_index)
                loss = F.binary_cross_entropy_with_logits(logits, batch.edge_label)
                total_loss += loss.item() * batch.edge_label.size(0)
                total_samples += batch.edge_label.size(0)
                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu())
                all_labels.append(batch.edge_label.cpu())

        mean_loss = total_loss / total_samples if total_samples > 0 else 0.0
        all_probs = torch.cat(all_probs)
        all_labels = torch.cat(all_labels)
        auc = roc_auc_score(all_labels, all_probs) if len(all_labels.unique()) > 1 else 0.5
        return auc, mean_loss

    def run(self):
        """Запускает обучение с валидацией, рисует кривые и считает тестовый AUC."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_val_auc = 0.0
        best_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            batches = 0

            for batch in self.train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch.x, batch.edge_index, batch.edge_label_index)
                loss = F.binary_cross_entropy_with_logits(logits, batch.edge_label)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batches += 1

            train_loss = epoch_loss / max(batches, 1)

            train_auc, _ = self._eval_loader(self.train_loader)
            val_auc, val_loss = self._eval_loader(self.val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_auc'].append(train_auc)
            self.history['val_auc'].append(val_auc)

            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                      f"train_auc={train_auc:.4f}, val_auc={val_auc:.4f}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"Best validation AUC: {best_val_auc:.4f}")

        test_auc, _ = self._eval_loader(self.test_loader)
        print(f"Test AUC: {test_auc:.4f}")
        self.test_auc = test_auc

        self.plot_learning_curves()
        return self.model, test_auc

    def plot_learning_curves(self):
        """Строит графики train/val AUC по эпохам."""
        epochs = range(1, len(self.history['train_auc']) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, self.history['train_auc'], label='Train AUC')
        plt.plot(epochs, self.history['val_auc'], label='Validation AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.title('Learning curves (AUC)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_test_roc_curve(self):
        """Строит ROC‑кривую для тестового набора."""
        from sklearn.metrics import RocCurveDisplay
        self.model.eval()
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for batch in self.test_loader:
                batch = batch.to(self.device)
                logits = self.model(batch.x, batch.edge_index, batch.edge_label_index)
                probs = torch.sigmoid(logits).cpu()
                all_probs.append(probs)
                all_labels.append(batch.edge_label.cpu())
        all_probs = torch.cat(all_probs)
        all_labels = torch.cat(all_labels)
        RocCurveDisplay.from_predictions(all_labels, all_probs)
        plt.title('Test ROC curve')
        plt.show()