import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay
from copy import deepcopy
from torch_geometric.utils import degree, softmax


class CustomMessagePassingLayer(MessagePassing):
    """
    Гибкий слой передачи сообщений.

    Параметры:
        in_dim: входная размерность признаков узлов.
        out_dim: выходная размерность.
        message_fn: функция сообщения (принимает x_j, x_i, edge_attr, index).
        aggr: тип агрегации ('add', 'mean', 'max' или объект Aggregation).
        update_fn: функция обновления (принимает aggr_out, x).
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
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, index=edge_index[1])

    def message(self, x_i, x_j, edge_attr=None, index=None):
        if self._custom_message:
            return self.message_fn(x_j, x_i, edge_attr, index)
        return self.message_lin(x_j)

    def update(self, aggr_out, x):
        if self._custom_update:
            return self.update_fn(aggr_out, x)
        return aggr_out


class LinkPredictionMessagePassingModel(nn.Module):
    """
    Модель предсказания связей на основе message passing.

    Параметры:
        in_channels: размерность входных признаков.
        hidden_channels: размерность скрытых слоёв.
        out_channels: размерность выходных эмбеддингов.
        num_layers: количество слоёв.
        message_fn: список message-функций для каждого слоя.
        aggr: строка или объект агрегации.
        update_fn: список update-функций для каждого слоя.
        decoder_fn: функция декодирования (по умолчанию скалярное произведение).
        dropout: вероятность dropout.
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

        if message_fn is None:
            message_fn = [None] * num_layers
        elif not isinstance(message_fn, (list, tuple)):
            message_fn = [message_fn] * num_layers

        if update_fn is None:
            update_fn = [None] * num_layers
        elif not isinstance(update_fn, (list, tuple)):
            update_fn = [update_fn] * num_layers

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.layers.append(
            CustomMessagePassingLayer(in_channels, hidden_channels[0],
                                      message_fn=message_fn[0], aggr=aggr,
                                      update_fn=update_fn[0])
        )
        self.norms.append(nn.LayerNorm(hidden_channels[0]))

        for i in range(num_layers - 1):
            self.layers.append(
                CustomMessagePassingLayer(hidden_channels[i], hidden_channels[i + 1],
                                          message_fn=message_fn[i + 1], aggr=aggr,
                                          update_fn=update_fn[i + 1])
            )
            self.norms.append(nn.LayerNorm(hidden_channels[i + 1]))

        self.proj = nn.Linear(hidden_channels[-1], out_channels)

    def default_decoder(self, z, edge_label_index):
        row, col = edge_label_index
        return (z[row] * z[col]).sum(dim=-1)

    def encode(self, x, edge_index, edge_attr=None):
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            if hasattr(layer, 'message_fn') and getattr(layer.message_fn, 'is_sym_norm', False):
                row, col = edge_index
                deg = degree(col, x.size(0), dtype=x.dtype)
                deg_inv_sqrt = deg.pow(-0.5)
                deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
                edge_attr = (deg_inv_sqrt[row] * deg_inv_sqrt[col]).unsqueeze(-1)
            else:
                edge_attr = None
            x = layer(x, edge_index, edge_attr)
            x = norm(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        z = self.proj(x)
        return z

    def decode(self, z, edge_label_index):
        return self.decoder_fn(z, edge_label_index)

    def forward(self, x, edge_index, edge_label_index, edge_attr=None):
        z = self.encode(x, edge_index, edge_attr)
        return self.decode(z, edge_label_index)


class LinkPredictor:
    """
    Пайплайн предсказания связей с использованием готовых загрузчиков.

    Параметры:
        train_loader: DataLoader для обучения.
        val_loader: DataLoader для валидации.
        test_loader: DataLoader для тестирования.
        model_class: класс модели.
        model_params: словарь параметров для инициализации модели.
        lr: learning rate.
        weight_decay: коэффициент регуляризации.
        epochs: число эпох.
        patience: терпение для ранней остановки (по валидационному AUCPR).
    """
    def __init__(self, train_loader, val_loader, test_loader,
                 model_class, model_params,
                 lr=0.01, weight_decay=5e-4,
                 epochs=200, patience=20):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = model_class(**model_params).to(self.device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.history = {'train_loss': [], 'val_loss': [],
                        'train_fpr': [], 'val_fpr': [],
                        'train_auprc': [], 'val_auprc': []}

    def _eval_loader(self, loader):
        """Возвращает FPR, AUCPR и средний loss для загрузчика."""
        self.model.eval()
        all_probs, all_labels = [], []
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

        if len(all_labels.unique()) < 2:
            auprc = 0.0
            fpr = 0.0
        else:
            auprc = average_precision_score(all_labels, all_probs)
            preds = (all_probs > 0.5).float()
            neg_mask = (all_labels == 0)
            fpr = (preds[neg_mask].sum() / neg_mask.sum()).item() if neg_mask.sum() > 0 else 0.0
        return fpr, auprc, mean_loss

    def run(self):
        """Запускает обучение, валидацию и тестирование. Early stopping по AUCPR."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_val_auprc = 0.0
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

            train_fpr, train_auprc, _ = self._eval_loader(self.train_loader)
            val_fpr, val_auprc, val_loss = self._eval_loader(self.val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_fpr'].append(train_fpr)
            self.history['val_fpr'].append(val_fpr)
            self.history['train_auprc'].append(train_auprc)
            self.history['val_auprc'].append(val_auprc)

            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                      f"train_fpr={train_fpr:.4f}, val_fpr={val_fpr:.4f}, "
                      f"train_auprc={train_auprc:.4f}, val_auprc={val_auprc:.4f}")

            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                best_state = deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch} (best val AUCPR = {best_val_auprc:.4f})")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"Best validation AUCPR: {best_val_auprc:.4f}")

        test_fpr, test_auprc, _ = self._eval_loader(self.test_loader)
        print(f"Test FPR: {test_fpr:.4f}, Test AUCPR: {test_auprc:.4f}")
        self.test_fpr = test_fpr
        self.test_auprc = test_auprc

        self.plot_learning_curves()
        return self.model, test_fpr, test_auprc

    def plot_learning_curves(self):
        """Рисует кривые обучения: FPR и AUCPR."""
        epochs = range(1, len(self.history['train_fpr']) + 1)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history['train_fpr'], label='Train FPR')
        plt.plot(epochs, self.history['val_fpr'], label='Validation FPR')
        plt.xlabel('Epoch')
        plt.ylabel('FPR')
        plt.title('FPR')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history['train_auprc'], label='Train AUCPR')
        plt.plot(epochs, self.history['val_auprc'], label='Validation AUCPR')
        plt.xlabel('Epoch')
        plt.ylabel('AUCPR')
        plt.title('AUCPR')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_test_pr_curve(self):
        """Строит Precision-Recall кривую для тестового набора."""
        self.model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in self.test_loader:
                batch = batch.to(self.device)
                logits = self.model(batch.x, batch.edge_index, batch.edge_label_index)
                probs = torch.sigmoid(logits).cpu()
                all_probs.append(probs)
                all_labels.append(batch.edge_label.cpu())
        all_probs = torch.cat(all_probs)
        all_labels = torch.cat(all_labels)
        PrecisionRecallDisplay.from_predictions(all_labels, all_probs)
        plt.title('Test Precision-Recall curve')
        plt.show()
