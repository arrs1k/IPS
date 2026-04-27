import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import numpy as np


class EthereumLinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=64, num_layers=2, dropout=0.3):
        super().__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(nn.Linear(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        for i in range(num_layers - 1):
            self.convs.append(nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.lin = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def encode(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        return x

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)


class EthereumLinkPredictionTrainer:
    def __init__(self, model, optimizer, criterion, device=None, batch_size=1024, **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.batch_size = batch_size
        self.history = {'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': []}

    def fit(self, train_data, val_data=None, epochs=100, verbose=True, eval_metrics=True):
        pos_mask = train_data.edge_label == 1
        neg_mask = train_data.edge_label == 0

        pos_edges = train_data.edge_label_index[:, pos_mask]
        neg_edges = train_data.edge_label_index[:, neg_mask]

        all_edges = torch.cat([pos_edges, neg_edges], dim=1)
        all_labels = torch.cat([torch.ones(pos_edges.size(1)), torch.zeros(neg_edges.size(1))])
        perm = torch.randperm(all_edges.size(1))
        all_edges = all_edges[:, perm]
        all_labels = all_labels[perm]

        dataset = TensorDataset(all_edges.t(), all_labels)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        x = train_data.x.to(self.device)
        edge_index = train_data.edge_index.to(self.device)

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            for batch_edges, batch_labels in train_loader:
                batch_edges = batch_edges.t().to(self.device)
                batch_labels = batch_labels.to(self.device)

                self.optimizer.zero_grad()
                out = self.model(x, edge_index, batch_edges)
                loss = self.criterion(out.flatten(), batch_labels.float())
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            self.history['train_loss'].append(avg_train_loss)

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
        self.model.eval()
        with torch.no_grad():
            pos_mask = data.edge_label == 1
            neg_mask = data.edge_label == 0

            pos_edges = data.edge_label_index[:, pos_mask]
            neg_edges = data.edge_label_index[:, neg_mask]

            all_edges = torch.cat([pos_edges, neg_edges], dim=1)
            all_labels = torch.cat([torch.ones(pos_edges.size(1)), torch.zeros(neg_edges.size(1))])

            x = data.x.to(self.device)
            edge_index = data.edge_index.to(self.device)

            total_loss = 0.0
            num_batches = 0
            for i in range(0, all_edges.size(1), self.batch_size):
                batch_edges = all_edges[:, i:i + self.batch_size].to(self.device)
                batch_labels = all_labels[i:i + self.batch_size].to(self.device)

                out = self.model(x, edge_index, batch_edges)
                loss = self.criterion(out.flatten(), batch_labels.float())
                total_loss += loss.item()
                num_batches += 1

            return total_loss / num_batches if num_batches > 0 else 0

    def evaluate_auc(self, data):
        self.model.eval()
        with torch.no_grad():
            pos_mask = data.edge_label == 1
            neg_mask = data.edge_label == 0

            pos_edges = data.edge_label_index[:, pos_mask]
            neg_edges = data.edge_label_index[:, neg_mask]

            all_edges = torch.cat([pos_edges, neg_edges], dim=1)
            all_labels = torch.cat([torch.ones(pos_edges.size(1)), torch.zeros(neg_edges.size(1))])

            x = data.x.to(self.device)
            edge_index = data.edge_index.to(self.device)

            all_preds = []
            all_labels_list = []

            for i in range(0, all_edges.size(1), self.batch_size):
                batch_edges = all_edges[:, i:i + self.batch_size].to(self.device)
                batch_labels = all_labels[i:i + self.batch_size]

                out = self.model(x, edge_index, batch_edges)
                probs = torch.sigmoid(out).flatten().cpu().numpy()

                all_preds.extend(probs)
                all_labels_list.extend(batch_labels.numpy())

            all_preds = np.array(all_preds)
            all_labels_list = np.array(all_labels_list)

            if len(np.unique(all_labels_list)) > 1:
                return roc_auc_score(all_labels_list, all_preds)
            return 0.5

    def predict(self, data, edge_label_index=None):
        if edge_label_index is None:
            edge_label_index = data.edge_label_index

        self.model.eval()
        with torch.no_grad():
            x = data.x.to(self.device)
            edge_index = data.edge_index.to(self.device)
            out = self.model(x, edge_index, edge_label_index.to(self.device))
            probs = torch.sigmoid(out).cpu()
        return probs.numpy().flatten()

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
        }, path)
        print(f"Модель сохранена в {path}")

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        print(f"Модель загружена из {path}")