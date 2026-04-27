import torch
import torch.nn as nn
from torch_geometric.loader import LinkNeighborLoader

class LinkPredictor:
    def __init__(self, model, optimizer, criterion, device=None, use_edge_attr=False, **loader_kwargs):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.use_edge_attr = use_edge_attr
        self.loader_kwargs = loader_kwargs
        self.history = {'train_loss': [], 'val_loss': []}

    def fit(self, train_data, val_data=None, epochs=100, verbose=True):
        train_loader = LinkNeighborLoader(
            train_data,
            edge_label_index=train_data.edge_label_index,
            edge_label=train_data.edge_label,
            **self.loader_kwargs,
        )

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            total_edges = 0
            for batch in train_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                x = batch.x
                edge_index = batch.edge_index
                edge_label_index = batch.edge_label_index
                if self.use_edge_attr:
                    out = self.model(x, edge_index, edge_label_index, batch.edge_attr)
                else:
                    out = self.model(x, edge_index, edge_label_index)
                loss = self.criterion(out.flatten(), batch.edge_label.float())
                loss.backward()
                self.optimizer.step()
                num_edges = batch.edge_label.size(0)
                total_loss += loss.item() * num_edges
                total_edges += num_edges

            avg_train_loss = total_loss / total_edges
            self.history['train_loss'].append(avg_train_loss)

            if val_data is not None:
                val_loss = self.evaluate_loss(val_data)
                self.history['val_loss'].append(val_loss)
                if verbose:
                    print(f'Epoch {epoch:03d}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')
            elif verbose:
                print(f'Epoch {epoch:03d}: Train Loss: {avg_train_loss:.4f}')

        return self.history

    def evaluate_loss(self, data):
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
            total_edges = 0
            for batch in loader:
                batch = batch.to(self.device)
                x = batch.x
                edge_index = batch.edge_index
                edge_label_index = batch.edge_label_index
                if self.use_edge_attr:
                    out = self.model(x, edge_index, edge_label_index, batch.edge_attr)
                else:
                    out = self.model(x, edge_index, edge_label_index)
                loss = self.criterion(out.flatten(), batch.edge_label.float())
                num_edges = batch.edge_label.size(0)
                total_loss += loss.item() * num_edges
                total_edges += num_edges

        return total_loss / total_edges

    def predict(self, data):
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            x = data.x
            edge_index = data.edge_index
            edge_label_index = data.edge_label_index
            if self.use_edge_attr:
                out = self.model(x, edge_index, edge_label_index, data.edge_attr)
            else:
                out = self.model(x, edge_index, edge_label_index)
            probs = torch.sigmoid(out).cpu()
        return probs.numpy().flatten()