# aggregate_update_methods.py
"""
Реализация 5 методов aggregate и 2 методов update для gnn.
Используется в linkpredictor.py для экспериментов с разными архитектурами.

Методы aggregate:
    1. mean - классическое нормализованное обновление (среднее соседей)
    2. sym_norm - симметрическая нормализация (kipf & welling, gcn)
    3. janossy - janossy pooling с lstm
    4. conv - свёрточная агрегация с обучаемыми весами
    5. attention - attention-агрегация (gat style)

Методы update:
    1. self_loop - классическое self-loop обновление (aggr_out + x)
    2. gru - gru-обновление (динамическая балансировка)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.transforms import RandomLinkSplit
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


# ============================================================================
# базовый слой с конфигурируемыми aggregate и update
# ============================================================================

class ConfigurableGNNLayer(MessagePassing):
    """
    Универсальный слой, поддерживающий все комбинации aggregate и update.
    
    параметры:
        in_dim: размерность входных признаков
        out_dim: размерность выходных признаков
        aggregate_method: метод агрегации ('mean', 'sym_norm', 'janossy', 'conv', 'attention')
        update_method: метод обновления ('self_loop', 'gru')
        attention_heads: количество голов в attention (для метода 'attention')
        janossy_hidden: размерность скрытого слоя в janossy pooling
    """
    
    def __init__(self, in_dim, out_dim, 
                 aggregate_method='mean', 
                 update_method='self_loop',
                 attention_heads=4,
                 janossy_hidden=64):
        super().__init__(aggr='add' if aggregate_method in ['sym_norm', 'conv', 'attention'] else 'mean')
        
        self.aggregate_method = aggregate_method
        self.update_method = update_method
        self.attention_heads = attention_heads
        self.janossy_hidden = janossy_hidden
        
        # базовое линейное преобразование
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        
        # для sym_norm (кэш для нормализации)
        self.norm = None
        
        # для conv (обучаемый вес)
        if aggregate_method == 'conv':
            self.conv_weight = nn.Parameter(torch.ones(1))
        
        # для attention
        if aggregate_method == 'attention':
            self.head_dim = out_dim // attention_heads
            self.att_lin = nn.Linear(out_dim, attention_heads, bias=False)
        
        # для janossy (lstm для агрегации)
        if aggregate_method == 'janossy':
            self.janossy_lin = nn.Linear(in_dim, janossy_hidden)
            self.janossy_lstm = nn.LSTM(janossy_hidden, janossy_hidden, 
                                         batch_first=True, bidirectional=True)
            self.janossy_out = nn.Linear(2 * janossy_hidden, out_dim)
        
        # для gru update
        if update_method == 'gru':
            self.gru = nn.GRUCell(out_dim, out_dim)
    
    def forward(self, x, edge_index):
        """Прямой проход слоя."""
        # предварительные вычисления для sym_norm
        if self.aggregate_method == 'sym_norm':
            row, col = edge_index
            deg = degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            self.norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # janossy pooling требует особой обработки
        if self.aggregate_method == 'janossy':
            return self._janossy_forward(x, edge_index)
        
        # стандартный propagate для остальных методов
        return self.propagate(edge_index, x=x)
    
    def message(self, x_i, x_j, index):
        """Вычисление сообщений от соседей (aggregate)."""
        
        if self.aggregate_method == 'mean':
            # 1. среднее соседей (нормализованное)
            return self.lin(x_j)
        
        elif self.aggregate_method == 'sym_norm':
            # 2. симметрическая нормализация (как в gcn)
            return self.lin(x_j) * self.norm.view(-1, 1)
        
        elif self.aggregate_method == 'conv':
            # 3. свёрточная агрегация с обучаемым весом
            return self.lin(x_j) * self.conv_weight
        
        elif self.aggregate_method == 'attention':
            # 4. attention-агрегация (gat style)
            x_j_proj = self.lin(x_j)
            x_i_proj = self.lin(x_i)
            
            # вычисляем attention веса
            att = (x_i_proj * x_j_proj).sum(dim=-1, keepdim=True)
            att = F.leaky_relu(att)
            att = self.att_lin(att)
            
            # нормализация по соседям
            att = torch.exp(att)
            att_sum = self.propagate(index, x=att, aggr='mean')
            att = att / (att_sum + 1e-8)
            
            return x_j_proj * att
        
        else:
            return self.lin(x_j)
    
    def update(self, aggr_out, x):
        """Обновление эмбеддингов вершин (update)."""
        
        if self.update_method == 'self_loop':
            # 1. self-loop обновление (aggr_out + x)
            return aggr_out + x
        
        elif self.update_method == 'gru':
            # 2. gru-обновление
            return self.gru(aggr_out, x)
        
        else:
            return aggr_out + x
    
    def _janossy_forward(self, x, edge_index):
        """
        Janossy pooling через lstm.
        Обрабатывает соседей как последовательность.
        """
        row, col = edge_index
        
        # собираем соседей для каждой вершины
        neighbors = [[] for _ in range(x.size(0))]
        for r, c in zip(row.tolist(), col.tolist()):
            neighbors[r].append(c)
        
        outputs = []
        for node in range(x.size(0)):
            if len(neighbors[node]) == 0:
                outputs.append(torch.zeros(self.lin.out_features, device=x.device))
                continue
            
            # применяем линейное преобразование к соседям
            neighbor_embs = self.janossy_lin(x[neighbors[node]])
            neighbor_embs = F.relu(neighbor_embs).unsqueeze(0)
            
            # lstm агрегация
            lstm_out, _ = self.janossy_lstm(neighbor_embs)
            pooled = lstm_out.mean(dim=1).squeeze(0)
            out = self.janossy_out(pooled)
            outputs.append(out)
        
        aggr_out = torch.stack(outputs, dim=0)
        
        # применяем update
        if self.update_method == 'self_loop':
            return aggr_out + x
        elif self.update_method == 'gru':
            return self.gru(aggr_out, x)
        else:
            return aggr_out + x


# ============================================================================
# гибкая gnn модель
# ============================================================================

class FlexibleGNNModel(nn.Module):
    """
    Гибкая gnn модель для link prediction с конфигурируемыми aggregate и update.
    
    параметры:
        in_channels: размерность входных признаков
        hidden_channels: размерность скрытых слоёв (int или list)
        out_channels: размерность выходных эмбеддингов
        num_layers: количество слоёв
        aggregate_method: метод агрегации
        update_method: метод обновления
        attention_heads: количество голов в attention
        janossy_hidden: скрытая размерность для janossy
        dropout: вероятность dropout
    """
    
    def __init__(self, in_channels, hidden_channels=64, out_channels=64, num_layers=2,
                 aggregate_method='mean', update_method='self_loop',
                 attention_heads=4, janossy_hidden=64, dropout=0.0):
        super().__init__()
        
        self.aggregate_method = aggregate_method
        self.update_method = update_method
        self.dropout = dropout
        self.num_layers = num_layers
        
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels] * num_layers
        
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # входной слой
        self.layers.append(
            ConfigurableGNNLayer(
                in_channels, hidden_channels[0],
                aggregate_method=aggregate_method,
                update_method=update_method,
                attention_heads=attention_heads,
                janossy_hidden=janossy_hidden
            )
        )
        self.norms.append(nn.LayerNorm(hidden_channels[0]))
        
        # скрытые слои
        for i in range(num_layers - 1):
            self.layers.append(
                ConfigurableGNNLayer(
                    hidden_channels[i], hidden_channels[i + 1],
                    aggregate_method=aggregate_method,
                    update_method=update_method,
                    attention_heads=attention_heads,
                    janossy_hidden=janossy_hidden
                )
            )
            self.norms.append(nn.LayerNorm(hidden_channels[i + 1]))
        
        # выходная проекция
        self.proj = nn.Linear(hidden_channels[-1], out_channels)
    
    def forward(self, x, edge_index, edge_label_index):
        """Прямой проход для предсказания связей."""
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)
    
    def encode(self, x, edge_index):
        """Получение эмбеддингов вершин."""
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = layer(x, edge_index)
            x = norm(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.proj(x)
    
    def decode(self, z, edge_label_index):
        """Декодирование эмбеддингов в вероятности связей."""
        row, col = edge_label_index
        return (z[row] * z[col]).sum(dim=-1)


# ============================================================================
# функции для сравнения методов
# ============================================================================

def compare_all_combinations(data, results_file='comparison_results.csv', epochs=50):
    """
    Сравнивает все 5x2 = 10 комбинаций методов aggregate и update.
    
    параметры:
        data: граф (torch_geometric.data.data)
        results_file: имя файла для сохранения результатов
        epochs: количество эпох обучения
    
    возвращает:
        dataframe с результатами всех комбинаций
    """
    from LinkPredictor import LinkPredictor
    
    aggregate_methods = ['mean', 'sym_norm', 'janossy', 'conv', 'attention']
    update_methods = ['self_loop', 'gru']
    
    results = []
    histories = {}
    
    print("\n" + "=" * 80)
    print("Сравнение всех комбинаций aggregate и update")
    print("=" * 80)
    print(f"Всего комбинаций: {len(aggregate_methods) * len(update_methods)}")
    print(f"Эпох на модель: {epochs}")
    print("=" * 80)
    
    for agg in aggregate_methods:
        for upd in update_methods:
            combo_name = f"{agg}_{upd}"
            print(f"\nТестирование: agg={agg}, update={upd}")
            
            model_params = {
                'in_channels': data.x.shape[1],
                'hidden_channels': 64,
                'out_channels': 64,
                'num_layers': 2,
                'aggregate_method': agg,
                'update_method': upd,
                'dropout': 0.3,
            }
            
            try:
                predictor = LinkPredictor(
                    data=data,
                    model_class=FlexibleGNNModel,
                    model_params=model_params,
                    epochs=epochs,
                    patience=epochs // 3,
                )
                
                model, test_auc = predictor.run()
                
                result = {
                    'aggregate': agg,
                    'update': upd,
                    'test_auc': test_auc,
                    'best_val_auc': max(predictor.history['val_auc']) if predictor.history['val_auc'] else 0,
                    'best_val_loss': min(predictor.history['val_loss']) if predictor.history['val_loss'] else 0,
                    'epochs_completed': len(predictor.history['train_loss']),
                }
                results.append(result)
                histories[combo_name] = predictor.history
                
                print(f"   test auc: {test_auc:.4f}, лучший val auc: {result['best_val_auc']:.4f}")
                
            except Exception as e:
                print(f"   Ошибка: {e}")
                results.append({
                    'aggregate': agg,
                    'update': upd,
                    'test_auc': 0,
                    'error': str(e),
                })
                histories[combo_name] = None
    
    # сохраняем результаты
    df_results = pd.DataFrame(results)
    df_results.to_csv(results_file, index=False)
    print(f"\nРезультаты сохранены в: {results_file}")
    
    return df_results, histories


def plot_comparison_results(results, save_path=None):
    """
    Визуализация результатов сравнения методов.
    
    параметры:
        results: dataframe с результатами или путь к csv файлу
        save_path: путь для сохранения графика
    """
    if isinstance(results, str):
        df = pd.read_csv(results)
    else:
        df = results.copy()
    
    if 'error' in df.columns:
        df = df[df['error'].isna()]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # график 1: барчарт test auc
    ax1 = axes[0]
    combinations = [f"{row['aggregate']}\n{row['update']}" for _, row in df.iterrows()]
    aucs = df['test_auc'].values
    
    colors = ['green' if auc >= 0.8 else 'orange' if auc >= 0.7 else 'red' for auc in aucs]
    bars = ax1.bar(range(len(combinations)), aucs, color=colors, edgecolor='black', linewidth=1.5)
    
    for bar, auc in zip(bars, aucs):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{auc:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.set_xlabel('Комбинация (aggregate + update)', fontsize=12)
    ax1.set_ylabel('test auc', fontsize=12)
    ax1.set_title('Сравнение test auc для всех комбинаций', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(combinations)))
    ax1.set_xticklabels(combinations, rotation=45, ha='right')
    ax1.set_ylim([0.5, 1.0])
    ax1.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='хорошо (0.7)')
    ax1.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='отлично (0.8)')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # график 2: тепловая карта
    ax2 = axes[1]
    agg_methods = df['aggregate'].unique()
    upd_methods = df['update'].unique()
    
    heatmap_data = np.zeros((len(agg_methods), len(upd_methods)))
    for i, agg in enumerate(agg_methods):
        for j, upd in enumerate(upd_methods):
            row = df[(df['aggregate'] == agg) & (df['update'] == upd)]
            if len(row) > 0:
                heatmap_data[i, j] = row['test_auc'].values[0]
    
    im = ax2.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=0.9)
    
    ax2.set_xticks(range(len(upd_methods)))
    ax2.set_yticks(range(len(agg_methods)))
    ax2.set_xticklabels(upd_methods)
    ax2.set_yticklabels([m.upper() for m in agg_methods])
    ax2.set_xlabel('Метод update', fontsize=12)
    ax2.set_ylabel('Метод aggregate', fontsize=12)
    ax2.set_title('Тепловая карта test auc', fontsize=14, fontweight='bold')
    
    for i in range(len(agg_methods)):
        for j in range(len(upd_methods)):
            ax2.text(j, i, f'{heatmap_data[i, j]:.4f}',
                    ha="center", va="center", color="black", fontsize=10, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('test auc', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_all_learning_curves(histories, save_path=None):
    """
    Рисует learning curves (auc) для всех комбинаций.
    
    параметры:
        histories: словарь {combo_name: history}
        save_path: путь для сохранения графика
    """
    n_combos = len(histories)
    n_cols = 5
    n_rows = (n_combos + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()
    
    for idx, (combo_name, history) in enumerate(histories.items()):
        if history is None:
            continue
            
        ax = axes[idx]
        
        if 'train_auc' in history:
            ax.plot(history['train_auc'], label='train auc', linewidth=2, color='blue')
        if 'val_auc' in history:
            ax.plot(history['val_auc'], label='val auc', linewidth=2, color='orange')
        
        best_val = max(history['val_auc']) if history['val_auc'] else 0
        ax.set_title(f'{combo_name}\nЛучший val auc: {best_val:.4f}', fontsize=10)
        ax.set_xlabel('Эпоха')
        ax.set_ylabel('auc')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.5, 1.0])
    
    # скрываем лишние подграфики
    for idx in range(len(histories), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('кривые обучения для всех комбинаций aggregate-update', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def train_and_plot_all_combinations(data, epochs=30, save_dir='training_plots'):
    """
    Обучает все комбинации и сохраняет результаты.
    
    параметры:
        data: граф (torch_geometric.data.data)
        epochs: количество эпох
        save_dir: директория для сохранения графиков
    """
    # создаём директорию для сохранения
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"{save_dir}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nРезультаты будут сохранены в: {save_dir}/")
    
    # сравниваем все комбинации
    df_results, histories = compare_all_combinations(data, 
                                                      results_file=f'{save_dir}/results.csv',
                                                      epochs=epochs)
    
    # сохраняем и отображаем learning curves
    plot_all_learning_curves(histories, save_path=f'{save_dir}/learning_curves.png')
    
    # сохраняем и отображаем сравнение
    plot_comparison_results(df_results, save_path=f'{save_dir}/comparison.png')
    
    # вывод лучшей комбинации
    best_idx = df_results['test_auc'].idxmax()
    best = df_results.loc[best_idx]
    
    print("\n" + "=" * 80)
    print("Лучшая комбинация методов")
    print("=" * 80)
    print(f"   метод aggregate: {best['aggregate'].upper()}")
    print(f"   метод update: {best['update'].upper()}")
    print(f"   test auc: {best['test_auc']:.4f}")
    if 'best_val_auc' in best:
        print(f"   Лучший val auc: {best['best_val_auc']:.4f}")
    
    # рейтинг всех методов
    print("\n" + "=" * 80)
    print("Рейтинг методов по test auc")
    print("=" * 80)
    ranking = df_results.sort_values('test_auc', ascending=False)
    for i, row in ranking.iterrows():
        rank = f"{i+1}."
        print(f"   {rank:3} {row['aggregate'].upper():12} + {row['update'].upper():12} : {row['test_auc']:.4f}")
    
    # сохраняем лучшую модель (если есть доступ к данным)
    try:
        best_combo_name = f"{best['aggregate']}_{best['update']}"
        print(f"\nЛучшая модель: {best_combo_name}")
        print(f"Результаты сохранены в: {save_dir}/")
    except:
        pass
    
    return df_results, histories


def create_model_summary_table(df_results):
    """
    Создаёт сводную таблицу с результатами.
    
    параметры:
        df_results: dataframe с результатами
    
    возвращает:
        dataframe со сводной статистикой
    """
    summary = []
    
    # статистика по aggregate методам
    agg_stats = df_results.groupby('aggregate')['test_auc'].agg(['mean', 'std', 'max']).round(4)
    for agg in agg_stats.index:
        summary.append({
            'категория': 'aggregate',
            'метод': agg,
            'средний auc': agg_stats.loc[agg, 'mean'],
            'std auc': agg_stats.loc[agg, 'std'],
            'макс auc': agg_stats.loc[agg, 'max'],
        })
    
    # статистика по update методам
    upd_stats = df_results.groupby('update')['test_auc'].agg(['mean', 'std', 'max']).round(4)
    for upd in upd_stats.index:
        summary.append({
            'категория': 'update',
            'метод': upd,
            'средний auc': upd_stats.loc[upd, 'mean'],
            'std auc': upd_stats.loc[upd, 'std'],
            'макс auc': upd_stats.loc[upd, 'max'],
        })
    
    return pd.DataFrame(summary)


# ============================================================================
# вспомогательные функции
# ============================================================================

def create_model(aggregate_method, update_method, in_channels, 
                 hidden_channels=64, out_channels=64, num_layers=2):
    """
    Создаёт модель с заданными методами aggregate и update.
    
    параметры:
        aggregate_method: метод агрегации
        update_method: метод обновления
        in_channels: размерность входных признаков
    
    возвращает:
        flexiblegnnmodel
    """
    return FlexibleGNNModel(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        aggregate_method=aggregate_method,
        update_method=update_method,
        dropout=0.3,
    )


def get_method_description(method_name, method_type='aggregate'):
    """
    Возвращает описание метода.
    
    параметры:
        method_name: название метода
        method_type: 'aggregate' или 'update'
    
    возвращает:
        str: описание метода
    """
    descriptions = {
        'aggregate': {
            'mean': 'классическое нормализованное обновление. сообщение = среднее значение соседей.',
            'sym_norm': 'симметрическая нормализация (как в gcn). учитывает степени вершин.',
            'janossy': 'janossy pooling через lstm. обрабатывает соседей как последовательность.',
            'conv': 'свёрточная агрегация с обучаемыми весами.',
            'attention': 'attention-агрегация (gat style). учится взвешивать соседей.',
        },
        'update': {
            'self_loop': 'классическое self-loop обновление. h_new = aggr_out + x',
            'gru': 'gru-обновление. динамическая балансировка информации.',
        }
    }
    return descriptions.get(method_type, {}).get(method_name, 'описание не найдено')


def print_methods_info():
    """Печатает информацию о всех доступных методах."""
    print("\n" + "=" * 80)
    print("Доступные методы aggregate и update")
    print("=" * 80)
    
    print("\nМетоды aggregate:")
    for method in ['mean', 'sym_norm', 'janossy', 'conv', 'attention']:
        print(f"   {method.upper():12} - {get_method_description(method, 'aggregate')}")
    
    print("\nМетоды update:")
    for method in ['self_loop', 'gru']:
        print(f"   {method.upper():12} - {get_method_description(method, 'update')}")
    
    print("\nВсего комбинаций: 5 x 2 = 10")


# ============================================================================
# пример использования
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Библиотека для тестирования методов aggregate и update")
    print("=" * 80)
    
    # вывод информации о методах
    print_methods_info()
    
    # загрузка графа (путь нужно указать)
    try:
        data = torch.load('ethereum_graph.pt', weights_only=False)
        print(f"\nГраф загружен: {data.num_nodes} вершин, {data.num_edges} рёбер")
        
        # запуск полного тестирования (раскомментируйте для выполнения)
        # df_results, histories = train_and_plot_all_combinations(
        #     data=data,
        #     epochs=30,
        #     save_dir='ethereum_experiment'
        # )
        
    except FileNotFoundError:
        print("\nФайл ethereum_graph.pt не найден.")
        print("Пожалуйста, загрузите граф или укажите правильный путь.")
        print("\nПример использования:")
        print("   from aggregate_update_methods import train_and_plot_all_combinations")
        print("   import torch")
        print("   data = torch.load('your_graph.pt')")
        print("   results = train_and_plot_all_combinations(data, epochs=30)")