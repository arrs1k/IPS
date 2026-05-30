import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from LinkPredictor import *
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
class MeanMessage(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x_j, x_i=None, edge_attr=None):
        return self.lin(x_j)


class SymNormMessage(nn.Module):
    """Требует, чтобы norm был передан как edge_attr."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x_j, x_i=None, edge_attr=None):
        # edge_attr -> norm (размер [E, 1])
        return self.lin(x_j) * edge_attr


class ConvMessage(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.conv_weight = nn.Parameter(torch.ones(1))

    def forward(self, x_j, x_i=None, edge_attr=None):
        return self.lin(x_j) * self.conv_weight


class AttentionMessage(nn.Module):
    """Нуждается в index, который передаётся через propagate."""
    def __init__(self, in_dim, out_dim, heads=4):
        super().__init__()
        self.out_dim = out_dim          # ← сохраняем для использования в forward
        self.heads = heads
        self.head_dim = out_dim // heads
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.att_lin = nn.Linear(out_dim, heads, bias=False)

    def forward(self, x_j, x_i, edge_attr=None, index=None):
        x_j_proj = self.lin(x_j)        # [E, out_dim]
        x_i_proj = self.lin(x_i)
        att = (x_i_proj * x_j_proj).sum(dim=-1, keepdim=True)  # [E, 1]
        att = F.leaky_relu(att)
        att = self.att_lin(att)          # [E, heads]
        att = softmax(att, index)        # index – номера получателей
        # Взвешиваем и возвращаем плоский тензор [E, out_dim]
        return (x_j_proj.view(-1, self.heads, self.head_dim) * att.unsqueeze(-1)).view(-1, self.out_dim)

class SelfLoopUpdate(nn.Module):
    """Остаточная связь с автоматической проекцией, если размерности не совпадают."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()

    def forward(self, aggr_out, x):
        return aggr_out + self.proj(x)


class GRUUpdate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gru = nn.GRUCell(dim, dim)

    def forward(self, aggr_out, x):
        return self.gru(aggr_out, x)

def _aggregate_name(agg):
    """Возвращает строковый идентификатор метода агрегации."""
    if isinstance(agg, str):
        return agg
    # если передан класс, а не экземпляр
    if isinstance(agg, type):
        cls = agg
    else:
        cls = type(agg)

    mapping = {
        MeanMessage: 'mean',
        SymNormMessage: 'sym_norm',
        ConvMessage: 'conv',
        AttentionMessage: 'attention',
    }
    return mapping.get(cls, cls.__name__)


def _update_name(upd):
    """Возвращает строковый идентификатор метода обновления."""
    if isinstance(upd, str):
        return upd
    if isinstance(upd, type):
        cls = upd
    else:
        cls = type(upd)

    mapping = {
        SelfLoopUpdate: 'self_loop',
        GRUUpdate: 'gru',
    }
    return mapping.get(cls, cls.__name__)



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

    aggregate_methods = [MeanMessage, SymNormMessage, ConvMessage, AttentionMessage]
    update_methods = [SelfLoopUpdate, GRUUpdate]
    names = []
    
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
                    model_class=LinkPredictionMessagePassingModel,
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
    return LinkPredictionMessagePassingModel(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        aggr=aggregate_method,
        update_fn=update_method,
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