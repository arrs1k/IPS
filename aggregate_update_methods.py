import pandas as pd
from LinkPredictor import *
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from sklearn.metrics import average_precision_score, roc_auc_score


class MeanMessage(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x_j, x_i=None, edge_attr=None, index=None):
        return self.lin(x_j)


class SymNormMessage(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.is_sym_norm = True

    def forward(self, x_j, x_i=None, edge_attr=None, index=None):
        return self.lin(x_j) * edge_attr


class ConvMessage(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.conv_weight = nn.Parameter(torch.ones(1))

    def forward(self, x_j, x_i=None, edge_attr=None, index=None):
        return self.lin(x_j) * self.conv_weight


class AttentionMessage(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4):
        super().__init__()
        self.out_dim = out_dim
        self.heads = heads
        self.head_dim = out_dim // heads
        self.lin = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x_j, x_i, edge_attr=None, index=None):
        x_j_proj = self.lin(x_j).view(-1, self.heads, self.head_dim)
        x_i_proj = self.lin(x_i).view(-1, self.heads, self.head_dim)
        att = (x_i_proj * x_j_proj).sum(dim=-1)
        att = softmax(att, index)
        return (x_j_proj * att.unsqueeze(-1)).view(-1, self.out_dim)


class SelfLoopUpdate(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()

    def forward(self, aggr_out, x):
        return aggr_out + self.proj(x)


class GRUUpdate(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, aggr_out, x):
        x = self.proj(x)
        return self.gru(aggr_out, x)


def _aggregate_name(agg):
    """Возвращает строковый идентификатор метода агрегации."""
    if isinstance(agg, str):
        return agg
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

def compare_all_combinations(train_loader, val_loader, test_loader,
                                 train_data, val_data, test_data,
                                 results_file='comparison_results.csv', epochs=50):
    aggregate_classes = [MeanMessage, SymNormMessage, ConvMessage, AttentionMessage]
    update_classes = [SelfLoopUpdate, GRUUpdate]
    results = []
    histories = {}
    figs = {}
    diag_results = []
    in_dim = train_data.x.shape[1]
    hidden_dim = 64
    out_dim = 64

    for AggClass in aggregate_classes:
        for UpdClass in update_classes:
            combo_name = f"{AggClass.__name__}_{UpdClass.__name__}"
            print(f"\nТестирование: agg={AggClass.__name__}, update={UpdClass.__name__}")

            if AggClass == AttentionMessage:
                msg_list = [AttentionMessage(in_dim, hidden_dim, heads=4),
                            AttentionMessage(hidden_dim, hidden_dim, heads=4)]
            else:
                msg_list = [AggClass(in_dim, hidden_dim),
                            AggClass(hidden_dim, hidden_dim)]

            if UpdClass == SelfLoopUpdate:
                upd_list = [SelfLoopUpdate(in_dim, hidden_dim),
                            SelfLoopUpdate(hidden_dim, hidden_dim)]
            else:
                upd_list = [GRUUpdate(in_dim, hidden_dim),
                            GRUUpdate(hidden_dim, hidden_dim)]

            aggr_str = 'mean' if AggClass == MeanMessage else 'add'
            model_params = {
                'in_channels': in_dim,
                'hidden_channels': hidden_dim,
                'out_channels': out_dim,
                'num_layers': 2,
                'message_fn': msg_list,
                'aggr': aggr_str,
                'update_fn': upd_list,
                'dropout': 0.3,
            }

            try:
                predictor = LinkPredictor(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    model_class=LinkPredictionMessagePassingModel,
                    model_params=model_params,
                    epochs=epochs, patience=epochs // 3
                )
                predictor.train_data = train_data
                predictor.val_data = val_data
                predictor.test_data = test_data

                model, test_fpr, test_auprc = predictor.run()
                best_val_fpr = min(predictor.history['val_fpr']) if predictor.history['val_fpr'] else 1.0
                best_val_auprc = max(predictor.history['val_auprc']) if predictor.history['val_auprc'] else 0.0
                fig, diag = diagnose_link_predictor(predictor, tolerance=0.04)
                figs[combo_name] = fig
                diag_results.append(diag)
                res = {
                    'aggregate': AggClass.__name__,
                    'update': UpdClass.__name__,
                    'test_fpr': test_fpr,
                    'test_auprc': test_auprc,
                    'best_val_fpr': best_val_fpr,
                    'best_val_auprc': best_val_auprc,
                    'epochs_completed': len(predictor.history['train_loss']),
                }
                results.append(res)
                histories[combo_name] = predictor.history
                print(f"   test fpr: {test_fpr:.4f}, test auprc: {test_auprc:.4f}, best val fpr: {best_val_fpr:.4f}")
            except Exception as e:
                print(f"   Ошибка: {e}")
                results.append({
                    'aggregate': AggClass.__name__,
                    'update': UpdClass.__name__,
                    'test_fpr': 1.0,
                    'test_auprc': 0.0,
                    'error': str(e)
                })
                histories[combo_name] = None

    df_results = pd.DataFrame(results)
    df_results.to_csv(results_file, index=False)
    print(f"\nРезультаты сохранены в: {results_file}")
    return df_results, histories, figs, diag_results

def plot_comparison_results(results, save_path=None):
    """Визуализация результатов: столбчатая диаграмма FPR и тепловая карта AUCPR."""
    if isinstance(results, str):
        df = pd.read_csv(results)
    else:
        df = results.copy()
    if 'error' in df.columns:
        df = df[df['error'].isna()]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax1 = axes[0]
    combos = [f"{r['aggregate']}\n{r['update']}" for _, r in df.iterrows()]
    fprs = df['test_fpr'].values
    colors = ['green' if f <= 0.3 else 'orange' if f <= 0.5 else 'red' for f in fprs]
    bars = ax1.bar(range(len(combos)), fprs, color=colors, edgecolor='black', linewidth=1.5)
    for bar, f in zip(bars, fprs):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                 f'{f:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax1.set_xlabel('Комбинация')
    ax1.set_ylabel('Test FPR')
    ax1.set_title('Сравнение Test FPR')
    ax1.set_xticks(range(len(combos)))
    ax1.set_xticklabels(combos, rotation=45, ha='right')
    ax1.axhline(y=0.3, color='green', linestyle='--', alpha=0.5, label='отлично ≤0.3')
    ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='приемлемо ≤0.5')
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)

    ax2 = axes[1]
    agg_methods = df['aggregate'].unique()
    upd_methods = df['update'].unique()
    heatmap = np.zeros((len(agg_methods), len(upd_methods)))
    for i, agg in enumerate(agg_methods):
        for j, upd in enumerate(upd_methods):
            row = df[(df['aggregate'] == agg) & (df['update'] == upd)]
            if len(row) > 0:
                heatmap[i, j] = row['test_auprc'].values[0]
    im = ax2.imshow(heatmap, cmap='RdYlGn', aspect='auto', vmin=0.0, vmax=1.0)
    ax2.set_xticks(range(len(upd_methods)))
    ax2.set_yticks(range(len(agg_methods)))
    ax2.set_xticklabels(upd_methods)
    ax2.set_yticklabels([m.upper() for m in agg_methods])
    ax2.set_xlabel('Метод update')
    ax2.set_ylabel('Метод aggregate')
    ax2.set_title('Тепловая карта Test AUCPR')
    for i in range(len(agg_methods)):
        for j in range(len(upd_methods)):
            ax2.text(j, i, f'{heatmap[i, j]:.4f}',
                     ha="center", va="center", color="black", fontsize=10, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Test AUCPR')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_all_learning_curves(histories, save_path=None):
    """Рисует кривые обучения (AUCPR) для всех комбинаций."""
    n_combos = len(histories)
    n_cols = 5
    n_rows = (n_combos + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()
    for idx, (combo_name, history) in enumerate(histories.items()):
        if history is None:
            continue
        ax = axes[idx]
        if 'train_auprc' in history:
            ax.plot(history['train_auprc'], label='train auprc', linewidth=2, color='blue')
        if 'val_auprc' in history:
            ax.plot(history['val_auprc'], label='val auprc', linewidth=2, color='orange')
        best_val = max(history['val_auprc']) if history['val_auprc'] else 0.0
        ax.set_title(f'{combo_name}\nЛучший val auprc: {best_val:.4f}', fontsize=10)
        ax.set_xlabel('Эпоха')
        ax.set_ylabel('AUCPR')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.0, 1.0])
    for idx in range(len(histories), len(axes)):
        axes[idx].set_visible(False)
    plt.suptitle('Кривые обучения (AUCPR)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def train_and_plot_all_combinations(train_loader, val_loader, test_loader,
                                        train_data, val_data, test_data,
                                        epochs=30, save_dir='training_plots'):
    """Обучает все комбинации, сохраняет результаты и графики."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"{save_dir}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nРезультаты будут сохранены в: {save_dir}/")

    df_results, histories, figs, diag_results = compare_all_combinations(
        train_loader, val_loader, test_loader,
        train_data, val_data, test_data,
        results_file=f'{save_dir}/results.csv',
        epochs=epochs
    )
    plot_all_learning_curves(histories, save_path=f'{save_dir}/learning_curves.png')
    plot_comparison_results(df_results, save_path=f'{save_dir}/comparison.png')

    diag_dir = os.path.join(save_dir, 'diagnostics')
    os.makedirs(diag_dir, exist_ok=True)
    for name, fig in figs.items():
        if fig is not None:
            path = os.path.join(diag_dir, f'{name}.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
    print(f"Диагностические графики сохранены в: {diag_dir}")

    best_idx = df_results['test_auprc'].idxmax()
    best = df_results.loc[best_idx]
    print("\n" + "=" * 80)
    print("Лучшая комбинация по AUCPR")
    print("=" * 80)
    print(f"   метод aggregate: {best['aggregate'].upper()}")
    print(f"   метод update: {best['update'].upper()}")
    print(f"   test FPR: {best['test_fpr']:.4f}")
    print(f"   test AUCPR: {best['test_auprc']:.4f}")

    print("\nРейтинг по test AUCPR:")
    ranking = df_results.sort_values('test_auprc', ascending=False)
    for i, row in ranking.iterrows():
        print(f"   {i+1:2d}. {row['aggregate']:12} + {row['update']:12} : FPR={row['test_fpr']:.4f}, AUCPR={row['test_auprc']:.4f}")
    return df_results, histories




def create_model_summary_table(df_results):
    """Создаёт сводную таблицу средних значений AUCPR по методам."""
    summary = []
    agg_stats = df_results.groupby('aggregate')['test_auprc'].agg(['mean', 'std', 'max']).round(4)
    for agg in agg_stats.index:
        summary.append({
            'категория': 'aggregate',
            'метод': agg,
            'средний AUCPR': agg_stats.loc[agg, 'mean'],
            'std AUCPR': agg_stats.loc[agg, 'std'],
            'макс AUCPR': agg_stats.loc[agg, 'max'],
        })
    upd_stats = df_results.groupby('update')['test_auprc'].agg(['mean', 'std', 'max']).round(4)
    for upd in upd_stats.index:
        summary.append({
            'категория': 'update',
            'метод': upd,
            'средний AUCPR': upd_stats.loc[upd, 'mean'],
            'std AUCPR': upd_stats.loc[upd, 'std'],
            'макс AUCPR': upd_stats.loc[upd, 'max'],
        })
    return pd.DataFrame(summary)


def create_model(aggregate_method, update_method, in_channels,
                 hidden_channels=64, out_channels=64, num_layers=2):
    """Создаёт модель с заданными методами aggregate и update."""
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
    """Возвращает описание метода."""
    descriptions = {
        'aggregate': {
            'mean': 'Классическое нормализованное обновление.',
            'sym_norm': 'Симметрическая нормализация (GCN).',
            'janossy': 'Janossy pooling через LSTM.',
            'conv': 'Свёрточная агрегация с обучаемыми весами.',
            'attention': 'Attention-агрегация (GAT style).',
        },
        'update': {
            'self_loop': 'Классическое self-loop обновление.',
            'gru': 'GRU-обновление.',
        }
    }
    return descriptions.get(method_type, {}).get(method_name, 'Описание не найдено')


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


class DataLeakError(Exception):
    """Обнаружена утечка данных."""
    pass


class MetricMismatchError(Exception):
    """Сохранённые метрики не совпадают с пересчитанными."""
    pass


def diagnose_link_predictor(predictor, tolerance=1e-3):
    """
    Проверяет модель и пайплайн на утечки данных и корректность метрик.
    Бросает исключения при критических проблемах.
    Возвращает fig с диагностическими графиками и словарь с результатами.
    """
    required_attrs = ['train_data', 'val_data', 'test_data', 'train_loader', 'val_loader',
                      'test_loader', 'model', 'device', 'test_fpr', 'test_auprc']
    for attr in required_attrs:
        if not hasattr(predictor, attr):
            raise AttributeError(f"predictor должен иметь атрибут '{attr}'")

    train_data = predictor.train_data
    mp_edges = set(map(tuple, train_data.edge_index.t().tolist()))

    if (id(predictor.val_loader.data) != id(train_data) or
        id(predictor.test_loader.data) != id(train_data)):
        raise DataLeakError("val/test загрузчики используют граф, отличный от train_data")

    for name, data in [("val", predictor.val_data), ("test", predictor.test_data)]:
        pos_mask = data.edge_label.bool()
        target_pos = set(map(tuple, data.edge_label_index[:, pos_mask].t().tolist()))
        if len(target_pos & mp_edges) > 0:
            raise DataLeakError(f"Целевые рёбра {name} пересекаются с графом message passing")

    model = predictor.model
    device = predictor.device
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in predictor.test_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_label_index)
            probs = torch.sigmoid(logits).cpu()
            all_probs.append(probs)
            all_labels.append(batch.edge_label.cpu())
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    auprc_calc = average_precision_score(all_labels, all_probs)
    auc_calc = roc_auc_score(all_labels, all_probs)
    preds = (all_probs > 0.5).astype(float)
    fpr_calc = preds[all_labels == 0].mean() if (all_labels == 0).sum() > 0 else 0.0

    if (abs(auprc_calc - predictor.test_auprc) > tolerance or
        abs(fpr_calc - predictor.test_fpr) > tolerance):
        raise MetricMismatchError(
            f"Метрики расходятся: AUCPR {auprc_calc:.4f} vs {predictor.test_auprc:.4f}, "
            f"FPR {fpr_calc:.4f} vs {predictor.test_fpr:.4f}"
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.hist(all_probs[all_labels == 1], bins=40, alpha=0.6, label='Positive', color='blue')
    ax1.hist(all_probs[all_labels == 0], bins=40, alpha=0.6, label='Negative', color='red')
    ax1.set_xlabel('Predicted probability')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Probability distribution (test)')
    ax1.legend()
    ax1.grid(alpha=0.3)

    PrecisionRecallDisplay.from_predictions(all_labels, all_probs, ax=ax2, name='Model')
    ax2.set_title('Precision-Recall curve (test)')
    ax2.grid(alpha=0.3)

    fig.tight_layout()

    results = {
        'auprc': auprc_calc,
        'auc_roc': auc_calc,
        'fpr': fpr_calc,
        'mean_prob_positive': float(all_probs[all_labels == 1].mean()),
        'mean_prob_negative': float(all_probs[all_labels == 0].mean()),
        'leak_detected': False
    }
    return fig, results
