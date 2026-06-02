import pandas as pd
from LinkPredictor import *
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import random
from sklearn.metrics import average_precision_score, roc_auc_score


def set_seed(seed=42):
    """Фиксирует random state для воспроизводимости результатов."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed установлен на {seed}")


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
                             results_file='comparison_results.csv',
                             epochs=50, lr=0.01, weight_decay=5e-4, patience=None,
                             hidden_dim=64, out_dim=64, dropout=0.3, num_layers=2,
                             attention_heads=4, seed=42, **extra_model_params):
    set_seed(seed)
    
    aggregate_classes = [MeanMessage, SymNormMessage, ConvMessage, AttentionMessage]
    update_classes = [SelfLoopUpdate, GRUUpdate]
    results = []
    histories = {}
    figs = {}
    diag_results = []
    in_dim = train_data.x.shape[1]

    if patience is None:
        patience = max(epochs // 3, 1)

    for AggClass in aggregate_classes:
        for UpdClass in update_classes:
            combo_name = f"{AggClass.__name__}_{UpdClass.__name__}"
            print(f"\nТестирование: agg={AggClass.__name__}, update={UpdClass.__name__}")

            if AggClass == AttentionMessage:
                msg_list = [
                    AttentionMessage(in_dim if i == 0 else hidden_dim,
                                     hidden_dim, heads=attention_heads)
                    for i in range(num_layers)
                ]
            else:
                msg_list = [
                    AggClass(in_dim if i == 0 else hidden_dim, hidden_dim)
                    for i in range(num_layers)
                ]

            if UpdClass == SelfLoopUpdate:
                upd_list = [
                    SelfLoopUpdate(in_dim if i == 0 else hidden_dim, hidden_dim)
                    for i in range(num_layers)
                ]
            else:
                upd_list = [
                    GRUUpdate(in_dim if i == 0 else hidden_dim, hidden_dim)
                    for i in range(num_layers)
                ]

            aggr_str = 'mean' if AggClass == MeanMessage else 'add'
            model_params = {
                'in_channels': in_dim,
                'hidden_channels': hidden_dim,
                'out_channels': out_dim,
                'num_layers': num_layers,
                'message_fn': msg_list,
                'aggr': aggr_str,
                'update_fn': upd_list,
                'dropout': dropout,
            }
            model_params.update(extra_model_params)

            try:
                predictor = LinkPredictor(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    model_class=LinkPredictionMessagePassingModel,
                    model_params=model_params,
                    epochs=epochs,
                    patience=patience,
                    lr=lr,
                    weight_decay=weight_decay
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
    cbar = plt.colorbar(im, ax=2)
    cbar.set_label('Test AUCPR')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_all_learning_curves(histories, save_path=None):
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
                                    epochs=30, lr=0.01, weight_decay=5e-4, patience=None,
                                    save_dir='training_plots',
                                    hidden_dim=64, out_dim=64, dropout=0.3, num_layers=2,
                                    attention_heads=4, seed=42, **extra_model_params):
    set_seed(seed)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"{save_dir}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nРезультаты будут сохранены в: {save_dir}/")

    df_results, histories, figs, diag_results = compare_all_combinations(
        train_loader, val_loader, test_loader,
        train_data, val_data, test_data,
        results_file=f'{save_dir}/results.csv',
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        dropout=dropout,
        num_layers=num_layers,
        attention_heads=attention_heads,
        seed=seed,
        **extra_model_params
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
    pass


class MetricMismatchError(Exception):
    pass


def diagnose_link_predictor(predictor, tolerance=1e-3):
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


def optimize_single_pair(train_loader, val_loader, test_loader,
                         train_data, val_data, test_data,
                         aggregate_class, update_class,
                         n_trials=50, timeout=None,
                         save_dir='optuna_single',
                         final_epochs=200,
                         seed=42):
    try:
        import optuna
        from optuna.trial import Trial
    except ImportError:
        raise ImportError("Установите optuna: pip install optuna")
    
    set_seed(seed)
    
    combo_name = f"{aggregate_class.__name__}_{update_class.__name__}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"{save_dir}_{combo_name}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print(f"ОПТИМИЗАЦИЯ ДЛЯ: {combo_name}")
    print("=" * 80)
    
    in_dim = train_data.x.shape[1]
    
    def objective(trial: Trial):
        trial_seed = seed + trial.number
        set_seed(trial_seed)
        
        params = {
            'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 96, 128]),
            'out_dim': trial.suggest_categorical('out_dim', [32, 48, 64, 96, 128]),
            'num_layers': trial.suggest_int('num_layers', 1, 6),
            'dropout': trial.suggest_float('dropout', 0.1, 0.6, step=0.05),
            'lr': trial.suggest_float('lr', 1e-4, 5e-3, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
            'patience': trial.suggest_int('patience', 20, 80, step=10),
        }
        
        if aggregate_class == AttentionMessage:
            params['attention_heads'] = trial.suggest_categorical('attention_heads', [2, 4, 8])
        
        if aggregate_class == AttentionMessage:
            msg_list = [
                AttentionMessage(in_dim if i == 0 else params['hidden_dim'],
                                 params['hidden_dim'], heads=params['attention_heads'])
                for i in range(params['num_layers'])
            ]
        else:
            msg_list = [
                aggregate_class(in_dim if i == 0 else params['hidden_dim'], params['hidden_dim'])
                for i in range(params['num_layers'])
            ]
        
        if update_class == SelfLoopUpdate:
            upd_list = [
                SelfLoopUpdate(in_dim if i == 0 else params['hidden_dim'], params['hidden_dim'])
                for i in range(params['num_layers'])
            ]
        else:
            upd_list = [
                GRUUpdate(in_dim if i == 0 else params['hidden_dim'], params['hidden_dim'])
                for i in range(params['num_layers'])
            ]
        
        aggr_str = 'mean' if aggregate_class == MeanMessage else 'add'
        
        model_params = {
            'in_channels': in_dim,
            'hidden_channels': params['hidden_dim'],
            'out_channels': params['out_dim'],
            'num_layers': params['num_layers'],
            'message_fn': msg_list,
            'aggr': aggr_str,
            'update_fn': upd_list,
            'dropout': params['dropout'],
        }
        
        try:
            predictor = LinkPredictor(
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                model_class=LinkPredictionMessagePassingModel,
                model_params=model_params,
                epochs=100,
                patience=params['patience'],
                lr=params['lr'],
                weight_decay=params['weight_decay']
            )
            
            predictor.train_data = train_data
            predictor.val_data = val_data
            predictor.test_data = test_data
            
            model, test_fpr, test_auprc = predictor.run()
            
            best_val_auprc = max(predictor.history['val_auprc']) if predictor.history['val_auprc'] else 0.0
            best_val_fpr = min(predictor.history['val_fpr']) if predictor.history['val_fpr'] else 1.0
            
            trial.set_user_attr('test_auprc', test_auprc)
            trial.set_user_attr('test_fpr', test_fpr)
            trial.set_user_attr('best_val_auprc', best_val_auprc)
            trial.set_user_attr('best_val_fpr', best_val_fpr)
            
            score = best_val_auprc
            
            return score
            
        except Exception as e:
            print(f"  Ошибка в trial: {e}")
            return -1.0
    
    study_name = f"{combo_name}_{timestamp}"
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        storage=f'sqlite:///{save_dir}/{combo_name}.db',
        load_if_exists=True,
        sampler=optuna.samplers.RandomSampler(seed=seed)
    )
    
    print(f"\nЗапуск оптимизации...")
    print(f"Количество попыток: {n_trials}")
    
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    
    best_params = study.best_params
    best_value = study.best_value
    
    print(f"\nЛучшие гиперпараметры для {combo_name}:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"Лучшее значение метрики: {best_value:.4f}")
    
    trials_df = study.trials_dataframe()
    trials_df.to_csv(f'{save_dir}/{combo_name}_trials.csv', index=False)
    
    print("\n" + "=" * 80)
    print("ФИНАЛЬНОЕ ОБУЧЕНИЕ С ЛУЧШИМИ ПАРАМЕТРАМИ")
    print("=" * 80)
    
    set_seed(seed)
    
    if aggregate_class == AttentionMessage:
        attention_heads = best_params.get('attention_heads', 4)
        msg_list = [
            AttentionMessage(in_dim if i == 0 else best_params['hidden_dim'],
                             best_params['hidden_dim'], heads=attention_heads)
            for i in range(best_params['num_layers'])
        ]
    else:
        msg_list = [
            aggregate_class(in_dim if i == 0 else best_params['hidden_dim'], best_params['hidden_dim'])
            for i in range(best_params['num_layers'])
        ]
    
    if update_class == SelfLoopUpdate:
        upd_list = [
            SelfLoopUpdate(in_dim if i == 0 else best_params['hidden_dim'], best_params['hidden_dim'])
            for i in range(best_params['num_layers'])
        ]
    else:
        upd_list = [
            GRUUpdate(in_dim if i == 0 else best_params['hidden_dim'], best_params['hidden_dim'])
            for i in range(best_params['num_layers'])
        ]
    
    aggr_str = 'mean' if aggregate_class == MeanMessage else 'add'
    
    model_params = {
        'in_channels': in_dim,
        'hidden_channels': best_params['hidden_dim'],
        'out_channels': best_params['out_dim'],
        'num_layers': best_params['num_layers'],
        'message_fn': msg_list,
        'aggr': aggr_str,
        'update_fn': upd_list,
        'dropout': best_params['dropout'],
    }
    
    predictor = LinkPredictor(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model_class=LinkPredictionMessagePassingModel,
        model_params=model_params,
        epochs=final_epochs,
        patience=best_params['patience'],
        lr=best_params['lr'],
        weight_decay=best_params['weight_decay']
    )
    
    predictor.train_data = train_data
    predictor.val_data = val_data
    predictor.test_data = test_data
    
    model, test_fpr, test_auprc = predictor.run()
    
    results = {
        'aggregate': aggregate_class.__name__,
        'update': update_class.__name__,
        'best_params': best_params,
        'optuna_best_value': best_value,
        'test_fpr': test_fpr,
        'test_auprc': test_auprc,
        'final_epochs': final_epochs
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv(f'{save_dir}/final_results.csv', index=False)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_params': best_params,
        'test_fpr': test_fpr,
        'test_auprc': test_auprc,
    }, f'{save_dir}/{combo_name}_model.pt')
    
    print("\n" + "=" * 80)
    print("ГОТОВО!")
    print("=" * 80)
    print(f"\nРезультаты для {combo_name}:")
    print(f"  Test FPR: {test_fpr:.4f}")
    print(f"  Test AUCPR: {test_auprc:.4f}")
    print(f"\nЛучшие гиперпараметры:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"\nРезультаты сохранены в: {save_dir}/")
    
    return best_params, test_fpr, test_auprc


def run_with_optuna(train_loader, val_loader, test_loader,
                    train_data, val_data, test_data,
                    n_trials=50, timeout=None,
                    save_dir='optuna_results',
                    aggregate_class=None, update_class=None,
                    study_name=None,
                    final_epochs=200,
                    seed=42):
    try:
        import optuna
        from optuna.trial import Trial
    except ImportError:
        raise ImportError("Установите optuna: pip install optuna")
    
    set_seed(seed)
    
    aggregate_classes = [MeanMessage, SymNormMessage, ConvMessage, AttentionMessage]
    update_classes = [SelfLoopUpdate, GRUUpdate]
    
    if aggregate_class is not None:
        aggregate_classes = [aggregate_class]
    if update_class is not None:
        update_classes = [update_class]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"{save_dir}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    all_best_params = {}
    
    for AggClass in aggregate_classes:
        for UpdClass in update_classes:
            combo_name = f"{AggClass.__name__}_{UpdClass.__name__}"
            print("\n" + "=" * 80)
            print(f"OPTUNA ОПТИМИЗАЦИЯ ДЛЯ: {combo_name}")
            print("=" * 80)
            
            in_dim = train_data.x.shape[1]
            
            def objective(trial: Trial):
                trial_seed = seed + trial.number
                set_seed(trial_seed)
                
                params = {
                    'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 48, 64, 96, 128]),
                    'out_dim': trial.suggest_categorical('out_dim', [32, 48, 64, 96, 128]),
                    'num_layers': trial.suggest_int('num_layers', 1, 6),
                    'dropout': trial.suggest_float('dropout', 0.1, 0.6, step=0.05),
                    'lr': trial.suggest_float('lr', 1e-4, 5e-3, log=True),
                    'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
                    'patience': trial.suggest_int('patience', 20, 80, step=10),
                }
                
                if AggClass == AttentionMessage:
                    params['attention_heads'] = trial.suggest_categorical('attention_heads', [2, 4, 8])
                
                if AggClass == AttentionMessage:
                    msg_list = [
                        AttentionMessage(in_dim if i == 0 else params['hidden_dim'],
                                         params['hidden_dim'], heads=params['attention_heads'])
                        for i in range(params['num_layers'])
                    ]
                else:
                    msg_list = [
                        AggClass(in_dim if i == 0 else params['hidden_dim'], params['hidden_dim'])
                        for i in range(params['num_layers'])
                    ]
                
                if UpdClass == SelfLoopUpdate:
                    upd_list = [
                        SelfLoopUpdate(in_dim if i == 0 else params['hidden_dim'], params['hidden_dim'])
                        for i in range(params['num_layers'])
                    ]
                else:
                    upd_list = [
                        GRUUpdate(in_dim if i == 0 else params['hidden_dim'], params['hidden_dim'])
                        for i in range(params['num_layers'])
                    ]
                
                aggr_str = 'mean' if AggClass == MeanMessage else 'add'
                
                model_params = {
                    'in_channels': in_dim,
                    'hidden_channels': params['hidden_dim'],
                    'out_channels': params['out_dim'],
                    'num_layers': params['num_layers'],
                    'message_fn': msg_list,
                    'aggr': aggr_str,
                    'update_fn': upd_list,
                    'dropout': params['dropout'],
                }
                
                try:
                    predictor = LinkPredictor(
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        model_class=LinkPredictionMessagePassingModel,
                        model_params=model_params,
                        epochs=100,
                        patience=params['patience'],
                        lr=params['lr'],
                        weight_decay=params['weight_decay']
                    )
                    
                    predictor.train_data = train_data
                    predictor.val_data = val_data
                    predictor.test_data = test_data
                    
                    model, test_fpr, test_auprc = predictor.run()
                    
                    best_val_auprc = max(predictor.history['val_auprc']) if predictor.history['val_auprc'] else 0.0
                    best_val_fpr = min(predictor.history['val_fpr']) if predictor.history['val_fpr'] else 1.0
                    
                    trial.set_user_attr('test_auprc', test_auprc)
                    trial.set_user_attr('test_fpr', test_fpr)
                    trial.set_user_attr('best_val_auprc', best_val_auprc)
                    trial.set_user_attr('best_val_fpr', best_val_fpr)
                    
                    score = best_val_auprc
                    
                    return score
                    
                except Exception as e:
                    print(f"  Ошибка в trial: {e}")
                    return -1.0
            
            study_name_current = study_name or f"{combo_name}_{timestamp}"
            study = optuna.create_study(
                direction='maximize',
                study_name=study_name_current,
                storage=f'sqlite:///{save_dir}/{combo_name}.db',
                load_if_exists=True,
                sampler=optuna.samplers.RandomSampler(seed=seed)
            )
            
            print(f"\nЗапуск оптимизации для {combo_name}")
            print(f"Количество попыток: {n_trials}")
            
            study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
            
            best_params = study.best_params
            best_value = study.best_value
            
            print(f"\nЛучшие гиперпараметры для {combo_name}:")
            for key, value in best_params.items():
                print(f"  {key}: {value}")
            print(f"Лучшее значение метрики: {best_value:.4f}")
            
            trials_df = study.trials_dataframe()
            trials_df.to_csv(f'{save_dir}/{combo_name}_trials.csv', index=False)
            
            all_best_params[combo_name] = {
                'best_params': best_params,
                'best_value': best_value,
                'aggregate_class': AggClass,
                'update_class': UpdClass,
                'in_dim': in_dim
            }
    
    print("\n" + "=" * 80)
    print("ОПТИМИЗАЦИЯ ЗАВЕРШЕНА")
    print("=" * 80)
    
    best_overall = None
    best_overall_value = -1
    
    for combo_name, result in all_best_params.items():
        if result['best_value'] > best_overall_value:
            best_overall_value = result['best_value']
            best_overall = (combo_name, result)
    
    if best_overall:
        print(f"\nЛучшая комбинация: {best_overall[0]}")
        print(f"Лучшее значение метрики: {best_overall_value:.4f}")
        print("Лучшие гиперпараметры:")
        for key, value in best_overall[1]['best_params'].items():
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)
    print("ФИНАЛЬНОЕ ОБУЧЕНИЕ С ЛУЧШИМИ ГИПЕРПАРАМЕТРАМИ")
    print("=" * 80)
    
    set_seed(seed)
    
    final_results = {}
    final_histories = {}
    
    for combo_name, result in all_best_params.items():
        AggClass = result['aggregate_class']
        UpdClass = result['update_class']
        best_params = result['best_params']
        in_dim = result['in_dim']
        
        print(f"\nФинальное обучение для: {combo_name}")
        print(f"Используемые параметры:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        
        if AggClass == AttentionMessage:
            attention_heads = best_params.get('attention_heads', 4)
            msg_list = [
                AttentionMessage(in_dim if i == 0 else best_params['hidden_dim'],
                                 best_params['hidden_dim'], heads=attention_heads)
                for i in range(best_params['num_layers'])
            ]
        else:
            msg_list = [
                AggClass(in_dim if i == 0 else best_params['hidden_dim'], best_params['hidden_dim'])
                for i in range(best_params['num_layers'])
            ]
        
        if UpdClass == SelfLoopUpdate:
            upd_list = [
                SelfLoopUpdate(in_dim if i == 0 else best_params['hidden_dim'], best_params['hidden_dim'])
                for i in range(best_params['num_layers'])
            ]
        else:
            upd_list = [
                GRUUpdate(in_dim if i == 0 else best_params['hidden_dim'], best_params['hidden_dim'])
                for i in range(best_params['num_layers'])
            ]
        
        aggr_str = 'mean' if AggClass == MeanMessage else 'add'
        
        try:
            df_results, histories, figs, diag_results = compare_all_combinations(
                train_loader, val_loader, test_loader,
                train_data, val_data, test_data,
                results_file=f'{save_dir}/final_{combo_name}_results.csv',
                epochs=final_epochs,
                lr=best_params['lr'],
                weight_decay=best_params['weight_decay'],
                patience=best_params['patience'],
                hidden_dim=best_params['hidden_dim'],
                out_dim=best_params['out_dim'],
                dropout=best_params['dropout'],
                num_layers=best_params['num_layers'],
                attention_heads=best_params.get('attention_heads', 4),
                seed=seed
            )
            
            final_results[combo_name] = df_results
            final_histories[combo_name] = histories
            
            if len(df_results) > 0:
                best_row = df_results.iloc[0]
                print(f"\nРезультаты финального обучения для {combo_name}:")
                print(f"  Test FPR: {best_row['test_fpr']:.4f}")
                print(f"  Test AUCPR: {best_row['test_auprc']:.4f}")
            
        except Exception as e:
            print(f"  Ошибка при финальном обучении: {e}")
            import traceback
            traceback.print_exc()
    
    summary = []
    for combo_name, result in all_best_params.items():
        row = {
            'комбинация': combo_name,
            'optuna_метрика': result['best_value'],
        }
        for key, value in result['best_params'].items():
            row[f'optuna_{key}'] = value
        summary.append(row)
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f'{save_dir}/best_params_summary.csv', index=False)
    
    print("\n" + "=" * 80)
    print("ГОТОВО!")
    print("=" * 80)
    print(f"\nРезультаты сохранены в: {save_dir}/")
    print("  - best_params_summary.csv - лучшие параметры для каждой комбинации")
    print("  - final_*_results.csv - результаты финального обучения")
    print("  - *_trials.csv - все попытки оптимизации")
    
    return all_best_params, final_results, final_histories


def optimize_best_combination(train_loader, val_loader, test_loader,
                               train_data, val_data, test_data,
                               n_trials=100, timeout=None,
                               save_dir='best_optimization',
                               seed=42):
    set_seed(seed)
    
    print("=" * 80)
    print("ШАГ 1: ОПРЕДЕЛЕНИЕ ЛУЧШЕЙ КОМБИНАЦИИ МЕТОДОВ")
    print("=" * 80)
    
    df_results, histories, figs, diag_results = compare_all_combinations(
        train_loader, val_loader, test_loader,
        train_data, val_data, test_data,
        epochs=50,
        hidden_dim=64,
        out_dim=64,
        dropout=0.3,
        num_layers=2,
        seed=seed
    )
    
    best_idx = df_results['test_auprc'].idxmax()
    best_row = df_results.loc[best_idx]
    best_aggregate_name = best_row['aggregate']
    best_update_name = best_row['update']
    
    agg_map = {
        'MeanMessage': MeanMessage,
        'SymNormMessage': SymNormMessage,
        'ConvMessage': ConvMessage,
        'AttentionMessage': AttentionMessage,
    }
    upd_map = {
        'SelfLoopUpdate': SelfLoopUpdate,
        'GRUUpdate': GRUUpdate,
    }
    
    best_agg_class = agg_map.get(best_aggregate_name)
    best_upd_class = upd_map.get(best_update_name)
    
    print(f"\nЛучшая комбинация: {best_aggregate_name} + {best_update_name}")
    print(f"Test AUCPR: {best_row['test_auprc']:.4f}")
    
    print("\n" + "=" * 80)
    print("ШАГ 2: ОПТИМИЗАЦИЯ ГИПЕРПАРАМЕТРОВ ДЛЯ ЛУЧШЕЙ КОМБИНАЦИИ")
    print("=" * 80)
    
    results = run_with_optuna(
        train_loader, val_loader, test_loader,
        train_data, val_data, test_data,
        n_trials=n_trials,
        timeout=timeout,
        save_dir=save_dir,
        aggregate_class=best_agg_class,
        update_class=best_upd_class,
        study_name=f"best_{best_aggregate_name}_{best_update_name}",
        seed=seed
    )
    
    return results


def run_optuna_for_all_combinations(train_loader, val_loader, test_loader,
                                     train_data, val_data, test_data,
                                     n_trials_per_combo=30, timeout=None,
                                     save_dir='optuna_all_combinations',
                                     seed=42):
    set_seed(seed)
    
    aggregate_classes = [MeanMessage, SymNormMessage, ConvMessage, AttentionMessage]
    update_classes = [SelfLoopUpdate, GRUUpdate]
    
    all_results = {}
    
    for AggClass in aggregate_classes:
        for UpdClass in update_classes:
            results = run_with_optuna(
                train_loader, val_loader, test_loader,
                train_data, val_data, test_data,
                n_trials=n_trials_per_combo,
                timeout=timeout,
                save_dir=save_dir,
                aggregate_class=AggClass,
                update_class=UpdClass,
                study_name=f"{AggClass.__name__}_{UpdClass.__name__}",
                seed=seed
            )
            all_results.update(results)
    
    print("\n" + "=" * 80)
    print("СРАВНЕНИЕ ОПТИМИЗИРОВАННЫХ КОМБИНАЦИЙ")
    print("=" * 80)
    
    comparison = []
    for combo_name, result in all_results.items():
        if isinstance(result, dict) and 'best_value' in result:
            comparison.append({
                'комбинация': combo_name,
                'лучшая_метрика': result['best_value'],
                'hidden_dim': result['best_params'].get('hidden_dim'),
                'out_dim': result['best_params'].get('out_dim'),
                'num_layers': result['best_params'].get('num_layers'),
                'dropout': result['best_params'].get('dropout'),
                'lr': result['best_params'].get('lr'),
                'weight_decay': result['best_params'].get('weight_decay'),
            })
    
    if comparison:
        comparison_df = pd.DataFrame(comparison)
        comparison_df = comparison_df.sort_values('лучшая_метрика', ascending=False)
        comparison_df.to_csv(f'{save_dir}/comparison.csv', index=False)
        
        print("\nРейтинг оптимизированных комбинаций:")
        for i, row in comparison_df.iterrows():
            print(f"  {i+1}. {row['комбинация']}: {row['лучшая_метрика']:.4f}")
    
    return all_results