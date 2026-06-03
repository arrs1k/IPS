import pandas as pd
from LinkPredictor import *
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import random
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.utils import softmax


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class MLPDecoder(nn.Module):
    def __init__(self, emb_dim, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    def forward(self, z, edge_label_index):
        row, col = edge_label_index
        zi, zj = z[row], z[col]
        h = torch.cat([zi, zj, zi * zj, torch.abs(zi - zj)], dim=-1)
        return self.mlp(h).squeeze(-1)


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
    def __init__(self, in_dim, out_dim, num_edges):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.edge_weights = nn.Embedding(num_edges, 1)
        self.is_conv = True
        nn.init.constant_(self.edge_weights.weight, 1.0)
    def forward(self, x_j, x_i=None, edge_attr=None, index=None):
        edge_id = edge_attr.squeeze(-1).long()
        w = self.edge_weights(edge_id)
        return self.lin(x_j) * w


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


def _with_mlp_decoder(model_params, out_dim, hidden_dim=128, dropout=0.2):
    model_params = dict(model_params)
    model_params['decoder_fn'] = MLPDecoder(emb_dim=out_dim, hidden_dim=hidden_dim, dropout=dropout)
    return model_params


def compare_all_combinations(train_loader, val_loader, test_loader,
                             train_data, val_data, test_data,
                             results_file='comparison_results.csv',
                             epochs=50, lr=0.01, weight_decay=5e-4, patience=None,
                             hidden_dim=64, out_dim=64, dropout=0.3, num_layers=2,
                             attention_heads=4, seed=42, **extra_model_params):
    set_seed(seed)
    
    aggregate_classes = [MeanMessage, SymNormMessage, ConvMessage, AttentionMessage]
    update_methods = ['self_loops', 'gru']
    results = []
    histories = {}
    figs = {}
    diag_results = []
    in_dim = train_data.x.shape[1]
    num_edges = train_data.edge_index.size(1)
    
    if patience is None:
        patience = max(epochs // 3, 1)
    
    for AggClass in aggregate_classes:
        for upd_method in update_methods:
            combo_name = f"{AggClass.__name__}_{upd_method}"
            print(f"\nТестирование: agg={AggClass.__name__}, update={upd_method}")
            
            if AggClass == ConvMessage:
                msg_list = [
                    ConvMessage(in_dim if i == 0 else hidden_dim, hidden_dim, num_edges)
                    for i in range(num_layers)
                ]
            elif AggClass == AttentionMessage:
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
            
            aggr_str = 'mean' if AggClass == MeanMessage else 'add'
            model_params = {
                'in_channels': in_dim,
                'hidden_channels': hidden_dim,
                'out_channels': out_dim,
                'num_layers': num_layers,
                'message_fn': msg_list,
                'aggr': aggr_str,
                'update': upd_method,
                'dropout': dropout,
            }
            model_params.update(extra_model_params)
            model_params = _with_mlp_decoder(model_params, out_dim)
            
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
                    'update': upd_method,
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
                    'update': upd_method,
                    'test_fpr': 1.0,
                    'test_auprc': 0.0,
                    'error': str(e)
                })
                histories[combo_name] = None
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(results_file, index=False)
    print(f"\nРезультаты сохранены в: {results_file}")
    return df_results, histories, figs, diag_results


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
            edge_id = getattr(batch, 'edge_id', None)
            logits = model(batch.x, batch.edge_index, batch.edge_label_index, edge_id=edge_id)
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


def run_with_optuna(train_loader, val_loader, test_loader,
                    train_data, val_data, test_data,
                    n_trials=50, timeout=None,
                    save_dir='optuna_results',
                    aggregate_class=None, update_method=None,
                    study_name=None,
                    final_epochs=200,
                    seed=42):
    set_seed(seed)
    
    try:
        import optuna
        from optuna.trial import Trial
    except ImportError:
        raise ImportError("Установите optuna: pip install optuna")
    
    aggregate_classes = [MeanMessage, SymNormMessage, ConvMessage, AttentionMessage]
    update_methods = ['self_loops', 'gru']
    
    if aggregate_class is not None:
        aggregate_classes = [aggregate_class]
    if update_method is not None:
        update_methods = [update_method]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"{save_dir}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    all_best_params = {}
    
    for AggClass in aggregate_classes:
        for upd_method in update_methods:
            combo_name = f"{AggClass.__name__}_{upd_method}"
            print("\n" + "=" * 80)
            print(f"OPTUNA ОПТИМИЗАЦИЯ ДЛЯ: {combo_name}")
            print("=" * 80)
            
            in_dim = train_data.x.shape[1]
            num_edges = train_data.edge_index.size(1)
            
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
                
                if AggClass == ConvMessage:
                    msg_list = [
                        ConvMessage(in_dim if i == 0 else params['hidden_dim'],
                                    params['hidden_dim'], num_edges)
                        for i in range(params['num_layers'])
                    ]
                elif AggClass == AttentionMessage:
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
                
                aggr_str = 'mean' if AggClass == MeanMessage else 'add'
                model_params = {
                    'in_channels': in_dim,
                    'hidden_channels': params['hidden_dim'],
                    'out_channels': params['out_dim'],
                    'num_layers': params['num_layers'],
                    'message_fn': msg_list,
                    'aggr': aggr_str,
                    'update': upd_method,
                    'dropout': params['dropout'],
                }
                model_params = _with_mlp_decoder(model_params, params['out_dim'])
                
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
            
            study = optuna.create_study(
                direction='maximize',
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
                'update_method': upd_method,
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
        upd_method = result['update_method']
        best_params = result['best_params']
        in_dim = result['in_dim']
        num_edges_final = train_data.edge_index.size(1)
        
        print(f"\nФинальное обучение для: {combo_name}")
        print(f"Используемые параметры:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        
        if AggClass == ConvMessage:
            msg_list = [
                ConvMessage(in_dim if i == 0 else best_params['hidden_dim'],
                            best_params['hidden_dim'], num_edges_final)
                for i in range(best_params['num_layers'])
            ]
        elif AggClass == AttentionMessage:
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
            'Комбинация': combo_name,
            'Test AUCPR': result['best_value'],
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

def run_optuna_for_all_combinations(train_loader, val_loader, test_loader,
                                     train_data, val_data, test_data,
                                     n_trials_per_combo=30, timeout=None,
                                     save_dir='optuna_all_combinations',
                                     seed=42):
    set_seed(seed)
    
    aggregate_classes = [MeanMessage, SymNormMessage, ConvMessage, AttentionMessage]
    update_methods = ['self_loops', 'gru']
    all_results = {}
    
    for AggClass in aggregate_classes:
        for upd_method in update_methods:
            results = run_with_optuna(
                train_loader, val_loader, test_loader,
                train_data, val_data, test_data,
                n_trials=n_trials_per_combo,
                timeout=timeout,
                save_dir=save_dir,
                aggregate_class=AggClass,
                update_method=upd_method,
                study_name=f"{AggClass.__name__}_{upd_method}",
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
                'Комбинация': combo_name,
                'Test AUCPR': result['best_value'],
                'hidden_dim': result['best_params'].get('hidden_dim'),
                'out_dim': result['best_params'].get('out_dim'),
                'num_layers': result['best_params'].get('num_layers'),
                'dropout': result['best_params'].get('dropout'),
                'lr': result['best_params'].get('lr'),
                'weight_decay': result['best_params'].get('weight_decay'),
            })
    
    if comparison:
        comparison_df = pd.DataFrame(comparison)
        comparison_df = comparison_df.sort_values('Test AUCPR', ascending=False)
        comparison_df.to_csv(f'{save_dir}/comparison.csv', index=False)
        print("\nРейтинг оптимизированных комбинаций:")
        for i, row in comparison_df.iterrows():
            print(f"  {i+1}. {row['комбинация']}: {row['Test AUCPR']:.4f}")
    
    return all_results