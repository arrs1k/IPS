import argparse
import os
from collections import defaultdict

import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import degree, remove_self_loops

from chartalist.common.bitcoin_graph_maker import BitcoinGraphMaker


def load_raw_data(in_path, out_path):
    df_in = pd.DataFrame({'trans': pd.read_csv(in_path, header=None, names=['trans'])['trans'].astype(str).str.strip()})
    df_out = pd.DataFrame({'trans': pd.read_csv(out_path, header=None, names=['trans'])['trans'].astype(str).str.strip()})
    df_in = df_in[df_in['trans'] != '']
    df_out = df_out[df_out['trans'] != '']
    return df_in, df_out


def collapse_transaction_vertices(G):
    add = defaultdict(float)
    rem = []
    for tx, attr in G.nodes(data=True):
        if attr.get('type') != 'trans':
            continue
        ins = [u for u, _ in G.in_edges(tx)]
        outs = [v for _, v in G.out_edges(tx)]
        for u in ins:
            wu = G[u][tx].get('value', 1.0) or 1.0
            for v in outs:
                wv = G[tx][v].get('value', 1.0) or 1.0
                add[(u, v)] += min(wu, wv)
        rem.append(tx)
    for (u, v), w in add.items():
        if G.has_edge(u, v):
            G[u][v]['value'] = float((G[u][v].get('value', 0.0) or 0.0) + w)
        else:
            G.add_edge(u, v, value=float(w))
    G.remove_nodes_from(rem)
    return G


def build_networkx_graph(df_in, df_out, collapse=True):
    G = BitcoinGraphMaker().make_graph(df_in, df_out)
    return collapse_transaction_vertices(G) if collapse else G


def convert_to_pyg_data(G):
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    e = list(G.edges(data=True))
    edge_index = torch.tensor([[idx[u] for u, v, _ in e], [idx[v] for u, v, _ in e]], dtype=torch.long)
    edge_attr = torch.tensor([float(d.get('value', 0.0) or 0.0) for _, _, d in e], dtype=torch.float32).unsqueeze(1)
    return Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(nodes))


def add_heuristic_node_features(data):
    n, ei, ew = data.num_nodes, data.edge_index, data.edge_attr.view(-1)
    src, dst = ei[0], ei[1]
    in_deg = degree(dst, n, dtype=torch.float32)
    out_deg = degree(src, n, dtype=torch.float32)
    in_sum = torch.zeros(n, dtype=torch.float32).scatter_add_(0, dst, ew)
    out_sum = torch.zeros(n, dtype=torch.float32).scatter_add_(0, src, ew)
    in_mean = in_sum / in_deg.clamp_min(1)
    out_mean = out_sum / out_deg.clamp_min(1)

    g = Data(edge_index=ei, num_nodes=n)

    def mean_score_per_node(score_fn):
        s = score_fn(g, edge_label_index=ei).float()
        a = torch.zeros(n, dtype=torch.float32).scatter_add_(0, src, s)
        c = torch.zeros(n, dtype=torch.float32).scatter_add_(0, src, torch.ones_like(s))
        return (a / c.clamp_min(1)).unsqueeze(1)

    x = torch.stack([
        torch.log1p(in_deg),
        torch.log1p(out_deg),
        torch.log1p(in_sum),
        torch.log1p(out_sum),
        torch.log1p(in_mean),
        torch.log1p(out_mean),
    ], dim=1)
    # data.x = torch.cat([x, mean_score_per_node(jaccard_scores), mean_score_per_node(adamic_adar_scores)], dim=1)
    data.x = torch.cat([x], dim=1)
    return data


def main(args):
    print('load_raw_data...')
    df_in, df_out = load_raw_data(args.in_file, args.out_file)
    print('raw loaded', len(df_in), len(df_out))

    print('build_networkx_graph...')
    G = build_networkx_graph(df_in, df_out, collapse=True)
    print('graph built', G.number_of_nodes(), G.number_of_edges())

    print('convert_to_pyg_data...')
    data = convert_to_pyg_data(G)
    print('pyg converted', data.num_nodes, data.edge_index.size(1))

    print('remove_self_loops...')
    data.edge_index, data.edge_attr = remove_self_loops(data.edge_index, data.edge_attr)
    print('self loops removed', data.edge_index.size(1))

    print('RandomLinkSplit...')
    train_data, val_data, test_data = RandomLinkSplit(
        is_undirected=False,
        num_val=args.num_val,
        num_test=args.num_test,
        neg_sampling_ratio=args.neg_sampling_ratio,
    )(data)
    print('split done')

    print('add_heuristic_node_features...')
    train_data = add_heuristic_node_features(train_data)
    print('features done')

    val_data.x = train_data.x
    test_data.x = train_data.x

    path = os.path.join(args.save_dir, 'bitcoin_link_pred_data.pt')
    print('saving...')
    torch.save({
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data,
        'num_neighbors': args.num_neighbors,
        'batch_size': args.batch_size,
    }, path)
    print(f'Данные сохранены в {path}')


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Подготовка данных Bitcoin для link prediction")
    p.add_argument("in_file")
    p.add_argument("out_file")
    p.add_argument("--save-dir", default=".")
    p.add_argument("--num-val", type=float, default=0.1)
    p.add_argument("--num-test", type=float, default=0.1)
    p.add_argument("--neg-sampling-ratio", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--num-neighbors", type=int, nargs='+', default=[10, 5])
    main(p.parse_args())