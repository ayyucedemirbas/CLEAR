import pandas as pd
import pickle
import numpy as np
import random
import networkx as nx
from sklearn.model_selection import KFold
import os
from tqdm.auto import tqdm
from collections import defaultdict
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv
from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import LinkNeighborLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
from torch.utils.data import Subset
import time


def impute_missing(feats, missing_idxs, sample_k):
    all_idxs = [i for i in range(feats.size(0)) if i not in missing_idxs]
    for _, mi in missing_idxs:
        sampled = random.sample(all_idxs, sample_k)
        feats[mi] = feats[sampled].mean(dim=0)
    return feats


def load_similarity_edges(file_path, src2idx, dst2idx):
    df = pd.read_csv(file_path)
    df = df.drop(columns=['Unnamed: 0'])
    src = [src2idx[x] for x in df.iloc[:, 0]]
    dst = [dst2idx[x] for x in df.iloc[:, 1]]
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)
    return edge_index, edge_weight


def load_bipartite_edges(file_path, src2idx, dst2idx):
    df = pd.read_csv(file_path)
    try:
        df = df.drop(columns=['Unnamed: 0'])
    except KeyError:
        pass
    src = [src2idx[x] for x in df.iloc[:, 0]]
    dst = [dst2idx[x] for x in df.iloc[:, 1]]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)
    return edge_index, edge_weight


def precompute_hop_neighbors(G_nx, data, k_hop):
    hop_neighbors = {}
    rel_iter = tqdm(data.edge_types, desc="Relations", unit="rel")
    for (u_t, rel, v_t) in rel_iter:
        N_u = data[u_t].num_nodes
        uv_neighbors = {}
        node_iter = tqdm(range(N_u), desc=f"{u_t}→{v_t} nodes",
                         unit="u", leave=False)
        prefix = f"{v_t}_"
        for u in node_iter:
            u_node = f"{u_t}_{u}"
            reachable = nx.single_source_shortest_path_length(
                G_nx, u_node, cutoff=k_hop).keys()
            vt_set = {int(n.split("_", 1)[1])
                      for n in reachable if n.startswith(prefix)}
            uv_neighbors[u] = vt_set
        hop_neighbors[(u_t, v_t)] = uv_neighbors
    return hop_neighbors


def sample_neg_uniform(data, pos_sets, u_t, rel, v_t, num_samples):
    N_u, N_v = data[u_t].num_nodes, data[v_t].num_nodes
    pos = pos_sets[(u_t, rel, v_t)]
    neg = []
    while len(neg) < num_samples:
        u = random.randrange(N_u)
        v = random.randrange(N_v)
        if (u, v) not in pos:
            neg.append((u, v))
    return torch.tensor(neg).t()


def sample_neg_3hop(data, pos_sets, hop_neighbors, u_t, rel, v_t, num_samples):
    N_u, N_v = data[u_t].num_nodes, data[v_t].num_nodes
    pos   = pos_sets[(u_t, rel, v_t)]
    hop_n = hop_neighbors[(u_t, v_t)]
    neg = []
    while len(neg) < num_samples:
        u = random.randrange(N_u)
        v = random.randrange(N_v)
        if (u, v) in pos:
            continue
        if v in hop_n[u]:
            continue
        neg.append((u, v))
    return torch.tensor(neg).t()


def sample_neg_prob_deg(data, pos_sets, hop_neighbors, endpoint_counts,
                        u_t, rel, v_t, num_samples, max_attempts):
    N_u, N_v     = data[u_t].num_nodes, data[v_t].num_nodes
    P_src, P_dst = endpoint_counts[(u_t, rel, v_t)]
    pos  = pos_sets[(u_t, rel, v_t)]
    hop_n = hop_neighbors[(u_t, v_t)]
    neg = []
    for _ in range(num_samples):
        for _try in range(max_attempts):
            u = np.random.choice(N_u, p=P_src)
            v = np.random.choice(N_v, p=P_dst)
            if (u, v) in pos or v in hop_n[u]:
                continue
            neg.append((u, v))
            break
        else:
            while True:
                u = random.randrange(N_u)
                v = random.randrange(N_v)
                if (u, v) in pos:
                    continue
                if v in hop_n[u]:
                    continue
                neg.append((u, v))
                break
    return torch.tensor(neg).t()


def sample_negatives(data, pos_sets, hop_neighbors, endpoint_counts,
                     u_t, rel, v_t, num_samples, strategy):
    if strategy == "random":
        return sample_neg_uniform(data, pos_sets, u_t, rel, v_t, num_samples)
    elif strategy == "3-hop":
        return sample_neg_3hop(data, pos_sets, hop_neighbors, u_t, rel, v_t, num_samples)
    elif strategy == "pdf":
        return sample_neg_prob_deg(
            data, pos_sets, hop_neighbors, endpoint_counts,
            u_t, rel, v_t, num_samples, max_attempts=20)
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")


def build_pos_sets(data):
    pos_sets = {}
    for (u_t, rel, v_t) in data.edge_types:
        edge_index = data[(u_t, rel, v_t)].edge_index.t().tolist()
        pos_sets[(u_t, rel, v_t)] = set(tuple(e) for e in edge_index)
    return pos_sets


def k_fold_link_splits(data, pos_sets, hop_neighbors, endpoint_counts,
                       G_nx, u_t, rel, v_t, k, random_state, strategy):
    pos_idx = data[(u_t, rel, v_t)].edge_index.t().tolist()
    pos_idx = np.array(pos_idx)
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    for train_ix, test_ix in kf.split(pos_idx):
        train_pos = torch.tensor(pos_idx[train_ix]).t()
        test_pos  = torch.tensor(pos_idx[test_ix]).t()
        M_train, M_test = train_pos.size(1), test_pos.size(1)
        train_neg = sample_negatives(data, pos_sets, hop_neighbors, endpoint_counts,
                                     u_t, rel, v_t, M_train, strategy=strategy)
        test_neg  = sample_negatives(data, pos_sets, hop_neighbors, endpoint_counts,
                                     u_t, rel, v_t, M_test,  strategy=strategy)
        yield {
            'train_pos': train_pos,
            'train_neg': train_neg,
            'test_pos' : test_pos,
            'test_neg' : test_neg,
        }


def build_train_graph(full_data, folds_by_relation, fold_idx):
    train_data = HeteroData()
    for ntype in full_data.node_types:
        train_data[ntype].x = full_data[ntype].x
    for (u_t, rel, v_t) in full_data.edge_types:
        key    = f"{u_t}-{rel}-{v_t}"
        e_full = full_data[u_t, rel, v_t].edge_index
        test_pos = folds_by_relation[key][fold_idx]['test_pos']
        test_set = {tuple(x) for x in test_pos.t().tolist()}
        mask = [(tuple(e_full[:, i].tolist()) not in test_set)
                for i in range(e_full.size(1))]
        mask = torch.tensor(mask, dtype=torch.bool, device=e_full.device)
        train_data[u_t, rel, v_t].edge_index = e_full[:, mask]
    return train_data


def compute_grouped_metrics(y_true_dict, y_prob_dict, bipartite_keys, similarity_keys):
    rel_metrics = {}
    for rel_key in y_true_dict.keys():
        y_true_rel = torch.cat(y_true_dict[rel_key]).numpy()
        y_prob_rel = torch.cat(y_prob_dict[rel_key]).numpy()
        y_pred_rel = (y_prob_rel >= 0.5).astype(int)
        f1   = f1_score(y_true_rel, y_pred_rel)
        auc  = roc_auc_score(y_true_rel, y_prob_rel)
        aupr = average_precision_score(y_true_rel, y_prob_rel)
        rel_metrics[rel_key] = (f1, auc, aupr)
    bip_metrics = {k: rel_metrics[k] for k in rel_metrics if k in bipartite_keys}
    sim_metrics = {k: rel_metrics[k] for k in rel_metrics if k in similarity_keys}
    if sim_metrics:
        sim_f1   = sum(m[0] for m in sim_metrics.values()) / len(sim_metrics)
        sim_auc  = sum(m[1] for m in sim_metrics.values()) / len(sim_metrics)
        sim_aupr = sum(m[2] for m in sim_metrics.values()) / len(sim_metrics)
    else:
        sim_f1, sim_auc, sim_aupr = None, None, None
    return rel_metrics, bip_metrics, (sim_f1, sim_auc, sim_aupr)


def normalize_rel_key(rel_key):
    if isinstance(rel_key, tuple):
        return "-".join(rel_key)
    if isinstance(rel_key, str):
        if "#" in rel_key:
            return rel_key.split("#")[0]
        return rel_key
    return str(rel_key)


def global_id_to_name(node_id: int):
    if node_id < offsets["disease"]:
        return drugs[node_id]
    elif node_id < offsets["protein"]:
        return diseases[node_id - offsets["disease"]]
    else:
        return proteins[node_id - offsets["protein"]]


def predict_links_for_pairs_unified(H_fused, model, pairs, device, batch_size=4096):
    """
    Original batched inference over a pre-built pairs tensor.
    Safe for small pair counts (drug–disease).
    For drug–protein or disease–protein, prefer predict_and_filter_streaming().
    """
    preds = []
    H_cpu = H_fused.cpu()
    model.eval()
    with torch.no_grad():
        for start in range(0, pairs.size(0), batch_size):
            end = min(start + batch_size, pairs.size(0))
            batch_pairs = pairs[start:end]

            hu = H_cpu[batch_pairs[:, 0]].to(device)
            hv = H_cpu[batch_pairs[:, 1]].to(device)
            x  = torch.cat([hu, hv], dim=-1)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model.link_mlp(x)

            probs = torch.sigmoid(logits.float()).cpu()
            preds.append(probs)

            del hu, hv, x, logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if (start // batch_size) % 100 == 0:
                print(f"  Predicted {end}/{pairs.size(0)} pairs")

    return torch.cat(preds, dim=0)



def predict_and_filter_streaming(
    H_fused,
    model,
    src_global_start: int,
    src_global_end:   int,
    dst_global_start: int,
    dst_global_end:   int,
    device,
    threshold: float = 0.5,
    # Peak GPU memory = src_batch × dst_batch × 2 × hidden_dim × 2 bytes
    # (32 × 1024 × 2 × 512 × 2) = 64 MB
    src_batch: int   = 32,
    dst_batch: int   = 1024,
):

    n_src   = src_global_end - src_global_start
    n_dst   = dst_global_end - dst_global_start
    total   = n_src * n_dst
    n_src_batches = (n_src + src_batch - 1) // src_batch
    n_dst_batches = (n_dst + dst_batch - 1) // dst_batch

    pos_pairs_list  = []
    pos_scores_list = []

    H_cpu = H_fused.cpu()
    model.eval()

    processed = 0
    with torch.no_grad():
        pbar = tqdm(total=total, desc="Streaming pairs", unit="pair",
                    unit_scale=True)

        for sb in range(n_src_batches):
            s0 = src_global_start + sb * src_batch
            s1 = min(s0 + src_batch, src_global_end)
            hu_block = H_cpu[s0:s1].to(device)   # [B_s, H]
            B_s = hu_block.size(0)

            for db in range(n_dst_batches):
                d0 = dst_global_start + db * dst_batch
                d1 = min(d0 + dst_batch, dst_global_end)
                hv_block = H_cpu[d0:d1].to(device)   # [B_d, H]
                B_d = hv_block.size(0)

                # Expand into all (src, dst) pairs via broadcasting (no copy)
                hu_exp = hu_block.unsqueeze(1).expand(B_s, B_d, -1)
                hv_exp = hv_block.unsqueeze(0).expand(B_s, B_d, -1)

                x = torch.cat([
                    hu_exp.reshape(B_s * B_d, -1),
                    hv_exp.reshape(B_s * B_d, -1)
                ], dim=-1)

                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    logits = model.link_mlp(x)

                probs = torch.sigmoid(logits.float()).squeeze(-1)  # [B_s*B_d]

                mask = probs > threshold
                if mask.any():
                    flat_pos  = mask.nonzero(as_tuple=True)[0].cpu()
                    src_local = flat_pos // B_d
                    dst_local = flat_pos %  B_d
                    global_src = (s0 + src_local).long()
                    global_dst = (d0 + dst_local).long()
                    pos_pairs_list.append(
                        torch.stack([global_src, global_dst], dim=1)
                    )
                    pos_scores_list.append(probs[mask].cpu().float())

                del hv_block, hu_exp, hv_exp, x, logits, probs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                processed += B_s * B_d
                pbar.update(B_s * B_d)

            del hu_block
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        pbar.close()

    if pos_pairs_list:
        return torch.cat(pos_pairs_list, dim=0), torch.cat(pos_scores_list, dim=0)
    else:
        return torch.zeros(0, 2, dtype=torch.long), torch.zeros(0)
