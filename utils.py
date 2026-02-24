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

#Impute missing node embeddings by sampling from the nearest neighbors
def impute_missing(feats, missing_idxs, sample_k):
    all_idxs = [i for i in range(feats.size(0)) if i not in missing_idxs]
    for _ , mi in missing_idxs:
        sampled = random.sample(all_idxs, sample_k)
        feats[mi] = feats[sampled].mean(dim = 0)
    return feats

#read and load similarity edges as edge weight tensor
#use same index mapping for source and destination nodes as in the node embeddings
def load_similarity_edges(file_path, src2idx, dst2idx):

    #read similarity edges from csv file
    df = pd.read_csv(file_path)
    df = df.drop(columns = ['Unnamed: 0'])
    #map to indices
    src = [src2idx[x] for x in df.iloc[:, 0]]
    dst = [dst2idx[x] for x in df.iloc[:, 1]]
    #generate edge index tensor
    edge_index = torch.tensor([src + dst, dst + src], dtype = torch.long)  # make undirected as similarity edges are bidirectional
    #generate edge weight tensor
    edge_weight = torch.ones(edge_index.size(1), dtype = torch.float)

    return edge_index, edge_weight

#read and load bipartite edges as edge weight tensor
#edge weight is 1 for all edges as they are binary
def load_bipartite_edges(file_path, src2idx, dst2idx):

    #read bipartite edges from csv file
    df = pd.read_csv(file_path)
    try:
        df = df.drop(columns = ['Unnamed: 0'])
    except KeyError:
        pass
    #map to indices
    src = [src2idx[x] for x in df.iloc[:, 0]]
    dst = [dst2idx[x] for x in df.iloc[:, 1]]
    #generate edge index tensor
    edge_index = torch.tensor([src,dst], dtype = torch.long)  # make undirected as associations are bidirectional
    #generate edge weight tensor
    edge_weight = torch.ones(edge_index.size(1), dtype = torch.float)

    return edge_index, edge_weight

def precompute_hop_neighbors(G_nx, data, k_hop):
    hop_neighbors = {}

    #outer bar: one entry per (u_t,rel,v_t)
    rel_iter = tqdm(
        data.edge_types,
        desc="Relations",
        unit="rel"
    )

    for (u_t, rel, v_t) in rel_iter:
        N_u = data[u_t].num_nodes
        uv_neighbors = {}

        #inner bar: N_u nodes per relation
        node_iter = tqdm(
            range(N_u),
            desc=f"{u_t}→{v_t} nodes",
            unit="u",
            leave=False
        )

        prefix = f"{v_t}_"
        for u in node_iter:
            u_node = f"{u_t}_{u}"
            reachable = nx.single_source_shortest_path_length(
                G_nx, u_node, cutoff=k_hop
            ).keys()

            # collect only those of type v_t
            vt_set = {
                int(n.split("_", 1)[1])
                for n in reachable
                if n.startswith(prefix)
            }
            uv_neighbors[u] = vt_set

        hop_neighbors[(u_t, v_t)] = uv_neighbors

    return hop_neighbors

#uniform random negative sampling per relation
def sample_neg_uniform(data, pos_sets, u_t, rel, v_t, num_samples):
    #number of source and destination nodes
    N_u, N_v = data[u_t].num_nodes, data[v_t].num_nodes
    #positive edge set
    pos = pos_sets[(u_t, rel, v_t)]
    #negative edge set
    neg = []

    #draw negative samples until we have num_samples
    while len(neg) < num_samples:
        #draw source and destination nodes uniformly
        u = random.randrange(N_u)
        v = random.randrange(N_v)

        #if the edge is not in the positive edge set, add it to the negative edge set
        if (u, v) not in pos:
            neg.append((u, v))

    return torch.tensor(neg).t()

#sample negative edges with k-hop exclusion
def sample_neg_3hop(data, pos_sets, hop_neighbors, u_t, rel, v_t, num_samples):

    #number of source and destination nodes
    N_u, N_v = data[u_t].num_nodes, data[v_t].num_nodes
    #positive edge set
    pos   = pos_sets[(u_t, rel, v_t)]
    #3-hop neighbors
    hop_n = hop_neighbors[(u_t, v_t)]

    neg = []

    #draw negative samples until we have num_samples
    while len(neg) < num_samples:
        #draw source and destination nodes uniformly
        u = random.randrange(N_u)
        v = random.randrange(N_v)

        #if the edge is in the positive edge set, skip
        if (u, v) in pos:
            continue
        #if the destination node is in the 3-hop neighbors of the source node, skip
        if v in hop_n[u]:
            continue
        neg.append((u, v))

    return torch.tensor(neg).t()

#sample negative edges with degree-matched and k-hop exclusion
def sample_neg_prob_deg(data,  #data object
                        pos_sets, #positive edge set
                        hop_neighbors, #3-hop neighbors
                        endpoint_counts, #endpoint counts, P_src and P_dst for source and destination nodes
                        u_t, rel, v_t, #edge type
                        num_samples, #number of negative samples
                        max_attempts): #maximum number of attempts

    #number of source and destination nodes
    N_u, N_v    = data[u_t].num_nodes, data[v_t].num_nodes
    #probability vectors for source and destination nodes
    P_src, P_dst= endpoint_counts[(u_t, rel, v_t)]
    #positive edge set
    pos = pos_sets[(u_t, rel, v_t)]
    #3-hop neighbors
    hop_n = hop_neighbors[(u_t, v_t)]

    neg = []
    for _ in range(num_samples):
        # try degree-matched draws
        for _try in range(max_attempts):
            #draw source node from P_src
            u = np.random.choice(N_u, p = P_src)
            #draw destination node from P_dst
            v = np.random.choice(N_v, p = P_dst)
            #if the edge is in the positive edge set or the destination node is in the k-hop neighbors of the source node, skip
            if (u, v) in pos or v in hop_n[u]:
                continue
            neg.append((u, v))
            break
        #if we have tried max_attempts and still haven't found a valid negative sample, fallback to uniform + hop check
        else:
            # fallback to uniform + hop check
            while True:
                u = random.randrange(N_u)
                v = random.randrange(N_v)

                if (u, v) in pos:
                    continue

                if (u, v) in pos or v in hop_n[u]:
                    continue

                neg.append((u, v))
                break
    return torch.tensor(neg).t()


def sample_negatives(data, #data object
                     pos_sets, #positive edge set
                     hop_neighbors, #3-hop neighbors
                     endpoint_counts, #endpoint counts, P_src and P_dst for source and destination nodes
                     u_t, rel, v_t, #edge type
                     num_samples, #number of negative samples
                     strategy): #sampling strategy

    if strategy == "random":
        return sample_neg_uniform(data, pos_sets, u_t, rel, v_t, num_samples)

    elif strategy == "3-hop":
        return sample_neg_3hop(data, pos_sets, hop_neighbors, u_t, rel, v_t, num_samples)

    elif strategy == "pdf":
        return sample_neg_prob_deg(
            data, pos_sets, hop_neighbors, endpoint_counts,
            u_t, rel, v_t, num_samples, max_attempts = 20)

    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")

#build positive edge sets
def build_pos_sets(data):
    pos_sets = {}
    for (u_t, rel, v_t) in data.edge_types:
        edge_index = data[(u_t, rel, v_t)].edge_index.t().tolist()
        pos_sets[(u_t, rel, v_t)] = set(tuple(e) for e in edge_index)
    return pos_sets

def k_fold_link_splits(data, #data object
                       pos_sets, #positive edge set
                       hop_neighbors, #3-hop neighbors
                       endpoint_counts, #endpoint counts, P_src and P_dst for source and destination nodes
                       G_nx, #networkx graph
                       u_t, # source node type
                       rel, # relation type
                       v_t, # destination node type
                       k, # number of folds
                       random_state,
                       strategy):  # "random"|"3-hop"|"pdf"

    #positive edge set indices
    pos_idx = data[(u_t, rel, v_t)].edge_index.t().tolist()
    pos_idx = np.array(pos_idx)

    #k-fold cross-validation
    kf = KFold(n_splits = k, shuffle = True, random_state = random_state)

    #iterate over folds
    #here splits means different relation types (u_t, rel, v_t)
    #train_ix and test_ix are indices of the positive edge set
    for train_ix, test_ix in kf.split(pos_idx):
        #train and test positive edge sets
        train_pos = torch.tensor(pos_idx[train_ix]).t()
        test_pos  = torch.tensor(pos_idx[test_ix]).t()

        M_train = train_pos.size(1)
        M_test  = test_pos.size(1)

        #select negative edges for train and test sets
        train_neg = sample_negatives(
            data, pos_sets, hop_neighbors, endpoint_counts,
            u_t, rel, v_t, M_train,
            strategy = strategy
        )
        test_neg = sample_negatives(
            data, pos_sets, hop_neighbors, endpoint_counts,
            u_t, rel, v_t, M_test,
            strategy = strategy
        )

        #yield the train and test sets for the current fold
        yield {
            'train_pos': train_pos,
            'train_neg': train_neg,
            'test_pos' : test_pos,
            'test_neg' : test_neg,
        }

def build_train_graph(full_data, folds_by_relation, fold_idx):
    train_data = HeteroData()
    # 1) copy node features
    for ntype in full_data.node_types:
        train_data[ntype].x = full_data[ntype].x

    # 2) for each relation, remove test edges
    for (u_t, rel, v_t) in full_data.edge_types:
        key    = f"{u_t}-{rel}-{v_t}"
        e_full = full_data[u_t, rel, v_t].edge_index

        # gather the test‐pos edges for this relation/fold
        test_pos = folds_by_relation[key][fold_idx]['test_pos']  # [2, E_test]
        test_set = {tuple(x) for x in test_pos.t().tolist()}

        # build a boolean mask: keep only edges NOT in test_set
        mask = [(tuple(e_full[:, i].tolist()) not in test_set)
                for i in range(e_full.size(1))]
        mask = torch.tensor(mask, dtype=torch.bool, device=e_full.device)

        # assign the pruned edges
        train_data[u_t, rel, v_t].edge_index = e_full[:, mask]

    return train_data

def compute_grouped_metrics(y_true_dict, y_prob_dict, bipartite_keys, similarity_keys):
    rel_metrics = {}

    # Compute per-relation metrics
    for rel_key in y_true_dict.keys():
        y_true_rel = torch.cat(y_true_dict[rel_key]).numpy()
        y_prob_rel = torch.cat(y_prob_dict[rel_key]).numpy()
        y_pred_rel = (y_prob_rel >= 0.5).astype(int)

        f1   = f1_score(y_true_rel, y_pred_rel)
        auc  = roc_auc_score(y_true_rel, y_prob_rel)
        aupr = average_precision_score(y_true_rel, y_prob_rel)
        rel_metrics[rel_key] = (f1, auc, aupr)

    # Separate bipartite vs similarity
    bip_metrics = {k: rel_metrics[k] for k in rel_metrics if k in bipartite_keys}
    sim_metrics = {k: rel_metrics[k] for k in rel_metrics if k in similarity_keys}

    # Average similarity metrics
    if sim_metrics:
        sim_f1   = sum(m[0] for m in sim_metrics.values()) / len(sim_metrics)
        sim_auc  = sum(m[1] for m in sim_metrics.values()) / len(sim_metrics)
        sim_aupr = sum(m[2] for m in sim_metrics.values()) / len(sim_metrics)
    else:
        sim_f1, sim_auc, sim_aupr = None, None, None

    return rel_metrics, bip_metrics, (sim_f1, sim_auc, sim_aupr)

# --- helper: normalize relation key ---
def normalize_rel_key(rel_key):
    # Handle tuple relation keys
    if isinstance(rel_key, tuple):
        return "-".join(rel_key)
    # Handle strings with possible suffixes
    if isinstance(rel_key, str):
        if "#" in rel_key:
            return rel_key.split("#")[0]
        return rel_key
    return str(rel_key)

# Reverse mapping from global ID → original name
def global_id_to_name(node_id: int):
    if node_id < offsets["disease"]:
        return drugs[node_id]
    elif node_id < offsets["protein"]:
        return diseases[node_id - offsets["disease"]]
    else:
        return proteins[node_id - offsets["protein"]]

def predict_links_for_pairs_unified(H_fused, model, pairs, device, batch_size=4096):
    preds = []
    for start in range(0, pairs.size(0), batch_size):
        end = min(start + batch_size, pairs.size(0))
        batch_pairs = pairs[start:end]

        hu = H_fused[batch_pairs[:,0]].to(device)
        hv = H_fused[batch_pairs[:,1]].to(device)

        x = torch.cat([hu, hv], dim=-1)   # [batch, 2*F]
        logits = model.link_mlp(x)
        probs = torch.sigmoid(logits).cpu()
        preds.append(probs)

        # Clear CUDA memory after each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (start // batch_size) % 100 == 0:
            print(f"  Predicted {end}/{pairs.size(0)} edges")


    return torch.cat(preds, dim=0)
