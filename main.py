import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from utils import *
from model import *


Dataset_name = "ADRD_dataset"

print("Loading node IDs...")
with open(f'{Dataset_name}/input_network/node_ids/drugs.pkl', 'rb') as f:
    drugs = pickle.load(f)
with open(f'{Dataset_name}/input_network/node_ids/diseases.pkl', 'rb') as f:
    diseases = pickle.load(f)
with open(f'{Dataset_name}/input_network/node_ids/proteins.pkl', 'rb') as f:
    proteins = pickle.load(f)

drug2idx    = {name: i for i, name in enumerate(drugs)}
disease2idx = {name: i for i, name in enumerate(diseases)}
prot2idx    = {name: i for i, name in enumerate(proteins)}

print("Loading pre-computed node embeddings...")
drug_feats_df = pd.read_csv(
    f'{Dataset_name}/input_network/initial_node_features/drug_node.csv',
    index_col=0)
disease_feats_df = pd.read_csv(
    f'{Dataset_name}/input_network/initial_node_features/disease_node.csv',
    index_col=1)
disease_feats_df = disease_feats_df.drop(columns=['Unnamed: 0'])
protein_feats_df = pd.read_csv(
    f'{Dataset_name}/input_network/initial_node_features/protein_node.csv',
    index_col=0)
protein_feats_df = protein_feats_df.drop_duplicates()

print("Imputing missing node embeddings...")
num_drugs, num_diseases, num_prots = len(drug2idx), len(disease2idx), len(prot2idx)

drug_feats    = torch.zeros((num_drugs,    768))
disease_feats = torch.zeros((num_diseases, 768))
protein_feats = torch.zeros((num_prots,    1280))
missing_drugs, missing_diseases, missing_proteins = [], [], []

for drug, idx in drug2idx.items():
    if drug in drug_feats_df.index:
        drug_feats[idx] = torch.tensor(drug_feats_df.loc[drug].values, dtype=torch.float)
    else:
        missing_drugs.append((drug, idx))

for disease, idx in disease2idx.items():
    if disease in disease_feats_df.index:
        disease_feats[idx] = torch.tensor(disease_feats_df.loc[disease].values, dtype=torch.float)
    else:
        missing_diseases.append((disease, idx))

for protein, idx in prot2idx.items():
    if protein in protein_feats_df.index:
        protein_feats[idx] = torch.tensor(protein_feats_df.loc[protein].values, dtype=torch.float)
    else:
        missing_proteins.append((protein, idx))

drug_feats    = impute_missing(drug_feats,    missing_drugs,    len(drug2idx))
disease_feats = impute_missing(disease_feats, missing_diseases, len(disease2idx))
protein_feats = impute_missing(protein_feats, missing_proteins, len(prot2idx))

print("Loading similarity edges...")
dr_dr_idx, dr_dr_w = load_similarity_edges(
    f"{Dataset_name}/input_network/sim_net/drug_sim.csv",    drug2idx,    drug2idx)
di_di_idx, di_di_w = load_similarity_edges(
    f"{Dataset_name}/input_network/sim_net/disease_sim.csv", disease2idx, disease2idx)
pr_pr_idx, pr_pr_w = load_similarity_edges(
    f"{Dataset_name}/input_network/sim_net/protein_sim.csv", prot2idx,    prot2idx)

print("Loading bipartite edges...")
dr_di_idx, dr_di_w = load_bipartite_edges(
    f"{Dataset_name}/input_network/bipartite_net/drug_disease.csv",    drug2idx,    disease2idx)
di_pr_idx, di_pr_w = load_bipartite_edges(
    f"{Dataset_name}/input_network/bipartite_net/disease_protein.csv", disease2idx, prot2idx)
dr_pr_idx, dr_pr_w = load_bipartite_edges(
    f"{Dataset_name}/input_network/bipartite_net/drug_protein.csv",    drug2idx,    prot2idx)

print("Creating PyG data object...")
data = HeteroData()
data["drug"].x    = drug_feats
data["disease"].x = disease_feats
data["protein"].x = protein_feats

data["drug",    "sim",       "drug"].edge_index    = dr_dr_idx
data["drug",    "sim",       "drug"].edge_weight   = dr_dr_w
data["disease", "sim",       "disease"].edge_index = di_di_idx
data["disease", "sim",       "disease"].edge_weight= di_di_w
data["protein", "sim",       "protein"].edge_index = pr_pr_idx
data["protein", "sim",       "protein"].edge_weight= pr_pr_w
data["drug",    "interacts", "disease"].edge_index = dr_di_idx
data["drug",    "interacts", "disease"].edge_weight= dr_di_w
data["disease", "assoc",     "protein"].edge_index = di_pr_idx
data["disease", "assoc",     "protein"].edge_weight= di_pr_w
data["drug",    "binds",     "protein"].edge_index = dr_pr_idx
data["drug",    "binds",     "protein"].edge_weight= dr_pr_w

print("Building NetworkX graph...")
G_nx = nx.Graph()
for ntype in data.node_types:
    G_nx.add_nodes_from(
        [(f"{ntype}_{i}", {"type": ntype}) for i in range(data[ntype].num_nodes)]
    )
for (src_type, rel, dst_type) in data.edge_types:
    src, dst = data[(src_type, rel, dst_type)].edge_index
    G_nx.add_edges_from(
        [(f"{src_type}_{u}", f"{dst_type}_{v}")
         for u, v in zip(src.tolist(), dst.tolist())]
    )

endpoint_counts = {}
for (u_t, rel, v_t) in data.edge_types:
    src, dst = data[(u_t, rel, v_t)].edge_index
    src_counts = torch.bincount(src, minlength=data[u_t].num_nodes).float()
    dst_counts = torch.bincount(dst, minlength=data[v_t].num_nodes).float()
    P_src = src_counts / src_counts.sum()
    P_dst = dst_counts / dst_counts.sum()
    src_counts += 1e-6
    dst_counts += 1e-6
    endpoint_counts[(u_t, rel, v_t)] = (P_src, P_dst)


print("Precomputing 3-hop neighbors...")
hop_neighbors = precompute_hop_neighbors(G_nx, data, k_hop=3)

os.makedirs(f"{Dataset_name}/intermediate_data_save", exist_ok=True)
with open(f"{Dataset_name}/intermediate_data_save/precomputed_hop_neighbors.pkl", "wb") as f:
    pickle.dump(hop_neighbors, f)
with open(f"{Dataset_name}/intermediate_data_save/precomputed_hop_neighbors.pkl", "rb") as f:
    hop_neighbors = pickle.load(f)

print("Building positive edge sets & sampling negatives...")
pos_sets = build_pos_sets(data)

folds_by_relation = {}
for u_t, rel, v_t in tqdm(list(data.edge_types), desc="Relations", unit="rel"):
    key   = f"{u_t}-{rel}-{v_t}"
    folds = list(tqdm(
        k_fold_link_splits(
            data, pos_sets, hop_neighbors, endpoint_counts,
            G_nx, u_t, rel, v_t,
            k=5, random_state=40, strategy="3-hop"
        ),
        total=5, desc=f"Folds for {key}", unit="fold", leave=False
    ))
    folds_by_relation[key] = folds

with open(f"{Dataset_name}/intermediate_data_save/negative_sampling_per_relation.pkl", "wb") as f:
    pickle.dump(folds_by_relation, f)
with open(f"{Dataset_name}/intermediate_data_save/negative_sampling_per_relation.pkl", "rb") as f:
    folds_by_relation = pickle.load(f)

print("Initializing model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ADRD_LinkPredictor(
    in_dims        = {'drug': 768, 'disease': 768, 'protein': 1280},
    hidden_dim     = 512,          # was 1024
    gat_heads      = 4,            # was 8
    fusion_heads   = 4,            # was 8
    beta           = 0.75,
    dropout        = 0.35,
    use_checkpoint = True,
).to(device)

opt     = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
use_amp = torch.cuda.is_available()

scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

ACCUM_STEPS = 1

pin_memory = (device.type == "cuda")
train_ds = FullGraphLinkDataset(folds_by_relation, 0, "train")
test_ds  = FullGraphLinkDataset(folds_by_relation, 0, "test")
train_ld = DataLoader(train_ds, batch_size=1024, shuffle=True,
                      num_workers=2, pin_memory=pin_memory)
test_ld  = DataLoader(test_ds,  batch_size=1024, shuffle=False,
                      num_workers=2, pin_memory=pin_memory)

bipartite_keys  = {"drug-interacts-disease", "disease-assoc-protein", "drug-binds-protein"}
similarity_keys = {"drug-sim-drug", "disease-sim-disease", "protein-sim-protein"}

print("Starting training loop...")
train_f1s, train_aucs, train_auprs, train_losses = [], [], [], []
val_f1s,   val_aucs,   val_auprs,   val_losses   = [], [], [], []

for epoch in range(1, 2):

    model.train()

    train_graph_gpu = build_train_graph(data, folds_by_relation, 0).to(device)

    with torch.cuda.amp.autocast(enabled=use_amp):
        H_fused, offsets = model.compute_full_graph_embeddings(train_graph_gpu)
    # H_fused: [N_total, hidden_dim], fp16 values under autocast, with GAT grad_fn

    H_leaf = H_fused.detach().requires_grad_(True)

    total_graph_loss  = 0.0
    total_samples     = 0
    train_probs_dict  = defaultdict(list)
    train_labels_dict = defaultdict(list)

    opt.zero_grad(set_to_none=True)

    for step, (rel_key_batch, u_t, v_t, u_idx, v_idx, lbl) in enumerate(train_ld):
        u_idx = u_idx.to(device, non_blocking=True)
        v_idx = v_idx.to(device, non_blocking=True)
        lbl   = lbl.to(device,   non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(H_leaf, offsets, u_t, v_t, u_idx, v_idx)
            norm_rel_keys = [normalize_rel_key(rk) for rk in rel_key_batch]
            loss  = model.compute_loss(logits, lbl, norm_rel_keys)
            loss_scaled_for_accum = loss / ACCUM_STEPS

        scaler.scale(loss_scaled_for_accum).backward()

        total_graph_loss += loss.item() * lbl.size(0)
        total_samples    += lbl.size(0)

        probs = torch.sigmoid(logits.detach()).cpu()
        for rk, p, l in zip(rel_key_batch, probs, lbl.detach().cpu()):
            nk = normalize_rel_key(rk)
            train_probs_dict[nk].append(p.unsqueeze(0))
            train_labels_dict[nk].append(l.unsqueeze(0))

    avg_loss = total_graph_loss / max(total_samples, 1)


    if H_leaf.grad is not None:
        H_fused.backward(H_leaf.grad)


    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)


    del H_leaf, H_fused, train_graph_gpu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rel_metrics, bip_metrics, sim_avg_metrics = compute_grouped_metrics(
        train_labels_dict, train_probs_dict, bipartite_keys, similarity_keys
    )
    print(f"\nEpoch {epoch:02d} – TRAIN:")
    for k, (f1, auc, aupr) in bip_metrics.items():
        print(f"  {k:<25} | F1={f1:.4f}  AUC={auc:.4f}  AUPR={aupr:.4f}")
    if sim_avg_metrics[0] is not None:
        print(f"  [Similarity Avg]          | F1={sim_avg_metrics[0]:.4f}  "
              f"AUC={sim_avg_metrics[1]:.4f}  AUPR={sim_avg_metrics[2]:.4f}")

    if "drug-interacts-disease" in bip_metrics:
        train_f1s.append(bip_metrics['drug-interacts-disease'][0])
        train_aucs.append(bip_metrics['drug-interacts-disease'][1])
        train_auprs.append(bip_metrics['drug-interacts-disease'][2])
    else:
        train_f1s.append(0.0); train_aucs.append(0.0); train_auprs.append(0.0)
    train_losses.append(avg_loss)

    model.eval()
    total_val_loss = 0.0
    val_samples    = 0
    val_probs_dict  = defaultdict(list)
    val_labels_dict = defaultdict(list)

    # Recompute embeddings, no grad graph stored at all.
    val_graph_gpu = build_train_graph(data, folds_by_relation, 0).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
        H_val, val_offsets = model.compute_full_graph_embeddings(val_graph_gpu)
    del val_graph_gpu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with torch.no_grad():
        for rel_key_batch, u_t, v_t, u_idx, v_idx, lbl in test_ld:
            u_idx = u_idx.to(device, non_blocking=True)
            v_idx = v_idx.to(device, non_blocking=True)
            lbl   = lbl.to(device,   non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(H_val, val_offsets, u_t, v_t, u_idx, v_idx)
                norm_rel_keys = [normalize_rel_key(rk) for rk in rel_key_batch]
                loss = model.compute_loss(logits, lbl, norm_rel_keys)

            total_val_loss += loss.item() * lbl.size(0)
            val_samples    += lbl.size(0)

            probs = torch.sigmoid(logits).cpu()
            for rk, p, l in zip(rel_key_batch, probs, lbl.cpu()):
                nk = normalize_rel_key(rk)
                val_probs_dict[nk].append(p.unsqueeze(0))
                val_labels_dict[nk].append(l.unsqueeze(0))

    avg_val_loss = total_val_loss / max(val_samples, 1)

    del H_val
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    val_rel_metrics, val_bip_metrics, val_sim_avg = compute_grouped_metrics(
        val_labels_dict, val_probs_dict, bipartite_keys, similarity_keys
    )
    print(f"\nEpoch {epoch:02d} – VALIDATION:")
    for k, (f1, auc, aupr) in val_bip_metrics.items():
        print(f"  {k:<25} | F1={f1:.4f}  AUC={auc:.4f}  AUPR={aupr:.4f}")
    if val_sim_avg[0] is not None:
        print(f"  [Similarity Avg]          | F1={val_sim_avg[0]:.4f}  "
              f"AUC={val_sim_avg[1]:.4f}  AUPR={val_sim_avg[2]:.4f}")
    print(f"  Loss -> Train: {avg_loss:.4f}  Val: {avg_val_loss:.4f}\n")

    if "drug-interacts-disease" in val_bip_metrics:
        val_f1s.append(val_bip_metrics['drug-interacts-disease'][0])
        val_aucs.append(val_bip_metrics['drug-interacts-disease'][1])
        val_auprs.append(val_bip_metrics['drug-interacts-disease'][2])
    else:
        val_f1s.append(0.0); val_aucs.append(0.0); val_auprs.append(0.0)
    val_losses.append(avg_val_loss)

    print("Saving node embeddings and model weights...")
    full_graph_gpu = data.to(device)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
        H_save, _ = model.compute_full_graph_embeddings(full_graph_gpu)
    torch.save(H_save.cpu(),
               f"{Dataset_name}/intermediate_data_save/CLEAR_node_embeddings.pt")
    torch.save(model.state_dict(),
               f"{Dataset_name}/intermediate_data_save/trained_model_weights.pt")
    del H_save, full_graph_gpu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Epoch {epoch:02d} complete.\n")
