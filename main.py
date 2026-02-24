from utils import *
from model import *

###########-------------- ADRD Knowledge Graph Construction --------------###########

#select dataset to run CLEAR model on
Dataset_name = "ADRD" #ADRD, C, F, Y, LAGCN and LAGCN

#Load lists of FDA-approved drugs, Neurological diseases, and assicated proteins
#here each nodes are mapped to standard IDs such as DrugBank, Mesh and UniProt IDs
print("Loading node IDs...")
with open(f'{Dataset_name}/input_network/node_ids/drugs.pkl', 'rb') as f:
    drugs = pickle.load(f)

with open(f'{Dataset_name}/input_network/node_ids/diseases.pkl', 'rb') as f:
    diseases = pickle.load(f)

with open(f'{Dataset_name}/input_network/node_ids/proteins.pkl', 'rb') as f:
    proteins = pickle.load(f)


#Create mapping from name -> integer index for each node type
drug2idx    = {name: i for i, name in enumerate(drugs)}
disease2idx = {name: i for i, name in enumerate(diseases)}
prot2idx    = {name: i for i, name in enumerate(proteins)}


###########-------------- Initial Node Embeddings Generated using Pretrainde LLM --------------###########
#Load pre-compute node embeddings
print("Loading pre-compute node embeddings...")
drug_feats_df    = pd.read_csv(f'{Dataset_name}/input_network/initial_node_features/drug_node.csv', index_col = 0)  # shape (N_drug, 768)

disease_feats_df = pd.read_csv(f'{Dataset_name}/input_network/initial_node_features/disease_node.csv', index_col = 1)  # shape (N_disease, 768)
disease_feats_df = disease_feats_df.drop(columns = ['Unnamed: 0'])

protein_feats_df    = pd.read_csv(f'{Dataset_name}/input_network/initial_node_features/protein_node.csv', index_col = 0)  # shape (N_protein, 1280)
protein_feats_df = protein_feats_df.drop_duplicates()

###########-------------- Initial Node Embedding Imputation --------------###########
#Impute missing node embeddings expecially for drugs as not all FDA approved drugs have SMILES
print("Imputing missing node embeddings...")
num_drugs    = len(drug2idx)
num_diseases = len(disease2idx)
num_prots    = len(prot2idx)

#Initialize feature matrices with zeros
drug_feats    = torch.zeros((num_drugs,    768))
disease_feats = torch.zeros((num_diseases, 768))
protein_feats    = torch.zeros((num_prots,    1280))

#list of nodes with missing intial node embeddings
missing_drugs, missing_diseases, missing_proteins = [], [], []

#Impute missing node embeddings for drugs
for drug, idx in drug2idx.items():
    if drug in drug_feats_df.index:
        drug_feats[idx] = torch.tensor(drug_feats_df.loc[drug].values, dtype = torch.float)
    else:
        missing_drugs.append((drug, idx))

#Impute missing node embeddings for diseases
for disease, idx in disease2idx.items():
    if disease in disease_feats_df.index:
        disease_feats[idx] = torch.tensor(disease_feats_df.loc[disease].values, dtype = torch.float)
    else:
        missing_diseases.append((disease, idx))

#Impute missing node embeddings for proteins
for protein, idx in prot2idx.items():
    if protein in protein_feats_df.index:
        protein_feats[idx] = torch.tensor(protein_feats_df.loc[protein].values, dtype = torch.float)
    else:
        missing_proteins.append((protein, idx))

drug_feats    = impute_missing(drug_feats, missing_drugs, len(drug2idx))
disease_feats = impute_missing(disease_feats, missing_diseases, len(disease2idx))
protein_feats    = impute_missing(protein_feats, missing_proteins, len(prot2idx))

###########`#####-------------- Load Similarity and Bipartite links --------------###########
#similarity edges
print("Loading similarity edges...")
dr_dr_idx, dr_dr_w = load_similarity_edges(f"{Dataset_name}/input_network/sim_net/drug_sim.csv", drug2idx, drug2idx)
di_di_idx, di_di_w = load_similarity_edges(f"{Dataset_name}/input_network/sim_net/disease_sim.csv", disease2idx, disease2idx)
pr_pr_idx, pr_pr_w = load_similarity_edges(f"{Dataset_name}/input_network/sim_net/protein_sim.csv", prot2idx, prot2idx)

#Bipartite edges
print("Loading bipartite edges...")
dr_di_idx, dr_di_w = load_bipartite_edges(f"{Dataset_name}/input_network/bipartite_net/drug_disease.csv", drug2idx, disease2idx)
di_pr_idx, di_pr_w = load_bipartite_edges(f"{Dataset_name}/input_network/bipartite_net/disease_protein.csv", disease2idx, prot2idx)
dr_pr_idx, dr_pr_w = load_bipartite_edges(f"{Dataset_name}/input_network/bipartite_net/drug_protein.csv", drug2idx, prot2idx)


#############-------------- Create PyG Data Object --------------################
#data object
print("Creating PyG data object...")
data = HeteroData()

#assign node features
data["drug"].x    = drug_feats
data["disease"].x = disease_feats
data["protein"].x = protein_feats

# drug similarity edges
data["drug", "sim", "drug"].edge_index  = dr_dr_idx
data["drug", "sim", "drug"].edge_weight = dr_dr_w

# disease similarity edges
data["disease", "sim", "disease"].edge_index  = di_di_idx
data["disease", "sim", "disease"].edge_weight = di_di_w

# protein similarity edges
data["protein", "sim", "protein"].edge_index  = pr_pr_idx
data["protein", "sim", "protein"].edge_weight = pr_pr_w


#Drug-disease bipartite edges
data["drug", "interacts", "disease"].edge_index    = dr_di_idx
data["drug", "interacts", "disease"].edge_weight   = dr_di_w

#Disease-protein bipartite edges
data["disease", "assoc", "protein"].edge_index   = di_pr_idx
data["disease", "assoc", "protein"].edge_weight  = di_pr_w

#Drug-protein bipartite edges
data["drug", "binds", "protein"].edge_index    = dr_pr_idx
data["drug", "binds", "protein"].edge_weight   = dr_pr_w


###########-------------- Negative sampling and Cross-validation --------------###########

#############-------------- Positive Edges Degree Distribution Probability --------------##############
#Build a networkx graph for hop-neighborhood queries
#graph object
print("Building networkx graph...")
G_nx = nx.Graph()

#add nodes
for ntype in data.node_types: #drug, disease, protein
    G_nx.add_nodes_from(
        [(f"{ntype}_{i}", {"type": ntype}) for i in range(data[ntype].num_nodes)]
    )

#add edges
for (src_type, rel, dst_type) in data.edge_types:
    src, dst = data[(src_type, rel, dst_type)].edge_index
    edges = [(f"{src_type}_{u}", f"{dst_type}_{v}") for u, v in zip(src.tolist(), dst.tolist())]
    G_nx.add_edges_from(edges)

#precompute per-node degree distributions
#endpoint_counts maps each edge type to its (P_src, P_dst) tuple.
endpoint_counts = {}
for (u_t, rel, v_t) in data.edge_types:
    src, dst = data[(u_t, rel, v_t)].edge_index

    # count how often each node appears as src or dst in POSITIVE edges
    src_counts = torch.bincount(src, minlength=data[u_t].num_nodes).float()
    dst_counts = torch.bincount(dst, minlength=data[v_t].num_nodes).float()

    # normalize into probability vectors
    P_src = (src_counts / src_counts.sum())
    P_dst = (dst_counts / dst_counts.sum())

    #add a small constant to make sure if nodes are isolated, they still have a small probability
    src_counts += 1e-6
    dst_counts += 1e-6

    #store the result
    endpoint_counts[(u_t, rel, v_t)] = (P_src, P_dst)

#############-------------- Precompute 3-hop neighbors --------------##############
#use same networkx graph as for positive edges degree distribution probability
#precompute hop neighbors

print("Precomputing 3-hop neighbors...")

# Uncomment to precompute 3-hop neighbors if not already precomputed

# hop_neighbors = precompute_hop_neighbors(G_nx, data, k_hop = 3)

# # Create the directory if it doesn't already exist
# os.makedirs(f"{Dataset_name}/intermediate_data_save", exist_ok=True)

# print("Saving 3-hop neighbors to file...")
# #save precomputed 3-hop neighbors for faster loading
# with open(f"{Dataset_name}/intermediate_data_save/precomputed_hop_neighbors.pkl", "wb") as f:
#     pickle.dump(hop_neighbors, f)

print("Loading 3-hop neighbors from file...")
with open(f"{Dataset_name}/intermediate_data_save/precomputed_hop_neighbors.pkl", "rb") as f:
    hop_neighbors = pickle.load(f)


###########-------------- Neagtive Sampling per Relation-type --------------###########
print("Building positive edge sets...")
pos_sets = build_pos_sets(data)

# Uncomment to precompute negative sampling per relation-type if not already precomputed

folds_by_relation = {}
relations = list(data.edge_types)

#outer bar: iterate through each (u_t,rel,v_t)
for u_t, rel, v_t in tqdm(relations, desc = "Relations", unit = "rel"):
    key = f"{u_t}-{rel}-{v_t}"
    
    # iner bar: wrap the generator from k_fold_link_splits
    folds = list(
        tqdm(
            k_fold_link_splits(
                data, #data object
                pos_sets, #positive edge set
                hop_neighbors, #3-hop neighbors
                endpoint_counts, #endpoint counts, P_src and P_dst for source and destination nodes
                G_nx, #networkx graph
                u_t, rel, v_t, #edge type
                k = 5, #number of folds
                random_state = 40, #random state
                strategy = "3-hop"    #  "random" | "3-hop" | "pdf": probability degree distribution of source and destination nodes
            ),
            total   = 5, #number of folds
            desc = f"Folds for {key}", #description
            unit = "fold", #unit
            leave = False #leave
        )
    )
    
    folds_by_relation[key] = folds

#save precomputed neighbors for faster loading
with open(f"{Dataset_name}/intermediate_data_save/negative_sampling_per_relation.pkl", "wb") as f:
    pickle.dump(folds_by_relation, f)

with open(f"{Dataset_name}/intermediate_data_save/negative_sampling_per_relation.pkl", "rb") as f:
    folds_by_relation = pickle.load(f)


#############-------------- Data Loader --------------##############
print("Loading data loader...")


#############-------------- Model initialization --------------##############
print("Initializing model...")
device = torch.device("cuda:3" if torch.cuda.device_count() > 1 else ("cuda" if torch.cuda.is_available() else "cpu"))
# device = torch.device("cpu")

data   = data.to(device)

model = ADRD_LinkPredictor(
    in_dims      = {'drug':768, 'disease':768, 'protein':1280},
    hidden_dim   = 1024,
    gat_heads    = 8,
    fusion_heads = 8,
    beta         = 0.75,
    dropout      = 0.35
).to(device)

opt = torch.optim.Adam(model.parameters(), lr = 0.0001, weight_decay = 1e-6)

train_ds = FullGraphLinkDataset(folds_by_relation, 0, "train")
test_ds  = FullGraphLinkDataset(folds_by_relation, 0, "test")

train_ld = DataLoader(train_ds, batch_size = 1024, shuffle = True)
test_ld  = DataLoader(test_ds,  batch_size = 1024, shuffle = False)


# --- define relation groups ---
bipartite_keys = {
    "drug-interacts-disease",
    "disease-assoc-protein",
    "drug-binds-protein"
}
similarity_keys = {
    "drug-sim-drug",
    "disease-sim-disease",
    "protein-sim-protein"
}

#############-------------- Main Training Loop --------------##############

print("Starting main training loop...")
train_f1s, train_aucs, train_auprs, train_losses = [], [], [], []
val_f1s, val_aucs, val_auprs, val_losses = [], [], [], []

for epoch in range(1, 2):

    # ====== TRAIN PHASE ======
    model.train()
    train_graph = build_train_graph(data, folds_by_relation, 0).to(device)
    H_fused, offsets = model.compute_full_graph_embeddings(train_graph)

    total_graph_loss = 0.0
    total_samples    = 0

    # store per-relation predictions
    train_probs_dict  = defaultdict(list)
    train_labels_dict = defaultdict(list)

    opt.zero_grad()

    for rel_key_batch, u_t, v_t, u_idx, v_idx, lbl in train_ld:
        u_idx, v_idx, lbl = u_idx.to(device), v_idx.to(device), lbl.to(device)
        logits = model(H_fused, offsets, u_t, v_t, u_idx, v_idx)
        probs  = torch.sigmoid(logits).detach().cpu()

        # unpack batch one edge at a time
        for rk, p, l in zip(rel_key_batch, probs, lbl.detach().cpu()):
            norm_key = normalize_rel_key(rk)  # ensure clean name
            train_probs_dict[norm_key].append(p.unsqueeze(0))
            train_labels_dict[norm_key].append(l.unsqueeze(0))

        # loss still can be computed on the whole batch
        norm_rel_keys = [normalize_rel_key(rk) for rk in rel_key_batch]
        loss = model.compute_loss(logits, lbl, norm_rel_keys)
        total_graph_loss += loss * lbl.size(0)
        total_samples    += lbl.size(0)

    avg_loss = total_graph_loss / total_samples
    avg_loss.backward()
    opt.step()

    # compute grouped metrics
    rel_metrics, bip_metrics, sim_avg_metrics = compute_grouped_metrics(
        train_labels_dict, train_probs_dict, bipartite_keys, similarity_keys
    )

    print(f"\nEpoch {epoch:02d} - TRAIN:")
    # print bipartite metrics
    for k, (f1, auc, aupr) in bip_metrics.items():
        print(f"  {k:<25} | F1={f1:.4f} AUC={auc:.4f} AUP={aupr:.4f}")
    # print similarity avg
    if sim_avg_metrics[0] is not None:
        print(f"  [Similarity Avg]        | F1={sim_avg_metrics[0]:.4f} "
              f"AUC={sim_avg_metrics[1]:.4f} AUP={sim_avg_metrics[2]:.4f}\n")

    # store overall bipartite metric (choose one relation or avg)
    if "drug-interacts-disease" in bip_metrics:
        train_f1s.append(bip_metrics['drug-interacts-disease'][0])
        train_aucs.append(bip_metrics['drug-interacts-disease'][1])
        train_auprs.append(bip_metrics['drug-interacts-disease'][2])
    else:
        train_f1s.append(0.0)
        train_aucs.append(0.0)
        train_auprs.append(0.0)
    train_losses.append(avg_loss.item())


    # ====== VALIDATION PHASE ======
    model.eval()
    total_val_loss = 0.0
    val_samples    = 0
    val_probs_dict  = defaultdict(list)
    val_labels_dict = defaultdict(list)

    with torch.no_grad():
        for rel_key_batch, u_t, v_t, u_idx, v_idx, lbl in test_ld:
            u_idx, v_idx, lbl = u_idx.to(device), v_idx.to(device), lbl.to(device)
            logits = model(H_fused, offsets, u_t, v_t, u_idx, v_idx)
            probs  = torch.sigmoid(logits).cpu()

            # unpack each edge separately
            for rk, p, l in zip(rel_key_batch, probs, lbl.cpu()):
                norm_key = normalize_rel_key(rk)
                val_probs_dict[norm_key].append(p.unsqueeze(0))
                val_labels_dict[norm_key].append(l.unsqueeze(0))

            norm_rel_keys = [normalize_rel_key(rk) for rk in rel_key_batch]
            loss = model.compute_loss(logits, lbl, norm_rel_keys)
            total_val_loss += loss * lbl.size(0)
            val_samples    += lbl.size(0)

    avg_val_loss = total_val_loss / val_samples

    # compute grouped metrics
    val_rel_metrics, val_bip_metrics, val_sim_avg = compute_grouped_metrics(
        val_labels_dict, val_probs_dict, bipartite_keys, similarity_keys
    )

    print(f"Epoch {epoch:02d} - VALIDATION:")
    for k, (f1, auc, aupr) in val_bip_metrics.items():
        print(f"  {k:<25} | F1={f1:.4f} AUC={auc:.4f} AUP={aupr:.4f}")
    if val_sim_avg[0] is not None:
        print(f"  [Similarity Avg]        | F1={val_sim_avg[0]:.4f} "
              f"AUC={val_sim_avg[1]:.4f} AUP={val_sim_avg[2]:.4f}\n")
    print(f"  Loss -> Train:{avg_loss:.4f}  Val:{avg_val_loss:.4f}\n")

    # store val metrics (example: just drug-interacts-disease)
    if "drug-interacts-disease" in val_bip_metrics:
        val_f1s.append(val_bip_metrics['drug-interacts-disease'][0])
        val_aucs.append(val_bip_metrics['drug-interacts-disease'][1])
        val_auprs.append(val_bip_metrics['drug-interacts-disease'][2])
    else:
        val_f1s.append(0.0)
        val_aucs.append(0.0)
        val_auprs.append(0.0)
    val_losses.append(avg_val_loss.item())

    #Save the final node embeddings as tensor
    print(f"Saved node embeddings")
    torch.save(H_fused, f"{Dataset_name}/intermediate_data_save/CLEAR_node_embeddings.pt")

    print(f"Saved model parameters and weights")
    torch.save(model.state_dict(), f"{Dataset_name}/intermediate_data_save/trained_model_weights.pt")
    