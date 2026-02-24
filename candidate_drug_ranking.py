from model import *
from utils import *

#############-------------- Predict Bipartite Links using Pretrained model --------------##############

########------Node ID indexing--------------##############
Dataset_name = "ADRD" #ADRD, C, F, Y, LAGCN and LAGCN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##########--------------Load node name lists & create mappings--------------##############
print("Loading node IDs...")
with open(f'{Dataset_name}/input_network/node_ids/drugs.pkl', 'rb') as f:
    drugs = pickle.load(f)

with open(f'{Dataset_name}/input_network/node_ids/diseases.pkl', 'rb') as f:
    diseases = pickle.load(f)

with open(f'{Dataset_name}/input_network/node_ids/proteins.pkl', 'rb') as f:
    proteins = pickle.load(f)

num_drug = len(drugs)
num_disease = len(diseases)
num_protein = len(proteins)

offsets = {
    "drug": 0,
    "disease": num_drug,
    "protein": num_drug + num_disease
}

total_nodes = num_drug + num_disease + num_protein
print(f"Nodes: drugs={num_drug}, diseases={num_disease}, proteins={num_protein}, total={total_nodes}")

#############-------------- Load model and Node Embeddings --------------##############
#Load trained model + CLEAR node embeddings

model = ADRD_LinkPredictor(
    in_dims      = {'drug':768, 'disease':768, 'protein':1280},
    hidden_dim   = 1024,
    gat_heads    = 8,
    fusion_heads = 8,
    beta         = 0.75,
    dropout      = 0.35
)

model.load_state_dict(torch.load(f"{Dataset_name}/intermediate_data_save/trained_model_weights.pt", map_location=device))
model.eval().to(device)

H_fused = torch.load(f"{Dataset_name}/intermediate_data_save/CLEAR_node_embeddings.pt", map_location=device)  # shape [total_nodes, emb_dim]
print("Loaded H_fused:", H_fused.shape)


##############-------------- Prepare all possible bipartite pairs --------------##############
#Build global ID ranges for each node type
drug_idx_global    = torch.arange(offsets["drug"], offsets["disease"])
disease_idx_global = torch.arange(offsets["disease"], offsets["protein"])
protein_idx_global = torch.arange(offsets["protein"], total_nodes)

# Cartesian products for all bipartite relations
drug_disease_pairs_global   = torch.cartesian_prod(drug_idx_global, disease_idx_global)
disease_protein_pairs_global = torch.cartesian_prod(disease_idx_global, protein_idx_global)
drug_protein_pairs_global    = torch.cartesian_prod(drug_idx_global, protein_idx_global)

print(f"Candidate pairs:")
print(f"  Drug–Disease = {drug_disease_pairs_global.shape[0]}")
print(f"  Disease–Protein = {disease_protein_pairs_global.shape[0]}")
print(f"  Drug–Protein = {drug_protein_pairs_global.shape[0]}")

#############-------------- Link Prediction using pretrained model on all possible bipartite pairs --------------##############
# ======================
# 5. Predict scores for each bipartite relation
# ======================
print("\nPredicting Drug–Disease edges...")
drug_disease_scores = predict_links_for_pairs_unified(H_fused, model, drug_disease_pairs_global, device)


print("\nPredicting Disease–Protein edges...")
disease_protein_scores = predict_links_for_pairs_unified(H_fused, model, disease_protein_pairs_global, device)


print("\nPredicting Drug–Protein edges...")
drug_protein_scores = predict_links_for_pairs_unified(H_fused, model, drug_protein_pairs_global, device)

    
#Threshold & collect predicted edges
print("Thresholding and collecting predicted edges...")
# Link prediction threshold value
threshold = 0.5  

# Create masks ONCE
mask_dd = drug_disease_scores.squeeze(-1) > threshold
mask_dp = disease_protein_scores.squeeze(-1) > threshold
mask_dr = drug_protein_scores.squeeze(-1) > threshold

# Use same masks for edges
pred_drug_disease_edges   = drug_disease_pairs_global[mask_dd]
pred_disease_protein_edges = disease_protein_pairs_global[mask_dp]
pred_drug_protein_edges    = drug_protein_pairs_global[mask_dr]

predicted_bipartite_edges = torch.cat([
    pred_drug_disease_edges,
    pred_disease_protein_edges,
    pred_drug_protein_edges
], dim=0)

# Use SAME masks for scores
predicted_scores = torch.cat([
    drug_disease_scores.squeeze(-1)[mask_dd],
    disease_protein_scores.squeeze(-1)[mask_dp],
    drug_protein_scores.squeeze(-1)[mask_dr]
], dim=0)

print(f"Total predicted bipartite edges: {predicted_scores.size(0)}")

##############-------------- Save Predicted Bipartite Edges --------------##############
print("Saving predicted bipartite edges...")

num_drug    = len(drugs)
num_disease = len(diseases)
num_protein = len(proteins)

offsets = {
    "drug":    0,
    "disease": num_drug,
    "protein": num_drug + num_disease
}

# map a global node‐ID back to its original name
def global_id_to_name(node_id: int):
    if node_id < offsets["disease"]:
        return drugs[node_id]
    elif node_id < offsets["protein"]:
        return diseases[node_id - offsets["disease"]]
    else:
        return proteins[node_id - offsets["protein"]]

# Build global bipartite‐pair tensors
drug_idx    = torch.arange(num_drug)
disease_idx = torch.arange(num_disease)
protein_idx = torch.arange(num_protein)

# local Cartesian products
local_dd = torch.cartesian_prod(drug_idx, disease_idx)   # [num_drug*num_disease, 2]
local_dp = torch.cartesian_prod(disease_idx, protein_idx)
local_dr = torch.cartesian_prod(drug_idx, protein_idx)

# convert to global IDs
global_dd = local_dd.clone()
global_dd[:, 0] += offsets["drug"]
global_dd[:, 1] += offsets["disease"]

global_dp = local_dp.clone()
global_dp[:, 0] += offsets["disease"]
global_dp[:, 1] += offsets["protein"]

global_dr = local_dr.clone()
global_dr[:, 0] += offsets["drug"]
global_dr[:, 1] += offsets["protein"]

# Threshold & mask
threshold = 0.5

mask_dd = drug_disease_scores.squeeze(-1)  > threshold  # [N_dd]
mask_dp = disease_protein_scores.squeeze(-1) > threshold
mask_dr = drug_protein_scores.squeeze(-1)  > threshold

# select only predicted edges
sel_dd_pairs  = global_dd[mask_dd]    # [M_dd, 2]
sel_dd_scores = drug_disease_scores.squeeze(-1)[mask_dd]

sel_dp_pairs  = global_dp[mask_dp]
sel_dp_scores = disease_protein_scores.squeeze(-1)[mask_dp]

sel_dr_pairs  = global_dr[mask_dr]
sel_dr_scores = drug_protein_scores.squeeze(-1)[mask_dr]

# Build & save DataFrames

# Drug–Disease
df_dd = pd.DataFrame({
    'source': [global_id_to_name(int(u)) for u in sel_dd_pairs[:,0].tolist()],
    'target': [global_id_to_name(int(v)) for v in sel_dd_pairs[:,1].tolist()],
    'score' : sel_dd_scores.tolist()
})
df_dd.to_csv('candidate_drug_ranking/predicted_knowledge_graph/predicted_drug_disease.csv', index=False)

# Disease–Protein
df_dp = pd.DataFrame({
    'source': [global_id_to_name(int(u)) for u in sel_dp_pairs[:,0].tolist()],
    'target': [global_id_to_name(int(v)) for v in sel_dp_pairs[:,1].tolist()],
    'score' : sel_dp_scores.tolist()
})
df_dp.to_csv('candidate_drug_ranking/predicted_knowledge_graph/predicted_disease_protein.csv', index=False)

# Drug–Protein
df_dr = pd.DataFrame({
    'source': [global_id_to_name(int(u)) for u in sel_dr_pairs[:,0].tolist()],
    'target': [global_id_to_name(int(v)) for v in sel_dr_pairs[:,1].tolist()],
    'score' : sel_dr_scores.tolist()
})
df_dr.to_csv('candidate_drug_ranking/predicted_knowledge_graph/predicted_drug_protein.csv', index=False)

print("Saved:")
print(f" • predicted_drug_disease.csv    ({len(df_dd)} edges)")
print(f" • predicted_disease_protein.csv ({len(df_dp)} edges)")
print(f" • predicted_drug_protein.csv    ({len(df_dr)} edges)")


#############-------------- Ranking the candidate drugs for each disease --------------##############
predicted_folder      = "candidate_drug_ranking/predicted_knowledge_graph/"
original_folder       = "candidate_drug_ranking/original_knowledge_graph/"
disease_of_interest   = "D000544"
# 'D000544' : 'Alzheimer Disease',
# 'D015140' : 'Vascular Dementia',
# 'D010300' : 'Parkinson Disease Dementia',
# 'D057180' : 'Frontotemporal Dementia',
# 'D006816' : 'Huntington Disease',
# 'D006850' : 'Normal Pressure Hydrocephalus',
# 'D020961' : 'Lewy Body Disease'

# Parameters
top_n                 = 20   # for run 1
top_k_for_overlap     = 750  # for run 2, initial pool size
top_n_overlap         = 20   # for run 2, final pick size
use_predicted_protein = True  # True -> use predicted drug_protein & disease_protein


# Read predicted and original drug_protein & disease_protein edges
dp_df   = pd.read_csv(os.path.join(predicted_folder, "drug_protein.csv"))
disp_df = pd.read_csv(os.path.join(predicted_folder, "disease_protein.csv"))

or_dp_df   = pd.read_csv(os.path.join(original_folder, "drug_protein.csv"))
or_disp_df = pd.read_csv(os.path.join(original_folder, "disease_protein.csv"))


#Load predicted drug–disease scores
dd_pred = pd.read_csv(os.path.join(predicted_folder, "drug_disease.csv"))

#Load known (original) drug–disease edges for novelty check
dd_known = pd.read_csv(os.path.join(original_folder, "drug_disease.csv"))[['Drug','Disease']]
known_edges = set(dd_known.itertuples(index=False, name=None))
#    known_edges contains tuples like ('DrugX','Alzheimer’s disease')

#Load drug–protein & disease–protein (original or predicted)
if use_predicted_protein:
    dp_df   = pd.read_csv(os.path.join(predicted_folder, "drug_protein.csv"))[['Drug','Protein']]
    disp_df = pd.read_csv(os.path.join(predicted_folder, "disease_protein.csv"))[['Disease','Protein']]
else:
    dp_df   = pd.read_csv(os.path.join(original_folder, "drug_protein.csv"))
    disp_df = pd.read_csv(os.path.join(original_folder, "disease_protein.csv"))

#Precompute the disease’s protein set
disease_proteins = set(
    disp_df.loc[disp_df["Disease"] == disease_of_interest, "Protein"]
)

#Helper to build overlap DataFrame given (drug, score) pairs
def build_overlap_df(drug_score_list):
    recs = []
    for drug, score in drug_score_list:
        prots = set(dp_df.loc[dp_df["Drug"] == drug, "Protein"])
        ov    = prots & disease_proteins
        uni   = prots | disease_proteins
        recs.append({
            "Drug":         drug,
            "score":        score,
            "n_drug_prot":  len(prots),
            "n_dis_prot":   len(disease_proteins),
            "n_overlap":    len(ov),
            "jaccard":      len(ov) / len(uni) if uni else 0.0,
            "overlap_list": sorted(ov),
        })
    return pd.DataFrame(recs)

#Run 1: top_n by score
top_scores = (
    dd_pred[dd_pred["Disease"] == disease_of_interest]
    .sort_values("score", ascending=False)
    .head(top_n)[["Drug","score"]]
    .values.tolist()
)
res1_df = build_overlap_df(top_scores)

#Annotate novelty vs. known edges
res1_df["is_known"] = res1_df["Drug"].apply(
    lambda d: (d, disease_of_interest) in known_edges
)
res1_df["is_novel"] = ~res1_df["is_known"]

print(f"Run 1: Top {top_n} drugs by score for '{disease_of_interest}'\n")
display(res1_df)


#Run 2: select by overlap within top_k_for_overlap
#take top_k_for_overlap by score
top_k = (
    dd_pred[dd_pred["Disease"] == disease_of_interest]
    .sort_values("score", ascending=False)
    .head(top_k_for_overlap)[["Drug","score"]]
    .values.tolist()
)
res_k_df = build_overlap_df(top_k)

#pick top_n_overlap by n_overlap desc, then score desc
res2_df = (
    res_k_df
    .sort_values(["n_overlap","score"], ascending=[False,False])
    .head(top_n_overlap)
    .reset_index(drop=True)
)

#Annotate novelty vs. known edges
res2_df["is_known"] = res2_df["Drug"].apply(
    lambda d: (d, disease_of_interest) in known_edges
)
res2_df["is_novel"] = ~res2_df["is_known"]

print(f"\nRun 2: Among top {top_k_for_overlap} by score, pick {top_n_overlap} by overlap →\n")
display(res2_df)




