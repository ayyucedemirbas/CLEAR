import os
import torch
import pickle
import pandas as pd

from model import ADRD_LinkPredictor
from utils import predict_and_filter_streaming

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

Dataset_name = "ADRD"
threshold    = 0.5
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SRC_BATCH = 32     # (was 64)
DST_BATCH = 1024   # (was 2048)


print("Loading node IDs...")
with open(f'{Dataset_name}/input_network/node_ids/drugs.pkl',    'rb') as f:
    drugs    = pickle.load(f)
with open(f'{Dataset_name}/input_network/node_ids/diseases.pkl', 'rb') as f:
    diseases = pickle.load(f)
with open(f'{Dataset_name}/input_network/node_ids/proteins.pkl', 'rb') as f:
    proteins = pickle.load(f)

num_drug    = len(drugs)
num_disease = len(diseases)
num_protein = len(proteins)

offsets = {
    "drug":    0,
    "disease": num_drug,
    "protein": num_drug + num_disease
}
total_nodes = num_drug + num_disease + num_protein
print(f"Nodes: drugs={num_drug}, diseases={num_disease}, "
      f"proteins={num_protein}, total={total_nodes}")


drug_global_start    = offsets["drug"]
drug_global_end      = offsets["disease"]
disease_global_start = offsets["disease"]
disease_global_end   = offsets["protein"]
protein_global_start = offsets["protein"]
protein_global_end   = total_nodes


model = ADRD_LinkPredictor(
    in_dims      = {'drug': 768, 'disease': 768, 'protein': 1280},
    hidden_dim   = 512,    # must match the value used during training
    gat_heads    = 4,  # TODO: Try to increase this
    fusion_heads = 4,
    beta         = 0.75,
    dropout      = 0.35,
    use_checkpoint = False   # not needed for inference
)
model.load_state_dict(
    torch.load(f"{Dataset_name}/intermediate_data_save/trained_model_weights.pt",
               map_location="cpu")
)
model.eval()
# Only the MLP sub-module is needed at inference time; keep GAT layers on CPU.
model.link_mlp = model.link_mlp.to(device)

print("Loading CLEAR node embeddings (CPU)...")
H_fused = torch.load(
    f"{Dataset_name}/intermediate_data_save/CLEAR_node_embeddings.pt",
    map_location="cpu"
)
print(f"H_fused shape: {H_fused.shape}  "
      f"({H_fused.numel() * 4 / 1e9:.2f} GB fp32 on CPU)")


def global_id_to_name(node_id: int) -> str:
    if node_id < offsets["disease"]:
        return drugs[node_id]
    elif node_id < offsets["protein"]:
        return diseases[node_id - offsets["disease"]]
    else:
        return proteins[node_id - offsets["protein"]]

os.makedirs("candidate_drug_ranking/predicted_knowledge_graph", exist_ok=True)

# Drug–Disease
print(f"\nStreaming Drug–Disease pairs "
      f"({num_drug} × {num_disease} = {num_drug*num_disease:,})...")
dd_pairs, dd_scores = predict_and_filter_streaming(
    H_fused, model,
    src_global_start = drug_global_start,
    src_global_end   = drug_global_end,
    dst_global_start = disease_global_start,
    dst_global_end   = disease_global_end,
    device           = device,
    threshold        = threshold,
    src_batch        = SRC_BATCH,
    dst_batch        = DST_BATCH,
)
print(f"  -> {len(dd_scores):,} drug–disease edges above threshold")

df_dd = pd.DataFrame({
    'source': [global_id_to_name(int(u)) for u in dd_pairs[:, 0].tolist()],
    'target': [global_id_to_name(int(v)) for v in dd_pairs[:, 1].tolist()],
    'score' : dd_scores.tolist(),
})
df_dd.to_csv(
    'candidate_drug_ranking/predicted_knowledge_graph/predicted_drug_disease.csv',
    index=False)
print(f"  Saved predicted_drug_disease.csv  ({len(df_dd)} edges)")

del dd_pairs, dd_scores
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Disease - Protein
print(f"\nStreaming Disease–Protein pairs "
      f"({num_disease} × {num_protein} = {num_disease*num_protein:,})...")
dp_pairs, dp_scores = predict_and_filter_streaming(
    H_fused, model,
    src_global_start = disease_global_start,
    src_global_end   = disease_global_end,
    dst_global_start = protein_global_start,
    dst_global_end   = protein_global_end,
    device           = device,
    threshold        = threshold,
    src_batch        = SRC_BATCH,
    dst_batch        = DST_BATCH,
)
print(f"  -> {len(dp_scores):,} disease–protein edges above threshold")

df_dp = pd.DataFrame({
    'source': [global_id_to_name(int(u)) for u in dp_pairs[:, 0].tolist()],
    'target': [global_id_to_name(int(v)) for v in dp_pairs[:, 1].tolist()],
    'score' : dp_scores.tolist(),
})
df_dp.to_csv(
    'candidate_drug_ranking/predicted_knowledge_graph/predicted_disease_protein.csv',
    index=False)
print(f"  Saved predicted_disease_protein.csv  ({len(df_dp)} edges)")

del dp_pairs, dp_scores
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Drug–Protein
print(f"\nStreaming Drug–Protein pairs "
      f"({num_drug} × {num_protein} = {num_drug*num_protein:,})  "
      f"← largest relation, may take a few minutes...")
dr_pairs, dr_scores = predict_and_filter_streaming(
    H_fused, model,
    src_global_start = drug_global_start,
    src_global_end   = drug_global_end,
    dst_global_start = protein_global_start,
    dst_global_end   = protein_global_end,
    device           = device,
    threshold        = threshold,
    src_batch        = SRC_BATCH,
    dst_batch        = DST_BATCH,
)
print(f"  -> {len(dr_scores):,} drug–protein edges above threshold")

df_dr = pd.DataFrame({
    'source': [global_id_to_name(int(u)) for u in dr_pairs[:, 0].tolist()],
    'target': [global_id_to_name(int(v)) for v in dr_pairs[:, 1].tolist()],
    'score' : dr_scores.tolist(),
})
df_dr.to_csv(
    'candidate_drug_ranking/predicted_knowledge_graph/predicted_drug_protein.csv',
    index=False)
print(f"  Saved predicted_drug_protein.csv  ({len(df_dr)} edges)")

del dr_pairs, dr_scores
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\nAll predicted bipartite edges saved.")

predicted_folder    = "candidate_drug_ranking/predicted_knowledge_graph/"
original_folder     = "candidate_drug_ranking/original_knowledge_graph/"
disease_of_interest = "D000544"

top_n                 = 20
top_k_for_overlap     = 750
top_n_overlap         = 20
use_predicted_protein = True

dd_pred  = pd.read_csv(os.path.join(predicted_folder, "drug_disease.csv"))
dd_known = pd.read_csv(os.path.join(original_folder,  "drug_disease.csv"))[['Drug', 'Disease']]
known_edges = set(dd_known.itertuples(index=False, name=None))

if use_predicted_protein:
    dp_df   = pd.read_csv(os.path.join(predicted_folder, "drug_protein.csv"))[['Drug', 'Protein']]
    disp_df = pd.read_csv(os.path.join(predicted_folder, "disease_protein.csv"))[['Disease', 'Protein']]
else:
    dp_df   = pd.read_csv(os.path.join(original_folder, "drug_protein.csv"))
    disp_df = pd.read_csv(os.path.join(original_folder, "disease_protein.csv"))

disease_proteins = set(
    disp_df.loc[disp_df["Disease"] == disease_of_interest, "Protein"]
)


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


top_scores = (
    dd_pred[dd_pred["Disease"] == disease_of_interest]
    .sort_values("score", ascending=False)
    .head(top_n)[["Drug", "score"]]
    .values.tolist()
)
res1_df = build_overlap_df(top_scores)
res1_df["is_known"] = res1_df["Drug"].apply(lambda d: (d, disease_of_interest) in known_edges)
res1_df["is_novel"] = ~res1_df["is_known"]

print(f"Run 1: Top {top_n} drugs by score for '{disease_of_interest}'\n")
try:
    display(res1_df)
except NameError:
    print(res1_df.to_string())

top_k = (
    dd_pred[dd_pred["Disease"] == disease_of_interest]
    .sort_values("score", ascending=False)
    .head(top_k_for_overlap)[["Drug", "score"]]
    .values.tolist()
)
res_k_df = build_overlap_df(top_k)
res2_df  = (
    res_k_df
    .sort_values(["n_overlap", "score"], ascending=[False, False])
    .head(top_n_overlap)
    .reset_index(drop=True)
)
res2_df["is_known"] = res2_df["Drug"].apply(lambda d: (d, disease_of_interest) in known_edges)
res2_df["is_novel"] = ~res2_df["is_known"]

print(f"\nRun 2: Among top {top_k_for_overlap} by score, pick {top_n_overlap} by overlap\n")
try:
    display(res2_df)
except NameError:
    print(res2_df.to_string())
