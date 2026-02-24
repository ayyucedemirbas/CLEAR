import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv
from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import LinkNeighborLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
import time


class ADRD_LinkPredictor(nn.Module):
    def __init__(self,
                 in_dims,
                 hidden_dim,
                 gat_heads,
                 fusion_heads,
                 dropout,
                 beta):
        super().__init__()
        self.beta = beta
        # A) Linear transforms
        self.type_linears = nn.ModuleDict({
            n: nn.Linear(in_dims[n], hidden_dim) for n in in_dims
        })
        self.shared_lin = nn.Linear(hidden_dim, hidden_dim)

        # All 6 relation keys
        self.rel_keys = [
            'drug-sim-drug', 'disease-sim-disease', 'protein-sim-protein',
            'drug-interacts-disease','disease-assoc-protein','drug-binds-protein'
        ]

        # B) 2-layer GATs
        self.gat1 = nn.ModuleDict({
            k: GATConv(hidden_dim, hidden_dim//gat_heads, heads=gat_heads, dropout = dropout,  concat=True)
            for k in self.rel_keys
        })
        self.gat2 = nn.ModuleDict({
            k: GATConv(hidden_dim, hidden_dim, heads=gat_heads, dropout = dropout, concat=False)
            for k in self.rel_keys
        })

        # C) Fusion self-attention (3→1)
        self.fusion_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=fusion_heads,
            batch_first=False,
            dropout=dropout
        )

        # D) Shared MLP
        self.link_mlp = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),  # no inplace
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1)
        )

    def compute_full_graph_embeddings(self, data):

        # A) Raw feature → shared space
        x = {}
        for ntype in data.node_types:
            h = F.relu(self.type_linears[ntype](data[ntype].x))
            x[ntype] = F.relu(self.shared_lin(h))

        # Prepare container for each node‐type’s list of [N_ntype×H] embeddings
        embs = {ntype: [] for ntype in data.node_types}

        # B) Run each of the 6 GATs
        for key in self.rel_keys:
            u_t, rel, v_t = key.split('-')
            c1, c2 = self.gat1[key], self.gat2[key]
            ei = data[u_t, rel, v_t].edge_index.to(x[u_t].device)

            if u_t == v_t:
                # similarity (homogeneous)
                h1 = F.elu(c1(x[u_t], ei))
                h2 = c2(h1, ei)                  # [N_u, H]
                embs[u_t].append(h2)

            else:
                # bipartite: ensure correct (src,dst) orientation
                N_u = x[u_t].size(0)
                N_v = x[v_t].size(0)
                max_src, max_dst = ei[0].max().item(), ei[1].max().item()
                if max_src >= N_u or max_dst >= N_v:
                    # flip rows if they got swapped at load time
                    ei = ei.flip(0).contiguous()

                # build the undirected view exactly once
                h_u, h_v = x[u_t], x[v_t]
                h_comb   = torch.cat([h_u, h_v], dim=0)  # [N_u+N_v, H]

                # forward edges map dst→[N_u..]
                ei_fwd = ei.clone()
                ei_fwd[1] += N_u

                # reverse edges map src→[N_u..]
                ei_rev = ei.flip(0).clone()
                ei_rev[0] += N_u

                ei_bid = torch.cat([ei_fwd, ei_rev], dim=1).to(h_comb.device)

                h1 = F.elu(c1(h_comb, ei_bid))
                h2 = c2(h1,    ei_bid)               # [N_u+N_v, H]

                # split back out
                embs[u_t].append(h2[:N_u])           # [N_u, H]
                embs[v_t].append(h2[N_u:])           # [N_v, H]

        # C) Fuse each node‐type’s 3 embeddings → 1
        H_fused = {}
        for ntype, lst in embs.items():
            # lst is 3 × [N_ntype, H]
            stacked = torch.stack(lst, dim=0)        # [3, N_ntype, H]
            fused, _ = self.fusion_attn(stacked, stacked, stacked)
            H_fused[ntype] = fused.mean(dim=0)       # [N_ntype, H]

        # —— NEW PART —— Combine into one giant embedding + record offsets
        H_drug = H_fused['drug']                    # [N_drug, H]
        H_dis  = H_fused['disease']                 # [N_disease, H]
        H_prot = H_fused['protein']                 # [N_protein, H]

        offsets = {
            'drug':    0,
            'disease': H_drug.size(0),
            'protein': H_drug.size(0) + H_dis.size(0)
        }

        H_all = torch.cat([H_drug, H_dis, H_prot], dim=0)  # [N_drug+N_disease+N_protein, H]

        return H_all, offsets


    def forward(self, H_all, offsets, u_types, v_types, u_idx, v_idx):
        # Build per‐sample offsets
        off_u = torch.tensor([offsets[ut] for ut in u_types], dtype=torch.long, device=H_all.device)
        off_v = torch.tensor([offsets[vt] for vt in v_types], dtype=torch.long, device=H_all.device)

        # Compute global indices
        u_glob = u_idx + off_u
        v_glob = v_idx + off_v

        # Gather embeddings in one go
        hu = H_all[u_glob]   # [B, H]
        hv = H_all[v_glob]   # [B, H]

        x  = torch.cat([hu, hv], dim=-1)  # [B, 2H]
        return self.link_mlp(x).squeeze(-1)


    def compute_loss(self, logits, labels, rel_types):
        # E) weighted BCE
        sim = {'drug-sim-drug','disease-sim-disease','protein-sim-protein'}
        is_sim = torch.tensor([rt in sim for rt in rel_types], device=logits.device)
        is_bip = ~is_sim
        fn = F.binary_cross_entropy_with_logits
        l_sim = fn(logits[is_sim], labels[is_sim]) if is_sim.any() else 0.0
        l_bip = fn(logits[is_bip], labels[is_bip]) if is_bip.any() else 0.0

        return self.beta * l_bip + (1-self.beta) * l_sim

#-------------- Edge‐batch Dataset & DataLoader --------------#
class FullGraphLinkDataset(Dataset):
    def __init__(self, folds, fold_idx, split = "train"):
        self.entries = []
        for rel_key, fl in folds.items():
            pos = fl[fold_idx][f"{split}_pos"].t().tolist()
            neg = fl[fold_idx][f"{split}_neg"].t().tolist()
            u_t, _, v_t = rel_key.split('-')
            for u,v in pos: self.entries.append((rel_key, u_t, v_t, u, v, 1.0))
            for u,v in neg: self.entries.append((rel_key, u_t, v_t, u, v, 0.0))

    def __len__(self): return len(self.entries)
    def __getitem__(self, i):
        rel_key, u_t, v_t, u, v, lbl = self.entries[i]
        return rel_key, u_t, v_t, torch.tensor(u), torch.tensor(v), torch.tensor(lbl)