import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv
from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import LinkNeighborLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torch.utils.checkpoint import checkpoint as grad_checkpoint
import time


class ADRD_LinkPredictor(nn.Module):
    def __init__(self,
                 in_dims,
                 hidden_dim,       # T4-safe default: 512  (was 1024)
                 gat_heads,        # T4-safe default: 4    (was 8)
                 fusion_heads,     # T4-safe default: 4    (was 8)
                 dropout,
                 beta,
                 use_checkpoint: bool = True):

        super().__init__()
        self.beta = beta
        self.use_checkpoint = use_checkpoint

        # Validate heads divides hidden_dim
        assert hidden_dim % gat_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by gat_heads ({gat_heads})"
        )
        assert hidden_dim % fusion_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by fusion_heads ({fusion_heads})"
        )

        self.type_linears = nn.ModuleDict({
            n: nn.Linear(in_dims[n], hidden_dim) for n in in_dims
        })
        self.shared_lin = nn.Linear(hidden_dim, hidden_dim)

        self.rel_keys = [
            'drug-sim-drug', 'disease-sim-disease', 'protein-sim-protein',
            'drug-interacts-disease', 'disease-assoc-protein', 'drug-binds-protein'
        ]

        # out_channels per head = hidden_dim // gat_heads
        # After concat=True in gat1: output width = gat_heads × out_channels = hidden_dim ✓
        gat_out = hidden_dim // gat_heads          # e.g. 512//4 = 128
        self.gat1 = nn.ModuleDict({
            k: GATConv(hidden_dim, gat_out,
                       heads=gat_heads, dropout=dropout, concat=True,
                       add_self_loops=True)
            for k in self.rel_keys
        })
        self.gat2 = nn.ModuleDict({
            k: GATConv(hidden_dim, hidden_dim,
                       heads=gat_heads, dropout=dropout, concat=False,
                       add_self_loops=True)
            for k in self.rel_keys
        })

        # Fusion self-attention (3 -> 1)
        self.fusion_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=fusion_heads,
            batch_first=False,
            dropout=dropout
        )

        # Deeper shared MLP (compensates for reduced hidden_dim)
        # Extra hidden layer adds capacity at very low memory cost (no edges).
        mlp_mid = hidden_dim
        self.link_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, mlp_mid),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(mlp_mid, mlp_mid // 2),   # extra layer vs. original
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(mlp_mid // 2, 1)
        )

    def _gat_homo(self, c1, c2, h, ei):
        """Two-layer GAT for homogeneous (similarity) edges."""
        h1 = F.elu(c1(h, ei))
        return c2(h1, ei)

    def _gat_bip(self, c1, c2, h_comb, ei_bid):
        """Two-layer GAT for bipartite (bidirectional) edges."""
        h1 = F.elu(c1(h_comb, ei_bid))
        return c2(h1, ei_bid)


    def compute_full_graph_embeddings(self, data):
        # Raw feature -> shared space
        x = {}
        for ntype in data.node_types:
            h = F.relu(self.type_linears[ntype](data[ntype].x))
            x[ntype] = F.relu(self.shared_lin(h))

        embs = {ntype: [] for ntype in data.node_types}

        # Run each of the 6 GATs
        for key in self.rel_keys:
            u_t, rel, v_t = key.split('-')
            c1, c2 = self.gat1[key], self.gat2[key]
            ei = data[u_t, rel, v_t].edge_index.to(x[u_t].device)

            if u_t == v_t:
                if self.use_checkpoint and self.training:
                    h2 = grad_checkpoint(self._gat_homo, c1, c2, x[u_t], ei,
                                         use_reentrant=False)
                else:
                    h2 = self._gat_homo(c1, c2, x[u_t], ei)
                embs[u_t].append(h2)

            else:
                # Bipartite
                N_u = x[u_t].size(0)
                N_v = x[v_t].size(0)
                max_src = ei[0].max().item()
                max_dst = ei[1].max().item()
                if max_src >= N_u or max_dst >= N_v:
                    ei = ei.flip(0).contiguous()

                h_u, h_v = x[u_t], x[v_t]
                h_comb = torch.cat([h_u, h_v], dim=0)      # [N_u+N_v, H]

                # Build bidirectional edge index
                ei_fwd = ei.clone()
                ei_fwd[1] += N_u
                ei_rev = ei.flip(0).clone()
                ei_rev[0] += N_u
                ei_bid = torch.cat([ei_fwd, ei_rev], dim=1).to(h_comb.device)

                if self.use_checkpoint and self.training:
                    h2 = grad_checkpoint(self._gat_bip, c1, c2, h_comb, ei_bid,
                                         use_reentrant=False)
                else:
                    h2 = self._gat_bip(c1, c2, h_comb, ei_bid)

                # FIX: .clone() so deleting h2 frees the underlying storage.
                # Without .clone(), slices are *views* that keep h2's data
                # alive until the list is later consumed — a hidden memory leak.
                embs[u_t].append(h2[:N_u].clone())
                embs[v_t].append(h2[N_u:].clone())

                # Now h2, h_comb, ei_bid can all be freed immediately.
                del h_comb, ei_bid, h2
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Fuse each node-type's 3 embeddings -> 1
        # FIX: pop each type's embedding list before stacking, then delete it
        # immediately after torch.stack().  This means only ONE type's source
        # tensors + stacked tensor are alive at a time instead of all 9.
        H_fused_dict = {}
        for ntype in list(embs.keys()):
            lst = embs.pop(ntype)               # remove from dict  ← FIX
            stacked = torch.stack(lst, dim=0)   # [3, N_ntype, H]
            del lst                             # free 3 source tensors ← FIX
            fused, _ = self.fusion_attn(stacked, stacked, stacked)
            # .contiguous() ensures the output owns its storage (no alias chains)
            H_fused_dict[ntype] = fused.mean(dim=0).contiguous()
            del stacked, fused
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        H_drug = H_fused_dict['drug']
        H_dis  = H_fused_dict['disease']
        H_prot = H_fused_dict['protein']

        offsets = {
            'drug':    0,
            'disease': H_drug.size(0),
            'protein': H_drug.size(0) + H_dis.size(0)
        }

        H_all = torch.cat([H_drug, H_dis, H_prot], dim=0)  # [N_total, H]
        return H_all, offsets


    def forward(self, H_all, offsets, u_types, v_types, u_idx, v_idx):
        off_u = torch.tensor([offsets[ut] for ut in u_types],
                             dtype=torch.long, device=H_all.device)
        off_v = torch.tensor([offsets[vt] for vt in v_types],
                             dtype=torch.long, device=H_all.device)

        u_glob = u_idx + off_u
        v_glob = v_idx + off_v

        hu = H_all[u_glob]
        hv = H_all[v_glob]

        x = torch.cat([hu, hv], dim=-1)      # [B, 2H]
        return self.link_mlp(x).squeeze(-1)


    def compute_loss(self, logits, labels, rel_types):
        sim = {'drug-sim-drug', 'disease-sim-disease', 'protein-sim-protein'}
        is_sim = torch.tensor([rt in sim for rt in rel_types],
                              device=logits.device)
        is_bip = ~is_sim
        fn = F.binary_cross_entropy_with_logits
        l_sim = fn(logits[is_sim], labels[is_sim]) if is_sim.any() else 0.0
        l_bip = fn(logits[is_bip], labels[is_bip]) if is_bip.any() else 0.0
        return self.beta * l_bip + (1 - self.beta) * l_sim


class FullGraphLinkDataset(Dataset):
    def __init__(self, folds, fold_idx, split="train"):
        self.entries = []
        for rel_key, fl in folds.items():
            pos = fl[fold_idx][f"{split}_pos"].t().tolist()
            neg = fl[fold_idx][f"{split}_neg"].t().tolist()
            u_t, _, v_t = rel_key.split('-')
            for u, v in pos:
                self.entries.append((rel_key, u_t, v_t, u, v, 1.0))
            for u, v in neg:
                self.entries.append((rel_key, u_t, v_t, u, v, 0.0))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        rel_key, u_t, v_t, u, v, lbl = self.entries[i]
        return rel_key, u_t, v_t, torch.tensor(u), torch.tensor(v), torch.tensor(lbl)
