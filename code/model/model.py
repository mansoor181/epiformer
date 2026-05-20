import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv

from CrossAttention import CrossAttentionBlock

class GraphLayer(nn.Module):
    """
    Single GNN layer: GCN/GAT/GIN + optional residual.
    """
    def __init__(self, in_dim, out_dim, model_type="GCN", use_residual=False):
        super().__init__()
        self.use_residual = use_residual
        # select convolution type
        if model_type == "GCN":
            self.conv = GCNConv(in_dim, out_dim)
        elif model_type == "GAT":
            self.conv = GATConv(in_dim, out_dim)
        elif model_type == "GIN":
            mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            )
            self.conv = GINConv(mlp)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        # residual projection if dims differ
        if use_residual and in_dim != out_dim:
            self.res_proj = nn.Linear(in_dim, out_dim)
        else:
            self.res_proj = None

    def forward(self, x, edge_index):
        out = self.conv(x, edge_index)
        if self.use_residual:
            res = x if self.res_proj is None else self.res_proj(x)
            out = out + res
        return F.relu(out)

class GraphEncoder(nn.Module):
    """
    Dynamic GNN encoder: uses `hidden_dims` list for layering.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dims: list,
                 output_dim: int,
                 model_type: str = "GCN",
                 use_residual: bool = False,
                 dropout: float = 0.5):
        super().__init__()
        # build dims from input -> hidden_dims -> output
        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList([
            GraphLayer(dims[i], dims[i+1], model_type, use_residual)
            for i in range(len(dims)-1)
        ])
        self.dropout = dropout

    def forward(self, x, edge_index):
        # apply all but last layer with dropout
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # last layer, no dropout
        return self.layers[-1](x, edge_index)

"""
### Runtime impact
* **Dot** fastest, O(NM) mat‑mul.  
* **MLP** same memory as dot; a single extra FC pass.  
* **Attention** adds multi‑head projection, slight memory overhead but still O(NM).
"""

# ─────────────────────────────────────────────────────────────
# 1) Dot‑product decoder (formerly BipartiteDecoder)
# ─────────────────────────────────────────────────────────────
class DotDecoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.interaction = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim)) # W is kxk
        nn.init.xavier_uniform_(self.interaction)

    def forward(self, ag_embed, ab_embed):
        # logits = ag_embed @ self.interaction @ ab_embed.t() # X_g @ W @ X_b^T
        logits = ag_embed @ ab_embed.t() # X_g @ X_b^T
        return torch.sigmoid(logits) # [Ng,Nb] matrix of probabilities where x_ij represents the 
                # probability of an edge between ag node i and ab node j (between 0 and 1)

# ─────────────────────────────────────────────────────────────
# 2) MLP‑based pairwise decoder
# ─────────────────────────────────────────────────────────────
class MLPDecoder(nn.Module):
    """Score each Ag–Ab pair by concatenating their embeddings into a tiny MLP."""
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, ag_embed, ab_embed):
        N, D = ag_embed.shape
        M    = ab_embed.shape[0]
        # expand to pairwise
        ag_exp = ag_embed.unsqueeze(1).expand(N, M, D)
        ab_exp = ab_embed.unsqueeze(0).expand(N, M, D)
        pair   = torch.cat([ag_exp, ab_exp], dim=-1)      # [N,M,2D]
        logits = self.net(pair).squeeze(-1)               # [N,M]
        return torch.sigmoid(logits)

# ─────────────────────────────────────────────────────────────
# 3) Cross‑Attention decoder
# ─────────────────────────────────────────────────────────────

class AttentionDecoder(nn.Module):
    """Cross‑attend each antigen node over all antibody nodes (multi‑head)."""
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = CrossAttentionBlock(dim, num_heads)

    def forward(self, ag_embed, ab_embed):
        # returns [N,M] attention‑score matrix
        return self.attn(ag_embed, ab_embed)
    

class M3EPI(nn.Module):
    def __init__(self, config):
        super().__init__()
        mt = config.model.name
        ur = config.model.use_residual
        dr = config.model.dropout
        print(f"[INFO] Model={mt} | residual={ur} | dropout={dr}")

        # two separate encoders
        self.ag_encoder = GraphEncoder(
            input_dim=config.model.encoder.antigen.input_dim,
            hidden_dims=config.model.encoder.antigen.hidden_dims,
            output_dim=config.model.encoder.antigen.output_dim,
            model_type=mt, use_residual=ur, dropout=dr
        )
        self.ab_encoder = GraphEncoder(
            input_dim=config.model.encoder.antibody.input_dim,
            hidden_dims=config.model.encoder.antibody.hidden_dims,
            output_dim=config.model.encoder.antibody.output_dim,
            model_type=mt, use_residual=ur, dropout=dr
        )

        # ─────────────────────────────────────────────────────
        # decoder dispatch
        dec_cfg = config.model.decoder
        dec_type = dec_cfg.type.lower()
        dim      = dec_cfg.interaction_dim

        if dec_type == "dot":
            self.decoder = DotDecoder(dim)
        elif dec_type == "mlp":
            self.decoder = MLPDecoder(dim, dec_cfg.mlp_hidden)
        elif dec_type == "attention":
            self.decoder = AttentionDecoder(dim, dec_cfg.heads)
        else:
            raise ValueError(f"Unknown decoder type: {dec_type}")

        self.threshold = dec_cfg.threshold
    # ─────────────────────────────────────────────────────────────
    def forward(self, batch):
        ag_x, ag_e = batch['x_g'], batch['edge_index_g']
        ab_x, ab_e = batch['x_b'], batch['edge_index_b']

        ag_emb = self.ag_encoder(ag_x, ag_e)
        ab_emb = self.ab_encoder(ab_x, ab_e)
        ip     = self.decoder(ag_emb, ab_emb)       # [N,M]
        # print(ip.shape, ag_emb.shape, ab_emb.shape)
        """
        TODO: 
        -  take the row-wise sum of probabilities rather than max probability
        """
        # epi_prob = torch.sigmoid(ip.sum(dim=1))   # sum of row-wise probabilties
        epi_prob = ip.max(dim=1).values   # pick the max row-wise probabilty

        return {
            'ag_embed': ag_emb,
            'ab_embed': ab_emb,
            'interaction_probs': ip,
            'epitope_prob': epi_prob
        }





