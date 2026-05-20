"""
epiformerEncoder with interleaved ResMP and cross-chain attention
Simplified implementation: Cross-attention between existing ResMP layers
NeurIPS 2026 version: No PLM embeddings by default (use_plm=False)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple

from model.res_mp import ResMP as EGNN_ResMP

from model.res_mp_regnn import RelationAwareResMP as REGNN_ResMP

# Import vanilla GNN implementations
from model.vanilla_gnns import VanillaAtomMP, VanillaEdgeMP, VanillaResMP, create_vanilla_gnn


def get_activation(activation):
    """Simple activation function factory"""
    activations = {
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'silu': nn.SiLU(),
        'swish': nn.SiLU(),
        'leaky_relu': nn.LeakyReLU()
    }
    return activations.get(activation.lower(), nn.ReLU())


class CrossAttentionLayer(nn.Module):
    """Simple cross-attention layer between antigen and antibody residues"""

    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"

        # Query, Key, Value projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, query, key_value, attention_mask=None):
        """
        Args:
            query: [batch_size, seq_len_q, d_model]
            key_value: [batch_size, seq_len_kv, d_model]
            attention_mask: [seq_len_q, seq_len_kv] or None
                           True/1 for positions that should be attended to
                           False/0 for positions that should be masked out
        Returns:
            output: [batch_size, seq_len_q, d_model]
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_kv = key_value.shape[1]

        # Project to Q, K, V
        Q = self.q_proj(query)  # [batch, seq_q, d_model]
        K = self.k_proj(key_value)  # [batch, seq_kv, d_model]
        V = self.v_proj(key_value)  # [batch, seq_kv, d_model]

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len_q, self.n_heads, self.head_dim).transpose(1, 2)  # [batch, heads, seq_q, head_dim]
        K = K.view(batch_size, seq_len_kv, self.n_heads, self.head_dim).transpose(1, 2)  # [batch, heads, seq_kv, head_dim]
        V = V.view(batch_size, seq_len_kv, self.n_heads, self.head_dim).transpose(1, 2)  # [batch, heads, seq_kv, head_dim]

        # Attention computation
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [batch, heads, seq_q, seq_kv]

        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask to match attention dimensions [batch, heads, seq_q, seq_kv]
            mask = attention_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_q, seq_kv]
            mask = mask.expand(batch_size, self.n_heads, -1, -1)

            # Apply mask: set masked positions to large negative value
            scores = scores.masked_fill(~mask, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # [batch, heads, seq_q, head_dim]

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)

        # Final projection
        output = self.out_proj(attn_output)

        return output



class EpiformerBlock(nn.Module):
    """epiformer block with parallel ResMP and cross-attention + adaptive gating"""

    def __init__(self,
                 residue_dim,
                 residue_hidden_dim,
                 edge_dim,
                 num_relations,
                 n_heads=1,
                 dropout=0.1,
                 use_layer_norm=True,
                 ffn_expansion_factor=4,
                 activation='silu',
                ag_resmp_type = "egnn",
                ab_resmp_type = "egnn",
                update_coords=True,
                cross_attn_mode="bidirectional"
                ):
        super().__init__()
        self.cross_attn_mode = cross_attn_mode

        # Separate ResMP layers for antigen and antibody with resmp_type option
        if ag_resmp_type.lower() == 'egnn':
            self.ag_resmp = EGNN_ResMP(
                node_dim=residue_dim,
                edge_dim=edge_dim,
                hidden_dim=residue_hidden_dim,
                num_layers=1,  # Single layer per block for efficiency
                num_relations=num_relations,
                dropout=dropout,
                layer_norm=use_layer_norm,
                update_coords=update_coords
            )
        elif ag_resmp_type.lower() == 'regnn':
            self.ag_resmp = REGNN_ResMP(
                node_dim=residue_dim,
                edge_dim=edge_dim,
                hidden_dim=residue_hidden_dim,
                num_layers=1,
                num_relations=num_relations,
                dropout=dropout,
                layer_norm=use_layer_norm
            )
        elif ag_resmp_type.lower() in ['gcn', 'gat', 'gin', 'rgcn']:
            self.ag_resmp = VanillaResMP(
                gnn_type=ag_resmp_type.lower(),
                node_dim=residue_dim,
                hidden_dim=residue_hidden_dim,
                num_layers=1,
                num_relations=num_relations,
                dropout=dropout,
                use_layer_norm=use_layer_norm
            )
        else:
            raise ValueError(f"Unknown ResMP type for antigen: {ag_resmp_type}")

        if ab_resmp_type.lower() == 'egnn':
            self.ab_resmp = EGNN_ResMP(
                node_dim=residue_dim,
                edge_dim=edge_dim,
                hidden_dim=residue_hidden_dim,
                num_layers=1,
                num_relations=num_relations,
                dropout=dropout,
                layer_norm=use_layer_norm,
                update_coords=update_coords
            )
        elif ab_resmp_type.lower() == 'regnn':
            self.ab_resmp = REGNN_ResMP(
                node_dim=residue_dim,
                edge_dim=edge_dim,
                hidden_dim=residue_hidden_dim,
                num_layers=1,
                num_relations=num_relations,
                dropout=dropout,
                layer_norm=use_layer_norm
            )
        elif ab_resmp_type.lower() in ['gcn', 'gat', 'gin', 'rgcn']:
            self.ab_resmp = VanillaResMP(
                gnn_type=ab_resmp_type.lower(),
                node_dim=residue_dim,
                hidden_dim=residue_hidden_dim,
                num_layers=1,
                num_relations=num_relations,
                dropout=dropout,
                use_layer_norm=use_layer_norm
            )
        else:
            raise ValueError(f"Unknown ResMP type for antibody: {ab_resmp_type}")

        # Cross-attention layers
        self.ag_cross_attn = CrossAttentionLayer(residue_dim, n_heads, dropout)
        self.ab_cross_attn = CrossAttentionLayer(residue_dim, n_heads, dropout)

        # Feedforward Networks (position-wise MLPs after attention)
        ffn_dim = ffn_expansion_factor * residue_dim
        self.ag_ffn = nn.Sequential(
            nn.Linear(residue_dim, ffn_dim),
            get_activation(activation),
            nn.Linear(ffn_dim, residue_dim),
            nn.Dropout(dropout)
        )

        self.ab_ffn = nn.Sequential(
            nn.Linear(residue_dim, ffn_dim),
            get_activation(activation),
            nn.Linear(ffn_dim, residue_dim),
            nn.Dropout(dropout)
        )

        # Learnable scalar gates for cross-attention (h ← h + α·Attn(h))
        self.ag_alpha = nn.Parameter(torch.tensor(0.05))  # Small initial weight for stability
        self.ab_alpha = nn.Parameter(torch.tensor(0.05))


        # Minimal layer normalization
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.ln_ag_pre = nn.LayerNorm(residue_dim)
            self.ln_ab_pre = nn.LayerNorm(residue_dim)

            self.ln_ag_attn_post = nn.LayerNorm(residue_dim)
            self.ln_ab_attn_post = nn.LayerNorm(residue_dim)

            # Pre/Post norm for Feedforward Networks
            self.ln_ag_ffn_post = nn.LayerNorm(residue_dim)
            self.ln_ab_ffn_post = nn.LayerNorm(residue_dim)

        self.dropout = nn.Dropout(dropout)



    def create_batch_attention_mask(self, ag_batch, ab_batch):
        """Create attention mask for batch-aware cross-attention"""
        ag_expanded = ag_batch.unsqueeze(1)  # [n_ag, 1]
        ab_expanded = ab_batch.unsqueeze(0)  # [1, n_ab]
        mask = (ag_expanded == ab_expanded)  # [n_ag, n_ab]
        return mask

    def forward(self, ag_hetero_data, ab_hetero_data, ag_features, ab_features):
        """
        Forward pass with parallel ResMP and cross-attention + adaptive gating

        Args:
            ag_hetero_data: Antigen graph data
            ab_hetero_data: Antibody graph data
            ag_features: Antigen residue features [n_ag, d_model]
            ab_features: Antibody residue features [n_ab, d_model]

        Returns:
            ag_out: Updated antigen features [n_ag, d_model]
            ab_out: Updated antibody features [n_ab, d_model]
        """
        # Pre-normalization
        if self.use_layer_norm:
            ag_input = self.ln_ag_pre(ag_features)
            ab_input = self.ln_ab_pre(ab_features)
        else:
            ag_input = ag_features
            ab_input = ab_features

        # Update hetero_data with normalized features
        ag_hetero_data['ag_res'].x = ag_input
        ab_hetero_data['ab_res'].x = ab_input

        # Extract batch information for attention masking
        ag_batch = ag_hetero_data['ag_res'].batch
        ab_batch = ab_hetero_data['ab_res'].batch

        # Create batch-aware attention masks
        ag_to_ab_mask = self.create_batch_attention_mask(ag_batch, ab_batch)
        ab_to_ag_mask = ag_to_ab_mask.transpose(0, 1)

        # ============ Parallel Processing ============
        # ResMP step (geometric message passing)
        ag_resmp_out, _ = self.ag_resmp(ag_hetero_data, 'ag_res')
        ab_resmp_out, _ = self.ab_resmp(ab_hetero_data, 'ab_res')

        # Cross-attention step (learned interaction patterns)
        # Supports: "bidirectional" (default), "ag_only" (AG queries AB),
        #           "ab_only" (AB queries AG), "none" (skip cross-attention)
        ag_input_batched = ag_input.unsqueeze(0)  # [1, n_ag, d_model]
        ab_input_batched = ab_input.unsqueeze(0)  # [1, n_ab, d_model]

        if self.cross_attn_mode == "none":
            ag_attn_out = ag_input
            ab_attn_out = ab_input
        else:
            if self.cross_attn_mode in ("bidirectional", "ag_only"):
                ag_cross = self.ag_cross_attn(ag_input_batched, ab_input_batched, ag_to_ab_mask).squeeze(0)
                ag_attn_out = ag_input + self.dropout(ag_cross)
            else:
                ag_attn_out = ag_input

            if self.cross_attn_mode in ("bidirectional", "ab_only"):
                ab_cross = self.ab_cross_attn(ab_input_batched, ag_input_batched, ab_to_ag_mask).squeeze(0)
                ab_attn_out = ab_input + self.dropout(ab_cross)
            else:
                ab_attn_out = ab_input

        # Post-normalization
        if self.use_layer_norm:
            ag_attn_out = self.ln_ag_attn_post(ag_attn_out)
            ab_attn_out = self.ln_ab_attn_post(ab_attn_out)

        # Apply feedforward networks
        ag_ffn_out = self.ag_ffn(ag_attn_out)
        ab_ffn_out = self.ab_ffn(ab_attn_out)

        # FFN residual connections + dropout
        ag_final = ag_attn_out + self.dropout(ag_ffn_out)
        ab_final = ab_attn_out + self.dropout(ab_ffn_out)

        # Post-norm for feedforward
        if self.use_layer_norm:
            ag_final = self.ln_ag_ffn_post(ag_final)
            ab_final = self.ln_ab_ffn_post(ab_final)

        # ============ Lean Residual Gating ============
        ag_out = ag_input + ag_resmp_out + self.ag_alpha * ag_final
        ab_out = ab_input + ab_resmp_out + self.ab_alpha * ab_final

        return ag_out, ab_out


class GatedFeatureFusion(nn.Module):
    """
    Gated feature fusion with learnable weights for each feature type
    Implements: fused = Σ (gate_i * W_i * feature_i)
    """
    def __init__(self, input_dims, output_dim, hidden_dim=32, activation='relu'):
        """
        Args:
            input_dims: List of dimensions for each input feature
            output_dim: Dimension of output fused features
            hidden_dim: Hidden dimension for gate networks
            activation: Activation function name
        """
        super().__init__()
        self.num_features = len(input_dims)

        # Feature projection layers
        self.projections = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in input_dims
        ])

        # Gate networks - learns importance weights
        total_input_dim = sum(input_dims)
        self.gate_network = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            get_activation(activation),
            nn.Linear(hidden_dim, self.num_features),
            nn.Softmax(dim=-1)
        )

    def forward(self, features):
        """
        Args:
            features: List of feature tensors [geom, atom, edge, plm]

        Returns:
            fused: Weighted combination of features
        """
        # Concatenate features for gate input
        concat_features = torch.cat(features, dim=-1)

        # Compute gate weights (softmax over features)
        gates = self.gate_network(concat_features)

        # Project and weight features
        weighted_features = 0
        for i in range(self.num_features):
            projected = self.projections[i](features[i])
            weighted_features += gates[:, i].unsqueeze(1) * projected

        return weighted_features



class EpiformerEncoder(nn.Module):
    """
    epiformer-style encoder with interleaved ResMP and cross-chain attention
    Processes AG and AB jointly through multiple epiformer blocks

    NeurIPS 2026: use_plm=False by default (geometric features only)
    """

    def __init__(self,
                 # Basic parameters
                 residue_dim=128,
                 residue_hidden_dim=128,
                 residue_layers=4,
                 geo_dim=105,

                 edge_dim=100,
                 num_relations=4,

                 # PLM parameters
                 plm_dim=128,

                 # Attention parameters
                 n_heads=1,

                 # General parameters
                 dropout=0.1,
                 use_layer_norm=True,
                 activation='relu',

                 # Feature fusion parameters
                 ag_feature_fusion_type="concat",
                 ab_feature_fusion_type="gated",
                 ag_plm_in_dim=480,
                 ab_plm_in_dim=512,
                 ag_plm_type="esm2_35m",
                 ag_resmp_type="egnn",
                 ab_resmp_type="egnn",
                 # New parameters for NeurIPS 2026
                 use_plm=False,  # Default False: no PLM embeddings
                 update_coords=True,
                 cross_attn_mode="bidirectional",
                 feature_mask="none",
                ):

        super().__init__()

        self.residue_dim = residue_dim
        self.residue_layers = residue_layers
        self.use_plm = use_plm
        self.feature_mask = feature_mask  # "none", "aa_only", "geo_only"

        if ag_plm_type == "esm2_35m":
            ag_plm_in_dim = 480
        elif ag_plm_type == "esm2_650m":
            ag_plm_in_dim = 1280
        elif ag_plm_type == "esm2_3b":
            ag_plm_in_dim = 2560
        elif ag_plm_type == "esm3_small":
            ag_plm_in_dim = 1536
        else:
            ag_plm_in_dim = 480

        if self.use_plm:
            # PLM projections (separate for AG and AB due to different PLM dimensions)
            self.ag_plm_proj = nn.Linear(ag_plm_in_dim, plm_dim)  # ESM-2 for antigen
            self.ab_plm_proj = nn.Linear(ab_plm_in_dim, plm_dim)  # AntiBERTy for antibody

        # ==================== Feature Fusion ====================
        if self.use_plm:
            # geometric(105) + plm(128) = 233 -> residue_dim
            ag_fusion_input_dims = [geo_dim, plm_dim]
            ab_fusion_input_dims = [geo_dim, plm_dim]
            ag_concat_dim = geo_dim + plm_dim
            ab_concat_dim = geo_dim + plm_dim
        else:
            # geometric(105) only -> residue_dim
            ag_fusion_input_dims = [geo_dim]
            ab_fusion_input_dims = [geo_dim]
            ag_concat_dim = geo_dim
            ab_concat_dim = geo_dim

        # Feature fusion for antigen
        if ag_feature_fusion_type == "gated":
            self.ag_feature_fusion = GatedFeatureFusion(
                input_dims=ag_fusion_input_dims,
                output_dim=residue_dim,
                hidden_dim=residue_hidden_dim,
                activation=activation
            )
        else:  # concat
            self.ag_feature_fusion = nn.Sequential(
                nn.Linear(ag_concat_dim, residue_hidden_dim),
                get_activation(activation),
                nn.Dropout(dropout),
                nn.Linear(residue_hidden_dim, residue_dim)
            )

        # Feature fusion for antibody
        if ab_feature_fusion_type == "gated":
            self.ab_feature_fusion = GatedFeatureFusion(
                input_dims=ab_fusion_input_dims,
                output_dim=residue_dim,
                hidden_dim=residue_hidden_dim,
                activation=activation
            )
        else:  # concat
            self.ab_feature_fusion = nn.Sequential(
                nn.Linear(ab_concat_dim, residue_hidden_dim),
                get_activation(activation),
                nn.Dropout(dropout),
                nn.Linear(residue_hidden_dim, residue_dim)
            )


        # Simplified epiformer blocks with adaptive gating
        self.epiformer_blocks = nn.ModuleList([
            EpiformerBlock(
                residue_dim=residue_dim,
                residue_hidden_dim=residue_hidden_dim,
                edge_dim=edge_dim,
                num_relations=num_relations,
                n_heads=n_heads,
                dropout=dropout,
                use_layer_norm=use_layer_norm,
                activation=activation,
                ag_resmp_type=ag_resmp_type,
                ab_resmp_type=ab_resmp_type,
                update_coords=update_coords,
                cross_attn_mode=cross_attn_mode
            )
            for _ in range(residue_layers)
        ])

        # Final output normalization (before passing to the decoder)
        self.final_ln_ag = nn.LayerNorm(residue_dim)
        self.final_ln_ab = nn.LayerNorm(residue_dim)

    def forward(self, hetero_data, ag_chain_type='ag', ab_chain_type='ab'):
        """
        Joint forward pass for antigen and antibody

        Args:
            hetero_data: Heterogeneous graph data containing both ag_res and ab_res
            ag_chain_type: Antigen chain type prefix (default: 'ag')
            ab_chain_type: Antibody chain type prefix (default: 'ab')

        Returns:
            ag_embeddings: Antigen residue embeddings [n_ag, residue_dim]
            ab_embeddings: Antibody residue embeddings [n_ab, residue_dim]
        """
        # Extract initial features and create separate hetero_data for each chain
        ag_key = f"{ag_chain_type}_res"
        ab_key = f"{ab_chain_type}_res"

        # Get initial features
        ag_geom_features = hetero_data[ag_key].x  # [n_ag, geometric_dim]
        ab_geom_features = hetero_data[ab_key].x  # [n_ab, geometric_dim]

        # Apply feature mask for ablation study
        # RAAD 105D = first 20 (one-hot AA) + remaining 85 (geometric: RSA, SS, B-factor, etc.)
        if self.feature_mask == "aa_only":
            ag_geom_features = ag_geom_features.clone()
            ab_geom_features = ab_geom_features.clone()
            ag_geom_features[:, 20:] = 0.0  # Zero out geometric features
            ab_geom_features[:, 20:] = 0.0
        elif self.feature_mask == "geo_only":
            ag_geom_features = ag_geom_features.clone()
            ab_geom_features = ab_geom_features.clone()
            ag_geom_features[:, :20] = 0.0  # Zero out AA one-hot
            ab_geom_features[:, :20] = 0.0

        # Prepare feature list - must match initialization logic exactly
        ag_feature_list = [ag_geom_features]
        ab_feature_list = [ab_geom_features]

        if self.use_plm:
            ag_plm_features = self.ag_plm_proj(hetero_data[ag_key].plm)
            ab_plm_features = self.ab_plm_proj(hetero_data[ab_key].plm)
            ag_feature_list.append(ag_plm_features)
            ab_feature_list.append(ab_plm_features)

        # Fuse features
        if isinstance(self.ag_feature_fusion, GatedFeatureFusion):
            ag_features = self.ag_feature_fusion(ag_feature_list)
        else:
            ag_combined = torch.cat(ag_feature_list, dim=1)
            ag_features = self.ag_feature_fusion(ag_combined)

        if isinstance(self.ab_feature_fusion, GatedFeatureFusion):
            ab_features = self.ab_feature_fusion(ab_feature_list)
        else:
            ab_combined = torch.cat(ab_feature_list, dim=1)
            ab_features = self.ab_feature_fusion(ab_combined)


        # Create separate hetero_data for each chain (to avoid interference)
        # Keep as original HeteroData format for ResMP compatibility
        ag_hetero_data = hetero_data
        ab_hetero_data = hetero_data

        # Pass through Simplified epiformer blocks
        for block in self.epiformer_blocks:
            ag_features, ab_features = block(ag_hetero_data, ab_hetero_data, ag_features, ab_features)

            # Final normalization before decoder
        ag_features = self.final_ln_ag(ag_features)
        ab_features = self.final_ln_ab(ab_features)

        return ag_features, ab_features



## *********** epiformer encoder ends here **************



class PairRepresentation(nn.Module):
    """Minimal pair representation with bilinear initialization"""

    def __init__(self, node_dim, pair_dim, dropout=0.1):
        super().__init__()
        self.left_proj = nn.Linear(node_dim, pair_dim)
        self.right_proj = nn.Linear(node_dim, pair_dim)
        self.norm = nn.LayerNorm(pair_dim)
        self.dropout = nn.Dropout(dropout)

    def init_from_nodes(self, ag_features, ab_features):
        """Initialize pair representation from node features using outer product"""
        ag_proj = self.left_proj(ag_features)   # [n_ag, pair_dim]
        ab_proj = self.right_proj(ab_features)  # [n_ab, pair_dim]

        # Outer product: [n_ag, 1, pair_dim] + [1, n_ab, pair_dim] -> [n_ag, n_ab, pair_dim]
        pair_repr = ag_proj.unsqueeze(1) + ab_proj.unsqueeze(0)

        return self.dropout(self.norm(pair_repr))


class TriangleUpdate(nn.Module):
    """Triangle multiplicative update for pair representation"""

    def __init__(self, pair_dim, dropout=0.1):
        super().__init__()
        self.pair_dim = pair_dim

        # Projections for triangle update
        self.left_proj = nn.Linear(pair_dim, pair_dim)
        self.right_proj = nn.Linear(pair_dim, pair_dim)
        self.gate_proj = nn.Linear(pair_dim, pair_dim)
        self.out_proj = nn.Linear(pair_dim, pair_dim)

        self.norm = nn.LayerNorm(pair_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, pair_repr, mask=None):
        """Apply simplified pair update (not true triangle for cross-chain)"""
        residual = pair_repr
        pair_repr = self.norm(pair_repr)

        # Project features
        left = self.left_proj(pair_repr)   # [n_ag, n_ab, pair_dim]
        right = self.right_proj(pair_repr) # [n_ag, n_ab, pair_dim]
        gate = torch.sigmoid(self.gate_proj(pair_repr))

        # Simple multiplicative interaction (not true triangle update)
        # This avoids dimension mismatch while maintaining pair interactions
        update = left * right  # Element-wise interaction [n_ag, n_ab, pair_dim]

        # Apply mask if provided (same complex constraint)
        if mask is not None:
            update = update * mask.unsqueeze(-1)

        # Gate and project
        update = gate * self.out_proj(update)

        return residual + self.dropout(update)
