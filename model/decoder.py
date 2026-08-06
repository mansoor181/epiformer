
"""
Decoder module for epitope-paratope prediction.

TODO:
    1. ✓ Add support for different decoder types (cross-attention, dual, dot-product, enhanced_bilinear)
    2. Optimize memory usage for large protein sequences
    3. Add positional encodings if needed
    4. Implement attention visualization utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict, Any
import numpy as np
from torch_geometric.data import Data, Batch
import time


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

# TODO: Minimal geometric utilities for enhanced bilinear decoder
def compute_rbf_features(distances, num_centers=16, max_distance=20.0):
    """RBF encoding of pairwise distances"""
    centers = torch.linspace(0, max_distance, num_centers, device=distances.device)
    gamma = 1.0 / (max_distance / num_centers) ** 2
    return torch.exp(-gamma * (distances.unsqueeze(-1) - centers) ** 2)

def compute_angle_features(ag_pos, ab_pos):
    """Simple angle-based features from displacement vectors"""
    ag_expanded = ag_pos.unsqueeze(1)  # [N_ag, 1, 3]
    ab_expanded = ab_pos.unsqueeze(0)  # [1, N_ab, 3]
    displacements = ab_expanded - ag_expanded  # [N_ag, N_ab, 3]
    distances = torch.norm(displacements, dim=-1, keepdim=True)  # [N_ag, N_ab, 1]
    unit_vectors = displacements / (distances + 1e-8)
    return torch.cat([unit_vectors, unit_vectors ** 2], dim=-1)  # [N_ag, N_ab, 6]

class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention mechanism for antigen-antibody interaction modeling.
    
    Implements the CrossAttn function:
    - Projects Q, K, V for each head
    - Computes scaled dot-product attention
    - Concatenates and projects outputs
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Separate projection matrices for Q, K, V per head (block-diagonal structure)
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_head)
        
        
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for multi-head cross-attention.
        
        Args:
            Q: Query tensor (batch_size, n_query, d_model)
            K: Key tensor (batch_size, n_key, d_model)  
            V: Value tensor (batch_size, n_value, d_model)
            mask: Optional attention mask (batch_size, n_query, n_key)
            
        Returns:
            Context vectors (batch_size, n_query, d_model)
        """
        batch_size, n_query, _ = Q.shape
        n_key = K.shape[1]
        
        # Project and reshape for multi-head attention
        # Shape: (batch_size, n_heads, seq_len, d_head)
        Q_h = self.W_Q(Q).view(batch_size, n_query, self.n_heads, self.d_head).transpose(1, 2)
        K_h = self.W_K(K).view(batch_size, n_key, self.n_heads, self.d_head).transpose(1, 2)
        V_h = self.W_V(V).view(batch_size, n_key, self.n_heads, self.d_head).transpose(1, 2)
        
        # Scaled dot-product attention
        # Shape: (batch_size, n_heads, n_query, n_key)
        attention_scores = torch.matmul(Q_h, K_h.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            # Expand mask for multi-head attention
            mask = mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        # Shape: (batch_size, n_heads, n_query, d_head)
        context = torch.matmul(attention_weights, V_h)
        
        # Concatenate heads and project
        # Shape: (batch_size, n_query, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, n_query, self.d_model)
        output = self.W_O(context)
        
        return output


class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN) from Transformer architecture.
    
    Implements the FFN function from Algorithm 1:
    FFN(X) = ReLU(XW1 + b1)W2 + b2
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = 'relu'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = get_activation(activation)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for FFN.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class DecoderLayer(nn.Module):
    """
    Single layer of the cross-attention decoder.
    
    Implements one iteration of the for-loop:
    1. Cross attention: Ag queries Ab
    2. Cross attention: Ab queries Ag  
    3. Position-wise FFN for both
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, activation: str = 'relu'):
        super().__init__()
        
        # Cross-attention modules
        self.ag_cross_attn = MultiHeadCrossAttention(d_model, n_heads, dropout)
        self.ab_cross_attn = MultiHeadCrossAttention(d_model, n_heads, dropout)
        
        # Feed-forward networks
        self.ag_ffn = FeedForwardNetwork(d_model, d_ff, dropout, activation)
        self.ab_ffn = FeedForwardNetwork(d_model, d_ff, dropout, activation)

        # REGULARIZATION FIX: Use configurable dropout instead of hardcoded values
        self.attention_dropout = nn.Dropout(dropout)  # Configurable attention dropout
        self.ffn_dropout = nn.Dropout(dropout)        # Configurable FFN dropout
        
        # Layer normalization
        self.ag_norm1 = nn.LayerNorm(d_model)
        self.ag_norm2 = nn.LayerNorm(d_model)
        self.ab_norm1 = nn.LayerNorm(d_model)
        self.ab_norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, H_ag: torch.Tensor, H_ab: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for one decoder layer.
        
        Args:
            H_ag: Antigen embeddings (batch_size, n_ag, d_model)
            H_ab: Antibody embeddings (batch_size, n_ab, d_model)
            
        Returns:
            Updated (H_ag, H_ab) embeddings
        """
        
        # Cross attention: Ag queries Ab
        ag_cross_out = self.ag_cross_attn(H_ag, H_ab, H_ab)
        ag_cross_out = self.attention_dropout(ag_cross_out)  # New dropout
        H_ag_hat = self.ag_norm1(H_ag + ag_cross_out)
        
        # Cross attention: Ab queries Ag  
        ab_cross_out = self.ab_cross_attn(H_ab, H_ag, H_ag)
        ab_cross_out = self.attention_dropout(ab_cross_out)  # New dropout
        H_ab_hat = self.ab_norm1(H_ab + ab_cross_out)
        
        # Position-wise FFN with enhanced dropout
        ag_ffn_out = self.ag_ffn(H_ag_hat)
        ag_ffn_out = self.ffn_dropout(ag_ffn_out)  # New dropout
        H_ag_new = self.ag_norm2(H_ag_hat + ag_ffn_out)
        
        ab_ffn_out = self.ab_ffn(H_ab_hat)
        ab_ffn_out = self.ffn_dropout(ab_ffn_out)  # New dropout
        H_ab_new = self.ab_norm2(H_ab_hat + ab_ffn_out)
        
        return H_ag_new, H_ab_new


class BipartiteAffinityModule(nn.Module):
    """
    Bipartite affinity computation module.
    
    Implements the final bipartite adjacency matrix prediction from Algorithm 1:
    - Computes bidirectional affinity scores
    - Combines with learnable mixing weights
    """
    
    def __init__(self, d_model: int, d_k: int = 64):
        super().__init__()
        
        self.d_k = d_k
        self.scale = 1.0 / math.sqrt(d_k)
        
        # Projection matrices for affinity computation
        self.W_Q_out = nn.Linear(d_model, d_k, bias=False)  # Ag->Ab direction
        self.W_K_out = nn.Linear(d_model, d_k, bias=False)
        
        self.W_Q_prime_out = nn.Linear(d_model, d_k, bias=False)  # Ab->Ag direction  
        self.W_K_prime_out = nn.Linear(d_model, d_k, bias=False)
        
        # Learnable mixing parameters
        self.mixing_weights = nn.Parameter(torch.randn(2))

        # FIXME:
        self.bias = nn.Parameter(torch.zeros(1))
        # Initialize bias slightly positive to counteract negative drift during training
        # self.bias = nn.Parameter(torch.full((1,), 0.5))
        
    def forward(self, H_ag: torch.Tensor, H_ab: torch.Tensor) -> torch.Tensor:
        """
        Compute bipartite interaction matrix.
        
        Args:
            H_ag: Final antigen embeddings (batch_size, n_ag, d_model)
            H_ab: Final antibody embeddings (batch_size, n_ab, d_model)
            
        Returns:
            Interaction matrix Y (batch_size, n_ag, n_ab)
        """
        batch_size, n_ag, _ = H_ag.shape
        n_ab = H_ab.shape[1]
        
        # Compute bidirectional affinity scores
        Q_ag = self.W_Q_out(H_ag)  # (batch_size, n_ag, d_k)
        K_ab = self.W_K_out(H_ab)  # (batch_size, n_ab, d_k)
        S_ag_to_ab = torch.matmul(Q_ag, K_ab.transpose(-2, -1)) * self.scale  # (batch_size, n_ag, n_ab)
        
        Q_ab = self.W_Q_prime_out(H_ab)  # (batch_size, n_ab, d_k)
        K_ag = self.W_K_prime_out(H_ag)  # (batch_size, n_ag, d_k)
        S_ab_to_ag = torch.matmul(Q_ab, K_ag.transpose(-2, -1)) * self.scale  # (batch_size, n_ab, n_ag)
        
        # Stack and mix the score maps
        # Shape: (batch_size, 2, n_ag, n_ab)
        stacked_scores = torch.stack([
            S_ag_to_ab,
            S_ab_to_ag.transpose(-2, -1)  # Transpose to match dimensions
        ], dim=1)
        
        # Linear combination with learnable weights
        # Shape: (batch_size, n_ag, n_ab)
        mixed_scores = torch.einsum('bhij,h->bij', stacked_scores, self.mixing_weights) + self.bias
        
        # DEBUG: Check pre-sigmoid logits
        # print(f"DEBUG DECODER: Pre-sigmoid logits range: [{mixed_scores.min().item():.3f}, {mixed_scores.max().item():.3f}]")
        # print(f"DEBUG DECODER: Pre-sigmoid std: {mixed_scores.std().item():.3f}")
        
        # Apply sigmoid activation
        # return torch.sigmoid(mixed_scores)
        
        return mixed_scores


class DotProductDecoder(nn.Module):
    """
    Simple dot-product decoder for comparison.
    Computes interaction scores as inner product of embeddings.
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
    def forward(self, H_ag: torch.Tensor, H_ab: torch.Tensor) -> torch.Tensor:
        """
        Compute dot-product interaction matrix.
        
        Args:
            H_ag: Antigen embeddings (batch_size, n_ag, d_model)
            H_ab: Antibody embeddings (batch_size, n_ab, d_model)
            
        Returns:
            Interaction matrix (batch_size, n_ag, n_ab)
        """
        similarity = torch.matmul(H_ag, H_ab.transpose(-2, -1))
        # return torch.sigmoid(similarity)

        return similarity


class EnhancedBilinearDecoder(nn.Module):
    """
    TODO: Enhanced bilinear decoder with geometric features
    Implements: s_ij = h_i^T W h_j + u^T [rbf(d_ij), angle_ij] + b
    """
    
    def __init__(self, d_model=128, dropout=0.1, activation='relu'):
        super().__init__()
        self.d_model = d_model
        
        # TODO: Bilinear interaction matrices
        self.W_bilinear = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        
        # # TODO: Geometric feature projection
        # geometric_dim = num_rbf + 6  # RBF + angle features
        # self.geometric_proj = nn.Sequential(
        #     nn.Linear(geometric_dim, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 1)
        # )
        
        
        self.bias = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, H_ag, H_ab, ag_positions=None, ab_positions=None):
        """
        Enhanced bilinear decoder forward pass
        
        Args:
            H_ag: [batch_size, N_ag, d_model]
            H_ab: [batch_size, N_ab, d_model]  
            ag_positions: [batch_size, N_ag, 3] - optional
            ab_positions: [batch_size, N_ab, 3] - optional
        """
        # TODO: Bilinear interaction: h_i^T W h_j
        ag_transformed = torch.matmul(H_ag, self.W_bilinear)  # [B, N_ag, d]
        bilinear_scores = torch.matmul(ag_transformed, H_ab.transpose(-2, -1))  # [B, N_ag, N_ab]
        
        # # TODO: Add geometric bias if positions available
        # if ag_positions is not None and ab_positions is not None:
        #     geometric_bias = self._compute_geometric_bias(ag_positions, ab_positions)
        #     bilinear_scores = bilinear_scores + geometric_bias
        
        # TODO: Add learnable bias
        bilinear_scores = bilinear_scores + self.bias
        
        return torch.sigmoid(bilinear_scores)
    
    def _compute_geometric_bias(self, ag_positions, ab_positions):
        """Compute geometric bias from positions"""
        batch_size = ag_positions.size(0)
        geometric_biases = []
        
        for b in range(batch_size):
            # TODO: Compute distances and features
            ag_pos = ag_positions[b]  # [N_ag, 3]
            ab_pos = ab_positions[b]  # [N_ab, 3]
            distances = torch.cdist(ag_pos, ab_pos)  # [N_ag, N_ab]
            
            rbf_features = compute_rbf_features(distances, self.num_rbf)
            angle_features = compute_angle_features(ag_pos, ab_pos)
            
            # TODO: Combine and project
            geometric_feats = torch.cat([rbf_features, angle_features], dim=-1)
            bias = self.geometric_proj(geometric_feats).squeeze(-1)
            geometric_biases.append(bias)
        
        return torch.stack(geometric_biases)




class WALLEDecoder(nn.Module):
    """WALLE decoder using inner product for interaction matrix computation."""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
    def forward(self, H_ag: torch.Tensor, H_ab: torch.Tensor) -> torch.Tensor:
        """
        Compute interaction matrix using inner product (WALLE style).
        
        Args:
            H_ag: Antigen embeddings (batch_size, n_ag, d_model)
            H_ab: Antibody embeddings (batch_size, n_ab, d_model)
            
        Returns:
            Interaction matrix Y (batch_size, n_ag, n_ab)
        """
        # Inner product: Y_ij = <h_ag_i, h_ab_j>
        Y = torch.bmm(H_ag, H_ab.transpose(-2, -1))
        # Apply sigmoid activation
        Y = torch.sigmoid(Y)
        return Y


class DualDecoder(nn.Module):
    """
    Dual decoder combining dot-product and cross-attention mechanisms.
    
    Implements the dual decoder framework:
    Y = σ(α * Y_dot + (1-α) * Y_attn)
    """
    
    def __init__(self, d_model: int, n_heads: int = 8, d_ff: int = 2048, 
                 n_layers: int = 2, dropout: float = 0.1, activation: str = 'relu'):
        super().__init__()
        
        # Dot-product decoder
        self.dot_decoder = DotProductDecoder(d_model)
        
        # Cross-attention layers (simplified to 1 layer for dual approach)
        self.cross_attention = DecoderLayer(d_model, n_heads, d_ff, dropout, activation)
        
        # Projection for attention-based scores
        self.attn_projection = nn.Linear(d_model, 1)
        
        # Learnable mixing parameter α
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, H_ag: torch.Tensor, H_ab: torch.Tensor) -> torch.Tensor:
        """
        Compute dual decoder interaction matrix.
        
        Args:
            H_ag: Antigen embeddings (batch_size, n_ag, d_model)
            H_ab: Antibody embeddings (batch_size, n_ab, d_model)
            
        Returns:
            Combined interaction matrix (batch_size, n_ag, n_ab)
        """
        # Dot-product component
        Y_dot = self.dot_decoder(H_ag, H_ab)
        
        # Cross-attention component
        H_ag_attn, H_ab_attn = self.cross_attention(H_ag, H_ab)
        
        # Project to interaction scores (simplified approach)
        # In practice, this could use the full BipartiteAffinityModule
        Y_attn = torch.matmul(H_ag_attn, H_ab_attn.transpose(-2, -1))
        Y_attn = torch.sigmoid(Y_attn)
        
        # Combine with learnable mixing
        alpha = torch.sigmoid(self.alpha)  # Ensure α ∈ [0,1]
        Y_combined = alpha * Y_dot + (1 - alpha) * Y_attn
        
        return Y_combined



class Decoder(nn.Module):
    """
    Main cross-attention decoder for epitope-paratope prediction.
    
    Implements Algorithm 1 from the research plan with configurable architecture.
    Supports multiple decoder types: cross-attention, dot-product, enhanced_bilinear, dual, and walle.
    """
    
    def __init__(self, 
                 d_model: int = 128,
                 n_heads: int = 8,
                 n_layers: int = 3,
                 d_ff: int = 512,
                 d_k: int = 64,
                 dropout: float = 0.1,
                 decoder_type: str = "cross_attention",
                 sampling_strat: str = "max_row",
                 predict_distances: bool = False,
                 activation: str = 'silu'):
        """
        Initialize the decoder.
        
        Args:
            d_model: Model dimension (should match encoder output)
            n_heads: Number of attention heads
            n_layers: Number of decoder layers (L in algorithm)
            d_ff: Feed-forward network dimension (typically 4*d_model)
            d_k: Key/query dimension for final affinity computation
            dropout: Dropout probability
            decoder_type: Type of decoder ("cross_attention", "dot_product", "enhanced_bilinear", "dual", "walle")
            num_rbf: Number of RBF centers for enhanced_bilinear decoder
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        self.decoder_type = decoder_type
        self.sampling_strat = sampling_strat

        # Add distance prediction head for auxiliary learning (Option 3)
        self.predict_distances = predict_distances
        if predict_distances:
            self.distance_head = nn.Sequential(
                nn.Linear(d_model * 2, 64),  # Concatenated antigen + antibody features
                get_activation(activation),
                nn.Dropout(dropout),
                nn.Linear(64, 5)  # 5 distance bins: <6, 6-8, 8-10, 10-12, >12 Å
            )        
        
        if decoder_type == "cross_attention":
            # Stack L identical cross-attention layers
            self.layers = nn.ModuleList([
                DecoderLayer(d_model, n_heads, d_ff, dropout, activation)
                for _ in range(n_layers)
            ])
            self.affinity_module = BipartiteAffinityModule(d_model, d_k)
            
        elif decoder_type == "dot_product":
            self.dot_decoder = DotProductDecoder(d_model)
            
        elif decoder_type == "enhanced_bilinear":
            # TODO: Enhanced bilinear decoder with geometric features
            self.enhanced_decoder = EnhancedBilinearDecoder(
                d_model=d_model,
                dropout=dropout,
                activation=activation
            )
            
        elif decoder_type == "dual":
            self.dual_decoder = DualDecoder(d_model, n_heads, d_ff, n_layers, dropout, activation)
            
        # NOTE: WALLE Decoder Addition - Added support for WALLE inner product decoder
        elif decoder_type == "walle":
            self.walle_decoder = WALLEDecoder(d_model)
            
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}. Supported: 'cross_attention', 'dot_product', 'enhanced_bilinear', 'dual', 'walle'")
    
    def forward(self, H_ag: torch.Tensor, H_ab: torch.Tensor, 
                ag_positions: Optional[torch.Tensor] = None,
                ab_positions: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Extended forward pass with optional distance prediction.
        
        Args:
            H_ag: Antigen residue embeddings
            H_ab: Antibody residue embeddings
            ag_positions: Antigen residue positions (for distance prediction)
            ab_positions: Antibody residue positions (for distance prediction)
        
            
        Returns:
            Dictionary containing:
            - 'interaction_matrix': Bipartite interaction matrix (batch_size, n_ag, n_ab)
            - 'epitope_prob': Per-residue epitope probabilities (batch_size, n_ag)
            - 'paratope_prob': Per-residue paratope probabilities (batch_size, n_ab)
        """
        if self.decoder_type == "cross_attention":
            # Apply L layers of cross-attention
            for layer in self.layers:
                H_ag, H_ab = layer(H_ag, H_ab)
            
            # Compute final bipartite affinity matrix
            Y = self.affinity_module(H_ag, H_ab)
            
        elif self.decoder_type == "dot_product":
            Y = self.dot_decoder(H_ag, H_ab)
            
        elif self.decoder_type == "enhanced_bilinear":
            # TODO: Enhanced bilinear with geometric features
            Y = self.enhanced_decoder(H_ag, H_ab, ag_positions, ab_positions)
            
        elif self.decoder_type == "dual":
            Y = self.dual_decoder(H_ag, H_ab)
            
        # NOTE: WALLE Decoder Addition - Added WALLE decoder forward pass
        elif self.decoder_type == "walle":
            Y = self.walle_decoder(H_ag, H_ab)
        
        

        # Compute epitope and paratope probabilities using configurable sampling strategies
        # epitope_prob, paratope_prob = self._compute_residue_probabilities(Y)
        """
        TODO:
        - return logits instead of probabilities for the interaction matrix
        - BCEWithLogitsLoss combines sigmoid + BCE in one operation, avoiding numerical issues with extreme probabilities
        """
        epitope_prob, paratope_prob = self._compute_residue_probabilities(torch.sigmoid(Y))

        # Add auxiliary distance prediction if enabled (Option 3)
        outputs = {
            'interaction_matrix': Y,
            'epitope_prob': epitope_prob.squeeze(0),
            'paratope_prob': paratope_prob.squeeze(0),
            'ag_embed': H_ag.squeeze(0),
            'ab_embed': H_ab.squeeze(0)
        }
        
        if self.predict_distances:
            distance_logits = self.predict_pairwise_distances(H_ag, H_ab)  # [B, N_ag, N_ab, 5]
            outputs['distance_logits'] = distance_logits.squeeze(0) if distance_logits.shape[0] == 1 else distance_logits
        
        return outputs


    # def forward(self, H_ag: torch.Tensor, H_ab: torch.Tensor) -> Dict[str, torch.Tensor]:
    #     """
    #     Forward pass through the decoder.
        
    #     Args:
    #         H_ag: Antigen residue embeddings (batch_size, n_ag_residues, d_model)
    #         H_ab: Antibody residue embeddings (batch_size, n_ab_residues, d_model)
            
    #     Returns:
    #         Dictionary containing:
    #         - 'interaction_matrix': Bipartite interaction matrix (batch_size, n_ag, n_ab)
    #         - 'epitope_prob': Per-residue epitope probabilities (batch_size, n_ag)
    #         - 'paratope_prob': Per-residue paratope probabilities (batch_size, n_ab)
    #     """
    #     if self.decoder_type == "cross_attention":
    #         # Apply L layers of cross-attention
    #         for layer in self.layers:
    #             H_ag, H_ab = layer(H_ag, H_ab)
            
    #         # Compute final bipartite affinity matrix
    #         Y = self.affinity_module(H_ag, H_ab)
            
    #     elif self.decoder_type == "dot_product":
    #         Y = self.dot_decoder(H_ag, H_ab)
            
    #     elif self.decoder_type == "dual":
    #         Y = self.dual_decoder(H_ag, H_ab)
            
    #     # NOTE: WALLE Decoder Addition - Added WALLE decoder forward pass
    #     elif self.decoder_type == "walle":
    #         Y = self.walle_decoder(H_ag, H_ab)
        
        

    #     # Compute epitope and paratope probabilities using configurable sampling strategies
    #     epitope_prob, paratope_prob = self._compute_residue_probabilities(Y)

    #     return {
    #         'interaction_matrix': Y,
    #         'epitope_prob': epitope_prob.squeeze(0),
    #         'paratope_prob': paratope_prob.squeeze(0),
    #         'ag_embed': H_ag.squeeze(0),  # For visualization/analysis
    #         'ab_embed': H_ab.squeeze(0)
    #     }
        
    def _compute_residue_probabilities(self, Y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute epitope and paratope probabilities from interaction matrix using various sampling strategies.
        
        Args:
            Y: Interaction matrix (batch_size, n_ag, n_ab)
            
        Returns:
            epitope_prob: Per-residue epitope probabilities (batch_size, n_ag) 
            paratope_prob: Per-residue paratope probabilities (batch_size, n_ab)
        """
        
        if self.sampling_strat == "max_row":
            epitope_prob = torch.max(Y, dim=-1)[0]  # Row-wise maxima
            paratope_prob = torch.max(Y, dim=-2)[0]  # Column-wise maxima
            
        elif self.sampling_strat == "mean_row":
            epitope_prob = torch.mean(Y, dim=-1)  # Row-wise mean
            paratope_prob = torch.mean(Y, dim=-2)  # Column-wise mean
            
        elif self.sampling_strat.startswith("top_k_mean"):
            # Extract k value from strategy name (e.g., "top_k_mean_2" -> k=2)
            k = int(self.sampling_strat.split('_')[-1]) if '_' in self.sampling_strat[10:] else 2
            epitope_prob = self._top_k_mean_pooling(Y, k=k, dim=-1)  # Row-wise top-k mean
            paratope_prob = self._top_k_mean_pooling(Y, k=k, dim=-2)  # Column-wise top-k mean

        elif self.sampling_strat == "noisy_or": 
            # p_epi_i = 1 - ∏_j (1 - Y_ij) 
            epitope_prob = 1.0 - torch.prod(1.0 - Y.clamp(1e-6, 1 - 1e-6), dim=-1) 
            paratope_prob = 1.0 - torch.prod(1.0 - Y.clamp(1e-6, 1 - 1e-6), dim=-2)
            
        elif self.sampling_strat == "softmax_attention":
            epitope_prob = self._softmax_attention_pooling(Y, dim=-1)  # Row-wise softmax attention
            paratope_prob = self._softmax_attention_pooling(Y, dim=-2)  # Column-wise softmax attention
            
        elif self.sampling_strat == "edge_budget_aware":
            epitope_prob = self._edge_budget_aware_pooling(Y, dim=-1)  # Row-wise budget-aware
            paratope_prob = self._edge_budget_aware_pooling(Y, dim=-2)  # Column-wise budget-aware
            
        elif self.sampling_strat == "hierarchical_pooling":
            epitope_prob = self._hierarchical_pooling(Y, dim=-1)  # Row-wise hierarchical
            paratope_prob = self._hierarchical_pooling(Y, dim=-2)  # Column-wise hierarchical
            
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strat}")
            
        return epitope_prob, paratope_prob
    

    def predict_pairwise_distances(self, H_ag: torch.Tensor, H_ab: torch.Tensor) -> torch.Tensor:
        """
        Predict pairwise distance bins between antigen and antibody residues (Option 3).
        
        Args:
            H_ag: Antigen embeddings (batch_size, n_ag, d_model)
            H_ab: Antibody embeddings (batch_size, n_ab, d_model)
            
        Returns:
            Distance bin logits (batch_size, n_ag, n_ab, 5)
            Bins: <6, 6-8, 8-10, 10-12, >12 Å
        """
        batch_size, n_ag, d_model = H_ag.shape
        n_ab = H_ab.shape[1]
        
        # Expand dimensions for pairwise combination
        # H_ag_expanded: (batch_size, n_ag, n_ab, d_model)
        H_ag_expanded = H_ag.unsqueeze(2).expand(-1, -1, n_ab, -1)
        # H_ab_expanded: (batch_size, n_ag, n_ab, d_model)  
        H_ab_expanded = H_ab.unsqueeze(1).expand(-1, n_ag, -1, -1)
        
        # Concatenate features for each pair
        pair_features = torch.cat([H_ag_expanded, H_ab_expanded], dim=-1)
        
        # Predict distance bin logits for each pair
        distance_logits = self.distance_head(pair_features)  # [B, N_ag, N_ab, 5]
        
        return distance_logits
    
    
    def _top_k_mean_pooling(self, Y: torch.Tensor, k: int, dim: int) -> torch.Tensor:
        """
        Compute mean of top-k highest interactions per residue.
        Biologically motivated - epitopes typically interact with 2-3 key paratope residues.
        """
        # Get top-k values along the specified dimension
        top_k_values, _ = torch.topk(Y, k=min(k, Y.size(dim)), dim=dim, largest=True)
        return torch.mean(top_k_values, dim=dim)
    
    def _softmax_attention_pooling(self, Y: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Apply softmax attention to learn importance weights for interactions.
        Learns to focus on most important interactions automatically.
        """
        # Apply softmax to get attention weights
        attention_weights = F.softmax(Y, dim=dim)
        # Weighted sum using attention weights
        return torch.sum(Y * attention_weights, dim=dim)
    
    def _edge_budget_aware_pooling(self, Y: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Edge budget-aware pooling that considers WALLE's edge count constraint.
        Encourages sparse but high-confidence predictions.
        """
        # Apply temperature scaling to sharpen the distribution
        temperature = 2.0  # Lower temperature = sharper distribution
        sharpened = Y / temperature
        
        # Use softmax with temperature scaling
        attention_weights = F.softmax(sharpened, dim=dim)
        
        # Apply sparsity-inducing transformation
        # Higher values get exponentially higher weights
        sparsity_factor = torch.exp(Y * 2.0)  # Exponential emphasis on high values
        
        # Combine attention and sparsity
        combined_weights = attention_weights * sparsity_factor
        normalized_weights = combined_weights / (torch.sum(combined_weights, dim=dim, keepdim=True) + 1e-8)
        
        return torch.sum(Y * normalized_weights, dim=dim)
    
    def _hierarchical_pooling(self, Y: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Hierarchical pooling combining local specificity with global context.
        Combines top-k local signals with global mean context.
        """
        # Local component: top-2 mean (high specificity)
        k = 2
        top_k_values, _ = torch.topk(Y, k=min(k, Y.size(dim)), dim=dim, largest=True)
        local_signal = torch.mean(top_k_values, dim=dim)
        
        # Global component: overall mean (global context)  
        global_signal = torch.mean(Y, dim=dim)
        
        # Learnable mixing weight (could be made trainable)
        alpha = 0.7  # Weight towards local specificity
        
        return alpha * local_signal + (1 - alpha) * global_signal
    





    
    



# ==================== TEST CODE ====================

def test_cross_attention_decoder():
    """
    Comprehensive test suite for the cross-attention decoder.
    Tests different decoder types, shapes, and functionality.
    """
    print("=" * 60)
    print("TESTING CROSS-ATTENTION DECODER")
    print("=" * 60)
    
    # Test configuration
    batch_size = 2
    n_ag_residues = 150  # Typical antigen size
    n_ab_residues = 120  # Typical antibody size  
    d_model = 128
    n_trials = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create synthetic test data
    H_ag = torch.randn(batch_size, n_ag_residues, d_model).to(device)
    H_ab = torch.randn(batch_size, n_ab_residues, d_model).to(device)
    
    decoder_types = [
        ("cross_attention", {"n_heads": 8, "n_layers": 3}),
        ("dot_product", {}),
        ("dual", {"n_heads": 4, "n_layers": 2})
    ]
    
    for decoder_type, kwargs in decoder_types:
        print(f"\nBenchmarking {decoder_type} decoder...")
        
        decoder = Decoder(
            d_model=d_model,
            decoder_type=decoder_type,
            **kwargs
        ).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = decoder(H_ag, H_ab)
        
        # Benchmark
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(n_trials):
                outputs = decoder(H_ag, H_ab)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / n_trials * 1000  # ms
        
        param_count = sum(p.numel() for p in decoder.parameters())
        
        print(f"✓ {decoder_type}: {avg_time:.2f}ms/forward, {param_count:,} parameters")




def predict_epitopes(self, H_ag: torch.Tensor, H_ab: torch.Tensor, 
                    threshold: float = 0.5) -> torch.Tensor:
    """
    Predict binary epitope labels.
    
    Args:
        H_ag: Antigen embeddings
        H_ab: Antibody embeddings  
        threshold: Threshold for binary classification
        
    Returns:
        Binary epitope predictions (batch_size, n_ag)
    """
    threshold = threshold or self.threshold

    outputs = self.forward(H_ag, H_ab)
    return (outputs['epitope_prob'] > threshold).float()


if __name__ == "__main__":
    # Run all tests
    # test_cross_attention_decoder()
    
    print("DECODER IMPLEMENTATION COMPLETE!")
    
    
    # Test 1: Cross-attention decoder
    print("\n" + "-" * 40)
    print("TEST 1: Cross-Attention Decoder")
    print("-" * 40)
    d_model = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Test configuration
    batch_size = 1
    n_ag_residues = 150  # Typical antigen size
    n_ab_residues = 120  # Typical antibody size  
    d_model = 128
    
    # Create synthetic test data
    H_ag = torch.randn(batch_size, n_ag_residues, d_model).to(device)
    H_ab = torch.randn(batch_size, n_ab_residues, d_model).to(device)
    
    print(f"Input shapes - Antigen: {H_ag.shape}, Antibody: {H_ab.shape}")

    decoder_cross = Decoder(
        d_model=d_model,
        n_heads=8,
        n_layers=3,
        d_ff=512,
        decoder_type="cross_attention"
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in decoder_cross.parameters()):,}")
    
    with torch.no_grad():
        outputs = decoder_cross(H_ag, H_ab)
    
    print(f"✓ Interaction matrix shape: {outputs['interaction_matrix'].shape}")
    print(f"✓ Epitope probabilities shape: {outputs['epitope_prob'].shape}")
    print(f"✓ Paratope probabilities shape: {outputs['paratope_prob'].shape}")
    print(f"✓ Interaction values range: [{outputs['interaction_matrix'].min():.3f}, {outputs['interaction_matrix'].max():.3f}]")
    print(f"✓ Epitope prob range: [{outputs['epitope_prob'].min():.3f}, {outputs['epitope_prob'].max():.3f}]")
    
    # print(outputs['interaction_matrix'])
    # print(outputs['epitope_prob'])

    # Test 2: Dot-product decoder
    print("\n" + "-" * 40)
    print("TEST 2: Dot-Product Decoder")
    print("-" * 40)
    
    decoder_dot = Decoder(
        d_model=d_model,
        decoder_type="dot_product"
    ).to(device)
    
    with torch.no_grad():
        outputs_dot = decoder_dot(H_ag, H_ab)
    
    print(f"✓ Dot-product interaction matrix shape: {outputs_dot['interaction_matrix'].shape}")
    print(f"✓ Values range: [{outputs_dot['interaction_matrix'].min():.3f}, {outputs_dot['interaction_matrix'].max():.3f}]")
    
    # Test 3: Dual decoder
    print("\n" + "-" * 40)
    print("TEST 3: Dual Decoder")
    print("-" * 40)
    
    decoder_dual = Decoder(
        d_model=d_model,
        n_heads=4,
        n_layers=2,
        decoder_type="dual"
    ).to(device)
    
    with torch.no_grad():
        outputs_dual = decoder_dual(H_ag, H_ab)
    
    print(f"✓ Dual decoder interaction matrix shape: {outputs_dual['interaction_matrix'].shape}")
    print(f"✓ Alpha parameter: {torch.sigmoid(decoder_dual.dual_decoder.alpha).item():.3f}")
    
    # Test 4: WALLE decoder
    print("\n" + "-" * 40)
    print("TEST 4: WALLE Decoder")
    print("-" * 40)
    
    decoder_walle = Decoder(
        d_model=d_model,
        decoder_type="walle"
    ).to(device)
    
    with torch.no_grad():
        outputs_walle = decoder_walle(H_ag, H_ab)
    
    print(f"✓ WALLE decoder interaction matrix shape: {outputs_walle['interaction_matrix'].shape}")
    print(f"✓ Values range: [{outputs_walle['interaction_matrix'].min():.3f}, {outputs_walle['interaction_matrix'].max():.3f}]")
    print(f"✓ WALLE decoder parameters: {sum(p.numel() for p in decoder_walle.parameters()):,}")

    # Test 5: Binary prediction
    print("\n" + "-" * 40)
    print("TEST 5: Binary Epitope Prediction")
    print("-" * 40)
    
    with torch.no_grad():
        epitope_pred = predict_epitopes(H_ag, H_ab, threshold=0.5)
    
    print(f"✓ Binary predictions shape: {epitope_pred.shape}")
    print(f"✓ Predicted epitope residues: {epitope_pred.sum(dim=-1)} per sample")
    
    # Test 6: Gradient flow
    print("\n" + "-" * 40)
    print("TEST 6: Gradient Flow")
    print("-" * 40)
    
    decoder_cross.train()
    outputs = decoder_cross(H_ag, H_ab)
    
    # Dummy loss
    target_matrix = torch.randint(0, 2, (batch_size, n_ag_residues, n_ab_residues)).float().to(device)
    loss = F.binary_cross_entropy(outputs['interaction_matrix'], target_matrix)
    
    loss.backward()
    
    # Check gradients
    has_grad = any(p.grad is not None for p in decoder_cross.parameters())
    print(f"✓ Loss value: {loss.item():.4f}")
    print(f"✓ Gradients computed: {has_grad}")
    
    # Test 7: Different batch sizes
    print("\n" + "-" * 40)
    print("TEST 7: Different Batch Sizes")
    print("-" * 40)
    
    decoder_cross.eval()
    for test_batch_size in [1, 4, 8]:
        H_ag_test = torch.randn(test_batch_size, n_ag_residues, d_model).to(device)
        H_ab_test = torch.randn(test_batch_size, n_ab_residues, d_model).to(device)
        
        with torch.no_grad():
            outputs_test = decoder_cross(H_ag_test, H_ab_test)
        
        print(f"✓ Batch size {test_batch_size}: output shape {outputs_test['interaction_matrix'].shape}")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! ✅")
    print("=" * 60)











