import os
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from typing import Dict, Any
from omegaconf import DictConfig
import numpy as np

# from model.epiformer_encoder import epiformerEncoder
from model.encoder import EpiformerEncoder

from model.decoder import Decoder 


class EpiformerModel(nn.Module):
    """
    epiformer-style Model for Epitope-Paratope Prediction
    Uses unified epiformerEncoder with interleaved ResMP and cross-chain attention.
    """
    
    def __init__(self, cfg: DictConfig):
        """
        Initialize the epiformer model.
        
        Args:
            cfg: Hydra configuration object
        """
        super().__init__()
        
        # Store thresholds for inference - SIMPLIFIED: Always use 0.5 (WALLE approach)
        self.epi_threshold = 0.5  # Fixed threshold following WALLE paper approach
        self.para_threshold = 0.5  # Fixed threshold following WALLE paper approach
        
        # Extract model configuration
        model_cfg = cfg.model        
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(model_cfg, 'epiformer') and hasattr(model_cfg.epiformer, 'use_gradient_checkpointing'):
            self.use_gradient_checkpointing = bool(model_cfg.epiformer.use_gradient_checkpointing)
            self.checkpoint_segments = int(getattr(model_cfg.epiformer, 'checkpoint_segments', 2))
        else:
            self.use_gradient_checkpointing = getattr(model_cfg, 'use_gradient_checkpointing', False)
            self.checkpoint_segments = getattr(model_cfg, 'checkpoint_segments', 2)
        

        
        # Initialize unified epiformer encoder (processes both AG and AB)
        evo_cfg = model_cfg.epiformer
        epiformer_params = {
            'residue_dim': int(evo_cfg.residue_dim),
            'residue_hidden_dim': int(evo_cfg.residue_hidden_dim),
            'residue_layers': int(evo_cfg.residue_layers),
            'geo_dim':  int(evo_cfg.geo_dim),
            'edge_dim': int(evo_cfg.edge_dim),
            'num_relations': int(cfg.dataset.graph_num_relations),
            'plm_dim': int(evo_cfg.plm_dim),
            'n_heads': int(evo_cfg.n_heads),
            'dropout': float(evo_cfg.dropout),
            'use_layer_norm': bool(evo_cfg.use_layer_norm),
            'activation': str(getattr(evo_cfg, 'activation', 'silu')),
            'ag_feature_fusion_type': str(evo_cfg.ag_feature_fusion_type),
            'ab_feature_fusion_type': str(evo_cfg.ab_feature_fusion_type),
            'ag_plm_in_dim': int(model_cfg.ag_encoder.plm_in_dim),
            'ab_plm_in_dim': int(model_cfg.ab_encoder.plm_in_dim),
            "ag_plm_type": str(cfg.dataset.plm_type),
            "ag_resmp_type": str(evo_cfg.ag_resmp_type),
            "ab_resmp_type": str(evo_cfg.ab_resmp_type),
            # NeurIPS 2026: No PLM by default
            "use_plm": bool(getattr(evo_cfg, 'use_plm', False)),
            "update_coords": bool(getattr(evo_cfg, 'update_coords', True)),
            "cross_attn_mode": str(getattr(evo_cfg, 'cross_attn_mode', 'bidirectional')),
            "feature_mask": str(getattr(evo_cfg, 'feature_mask', 'none')),
        }

        self.epiformer_encoder = EpiformerEncoder(**epiformer_params)

        # Initialize decoder
        # REGULARIZATION FIX: Use specific decoder dropout rate
        decoder_dropout = float(model_cfg.dropout_rates.decoder) if hasattr(model_cfg, 'dropout_rates') else float(model_cfg.dropout)
        # Decoder d_model must match encoder residue_dim (encoder output dimension)
        residue_dim = int(evo_cfg.residue_dim)
        decoder_params = {
            'd_model': residue_dim,
            'n_heads': int(model_cfg.decoder.n_heads),
            'n_layers': int(model_cfg.decoder.decoder_layers),
            'd_ff': int(model_cfg.decoder.d_ff),
            'd_k': int(model_cfg.decoder.d_k),
            'dropout': decoder_dropout,
            'decoder_type': str(model_cfg.decoder.type),
            'sampling_strat': str(model_cfg.decoder.sampling_strat),
            'predict_distances': bool(model_cfg.decoder.predict_distances),
            'activation': str(getattr(evo_cfg, 'activation', 'silu'))
        }
        self.decoder = Decoder(**decoder_params)



    def forward(self, hetero_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # Encode antigen and antibody jointly using epiformer
        if self.use_gradient_checkpointing and self.training:
            encoder_output = checkpoint.checkpoint(self.epiformer_encoder, hetero_data, 'ag', 'ab')
        else:
            encoder_output = self.epiformer_encoder(hetero_data, 'ag', 'ab')
        
        # Handle pair representation output
        if len(encoder_output) == 3:
            ag_embeddings, ab_embeddings, pair_repr = encoder_output
        else:
            ag_embeddings, ab_embeddings = encoder_output
            pair_repr = None

        # Get batch information
        ag_batch = hetero_data['ag_res'].batch  # shape: [total_ag_residues]
        ab_batch = hetero_data['ab_res'].batch  # shape: [total_ab_residues]
        num_graphs = ag_batch.max().item() + 1

        total_ag = ag_embeddings.shape[0]
        total_ab = ab_embeddings.shape[0]
        device = ag_embeddings.device

        # Initialize large matrices with zeros
        # basically a sparse matrix with interaction sub-matrices on the diagonal
        interaction_matrix = torch.zeros(total_ag, total_ab, device=device) 
        distance_logits = torch.zeros(total_ag, total_ab, 5, device=device) if self.decoder.predict_distances else None

        epitope_prob_list = []
        paratope_prob_list = []

        for i in range(num_graphs):
            # Get indices for this graph
            ag_indices = torch.where(ag_batch == i)[0]
            ab_indices = torch.where(ab_batch == i)[0]
            
            ag_emb_i = ag_embeddings[ag_indices]  # [n_ag_i, d_model]
            ab_emb_i = ab_embeddings[ab_indices]  # [n_ab_i, d_model]

            # Skip if no residues in either antigen or antibody
            if ag_emb_i.size(0) == 0 or ab_emb_i.size(0) == 0:
                continue

            # Add batch dimension
            ag_emb_i = ag_emb_i.unsqueeze(0)  # [1, n_ag_i, d_model]
            ab_emb_i = ab_emb_i.unsqueeze(0)  # [1, n_ab_i, d_model]

            # Pass to decoder
            outputs_i = self.decoder(ag_emb_i, ab_emb_i)

            # Get the interaction matrix for graph i
            inter_mat_i = outputs_i['interaction_matrix'].squeeze(0)  # [n_ag_i, n_ab_i]
            
            # Use advanced indexing with meshgrid
            ag_grid, ab_grid = torch.meshgrid(ag_indices, ab_indices, indexing='ij')
            interaction_matrix[ag_grid, ab_grid] = inter_mat_i

            # Get distance logits if enabled
            if self.decoder.predict_distances and 'distance_logits' in outputs_i and outputs_i['distance_logits'] is not None:
                dist_logits_i = outputs_i['distance_logits'].squeeze(0)  # [n_ag_i, n_ab_i, 5]
                distance_logits[ag_grid, ab_grid] = dist_logits_i

            epitope_prob_i = outputs_i['epitope_prob'].squeeze(0)  # [n_ag_i]
            paratope_prob_i = outputs_i['paratope_prob'].squeeze(0)  # [n_ab_i]

            epitope_prob_list.append(epitope_prob_i)
            paratope_prob_list.append(paratope_prob_i)

        # Concatenate probabilities from all graphs
        epitope_prob = torch.cat(epitope_prob_list, dim=0) if epitope_prob_list else torch.tensor([], device=device)
        paratope_prob = torch.cat(paratope_prob_list, dim=0) if paratope_prob_list else torch.tensor([], device=device)

        # Apply thresholds to get binary predictions
        epitope_pred = (epitope_prob > self.epi_threshold).float() if epitope_prob.numel() > 0 else torch.tensor([], device=device)
        paratope_pred = (paratope_prob > self.para_threshold).float() if paratope_prob.numel() > 0 else torch.tensor([], device=device)

        # Prepare output dictionary
        output_dict = {
            'interaction_matrix': interaction_matrix,
            'epitope_prob': epitope_prob,
            'paratope_prob': paratope_prob,
            'epitope_pred': epitope_pred,
            'paratope_pred': paratope_pred,
            'ag_embed': ag_embeddings,
            'ab_embed': ab_embeddings,
        }

        # Add distance logits to output if enabled
        if self.decoder.predict_distances:
            output_dict['distance_logits'] = distance_logits

        return output_dict