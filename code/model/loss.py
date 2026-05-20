# - Avoid double-weighting contrastive loss (weight applied only once at the end).
# - Fix positive weighting for edge BCE (use element-wise weights for positives when using probabilities).
# - Make AdaptiveFocalLoss honor pos_weight as a proper α for the positive class.
# - Fix L2 regularization to true squared L2 (weight decay style).
# - Make force/geometric loss no-op if positions are not learnable (requires_grad=False) or data is missing.

"""
Losses for Hierarchical M3EPI Model
Includes:
  • Hierarchical loss (node, edge, contrastive)
  • Supervised BCE losses
  • Classic InfoNCE (intra/inter)
  • ReGCL-style Gradient-Weighted InfoNCE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.utils import to_scipy_sparse_matrix  # unused
import numpy as np
import math
from typing import Tuple, Dict, Optional

from torch.nn.functional import binary_cross_entropy_with_logits


# ===========================
# MAIN LOSS FUNCTIONS
# ===========================

"""
Adaptive Focal Loss:
    - Gamma increases from 2.0 → 5.0 during training
    - Automatically focuses on harder examples over time
    - Class-balanced alpha calculation

Class-Balanced Loss:
    - Handles extreme class imbalance (1:100+ ratio)
    - Based on effective number of samples
    - Prevents majority class domination

Multi-Task Weighting:
    - Learns optimal weights for different loss components
    - Reduces need for manual loss balancing
    - Automatically adjusts during training
    - automatically discovers the optimal balance between:
        epitope prediction, paratope prediction, interaction prediction, contrastive learning

Label Smoothing:
    - Improves generalization
    - Reduces overconfidence on training data
    - Helps with noisy labels
"""


class AdaptiveFocalLoss(nn.Module):
    """
    Focal loss with adaptive gamma and class balancing
    - Gamma increases as training progresses to focus more on hard examples
    - Alpha adjusts based on class imbalance ratio or provided pos_weight
    """
    def __init__(self, gamma=2.0, base_alpha=0.25, max_gamma=4.0, step_size=0.1):
        super().__init__()
        self.base_gamma = gamma
        self.current_gamma = gamma
        self.max_gamma = max_gamma
        self.step_size = step_size
        self.base_alpha = base_alpha
        
    def update_gamma(self, epoch):
        """Gradually increase gamma during training"""
        self.current_gamma = min(
            self.base_gamma + epoch * self.step_size, 
            self.max_gamma
        )
        
    def forward(self, preds, targets, pos_weight=None):
        """
        Args:
            preds: probabilities in [0,1]
            targets: 0/1 labels
            pos_weight: scalar ratio or weight for positives.
                        If provided and >0, convert to alpha via alpha = pos_weight / (1 + pos_weight).
                        Else, alpha is inferred from batch class ratio.
        """
        targets = targets.float()

        # Calculate alpha (positive-class weight)
        if pos_weight is not None:
            # Map a ratio r = (#neg/#pos) or a "weight" to alpha in (0,1)
            # If r is large -> alpha close to 1
            r = float(pos_weight)
            alpha_pos = r / (1.0 + r) if r > 0 else self.base_alpha
        else:
            # alpha from batch class ratio
            pos_count = targets.sum().clamp(min=1.0)
            total = targets.numel()
            neg_count = total - pos_count
            alpha_pos = (neg_count / total).item() if total > 0 else self.base_alpha

        # Standard BCE on probabilities
        bce_loss = F.binary_cross_entropy(preds, targets, reduction='none')

        # p_t used in focal modulating factor
        p_t = torch.where(targets == 1, preds, 1 - preds)
        
        # Focal factors
        modulating_factor = (1 - p_t).clamp(min=0.0) ** self.current_gamma
        alpha_factor = torch.where(targets == 1, preds.new_tensor(alpha_pos), preds.new_tensor(1.0 - alpha_pos))
        
        return (alpha_factor * modulating_factor * bce_loss).mean()


class ClassBalancedLoss(nn.Module):
    """
    Class-balanced loss to handle extreme class imbalance
    Reference: "Class-Balanced Loss Based on Effective Number of Samples"
    """
    def __init__(self, beta=0.9999):
        super().__init__()
        self.beta = beta
        
    def forward(self, preds, targets):
        targets = targets.float()
        # Convert targets to long for indexing
        unique_classes, counts = torch.unique(targets.long(), return_counts=True)
        
        weights = []
        for i in range(len(unique_classes)):
            n = counts[i].item()
            eff_num = (1 - self.beta ** n) / (1 - self.beta) if n > 0 else 0.0
            weight = (1 / eff_num) if eff_num > 0 else 0.0
            weights.append(weight)
        
        weights = torch.tensor(weights, device=preds.device, dtype=torch.float)
        if weights.sum() > 0:
            weights = weights / weights.sum() * len(unique_classes)
        
        weight_mask = torch.ones_like(preds, dtype=torch.float, device=preds.device)
        for i, c in enumerate(unique_classes):
            weight_mask[targets.long() == c] = weights[i]
            
        return (F.binary_cross_entropy(preds, targets, reduction='none') * weight_mask).mean()


class MultiTaskLossWrapper(nn.Module):
    """
    Learns weights for multiple loss components
    Reference: "Multi-Task Learning Using Uncertainty to Weigh Losses"
    """
    def __init__(self, num_tasks):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        
    def forward(self, losses):
        assert len(losses) == len(self.log_vars)
        
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
        return total_loss


class LabelSmoothing(nn.Module):
    """Applies label smoothing to binary classification"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        
    def forward(self, preds, targets):
        targets = targets.float()
        smoothed_targets = targets * self.confidence + 0.5 * self.smoothing
        return F.binary_cross_entropy(preds, smoothed_targets)


# --------------------------
# Hierarchical Loss Function
# --------------------------

def hierarchical_loss(outputs, batch, device, cfg, model=None, epoch=None, return_components=False):
    """
    Enhanced hierarchical loss with:
    - Adaptive focal loss
    - Class balancing
    - Multi-task weighting (disabled by default for stability)
    - Label smoothing (optional)
    """
    # Initialize loss modules
    focal_criterion = AdaptiveFocalLoss(
        gamma=cfg.loss.focal.gamma,
        base_alpha=cfg.loss.focal.alpha,
        max_gamma=cfg.loss.focal.max_gamma,
        step_size=cfg.loss.focal.step_size
    )
    if epoch is not None:
        focal_criterion.update_gamma(epoch)
    
    class_balanced_criterion = ClassBalancedLoss(beta=cfg.loss.class_balance.beta)
    label_smoothing_criterion = LabelSmoothing(smoothing=cfg.loss.label_smoothing)
    # multi_task = MultiTaskLossWrapper(num_tasks=3)  # disabled by default
    
    epitope_bce_loss = torch.tensor(0.0, device=device)
    epitope_dice_loss = torch.tensor(0.0, device=device)
    epitope_smoothness_loss_val = torch.tensor(0.0, device=device)
    epitope_count_loss = torch.tensor(0.0, device=device)
    paratope_bce_loss = torch.tensor(0.0, device=device)
    paratope_dice_loss = torch.tensor(0.0, device=device)
    paratope_count_loss = torch.tensor(0.0, device=device)
    paratope_smoothness_loss_val = torch.tensor(0.0, device=device)
    consistency_loss_val = torch.tensor(0.0, device=device)

    node_loss = torch.tensor(0.0, device=device)

    bce_weight = getattr(cfg.loss.node_prediction, 'bce_weight', 1.0)
    dice_weight = getattr(cfg.loss.node_prediction, 'dice_weight', 0.3)
    
    if cfg.loss.node_prediction.enabled:
        y_epi = batch['ag_res'].y.float()
        p_epi = outputs['epitope_prob'].float()

        if cfg.loss.node_prediction.task == "epi_only":
            # choose either bce or focal as primary loss
            if cfg.loss.node_prediction.name == "focal":
                epitope_bce_loss = focal_criterion(
                    p_epi, y_epi,
                    pos_weight=cfg.loss.node_prediction.epi_pos_weight
                )
                node_loss = bce_weight * epitope_bce_loss
            elif cfg.loss.node_prediction.name == "bce":
                epitope_bce_loss = binary_cross_entropy(
                    p_epi, y_epi,
                    pos_weight=cfg.loss.node_prediction.epi_pos_weight
                )
                node_loss = bce_weight * epitope_bce_loss

            else:
                raise ValueError(f"Unknown node_prediction.name: {cfg.loss.node_prediction.name}")
            
            # Add Dice loss for better imbalanced segmentation 
            if cfg.loss.node_prediction.dice_enabled:
                #    Enhancement: Use per-graph Dice if configured
                per_graph_dice = getattr(cfg.loss.node_prediction, 'dice_per_graph', False)
                batch_indices = batch['ag_res'].batch if hasattr(batch['ag_res'], 'batch') else None
                
                epitope_dice_loss = dice_loss(p_epi, y_epi, batch_indices, per_graph_dice)

                node_loss  = (node_loss + dice_weight * epitope_dice_loss)

            # Count regularizer for calibration
            if hasattr(cfg.loss, 'count_regularizer') and cfg.loss.node_prediction.count_regularizer_enabled:
                if cfg.loss.count_regularizer.per_graph_matching and hasattr(batch['ag_res'], 'batch'):
                    epitope_count_loss = per_graph_count_loss(
                        p_epi, y_epi, batch['ag_res'].batch, cfg.loss.count_regularizer.epitope_weight
                    )
                node_loss = (node_loss + epitope_count_loss)
                
            # Smoothness regularizer for contiguous patches
            if (hasattr(cfg.loss.node_prediction, 'smoothness_weight') and 
                cfg.loss.node_prediction.smoothness_weight > 0 and
                ('ag_res', 'r3', 'ag_res') in batch) and cfg.loss.node_prediction.smoothness_enabled:
                ag_edge_index = batch[('ag_res', 'r3', 'ag_res')].edge_index
                epitope_smoothness_loss_val = epitope_smoothness_loss(
                    p_epi, ag_edge_index, cfg.loss.node_prediction.smoothness_weight
                )

                node_loss = (node_loss + epitope_smoothness_loss_val )
                

        elif cfg.loss.node_prediction.task == "joint":
            y_epi = batch['ag_res'].y.float()
            p_epi = outputs['epitope_prob'].float()

            # Paratope loss
            y_para = batch['ab_res'].y.float()
            p_para = outputs['paratope_prob'].float()
            
            # choose either bce or focal as primary loss
            if cfg.loss.node_prediction.name == "focal":
                epitope_bce_loss = focal_criterion(
                    p_epi, y_epi,
                    pos_weight=cfg.loss.node_prediction.epi_pos_weight
                )
                paratope_bce_loss = focal_criterion(
                    p_para, y_para,
                    pos_weight=cfg.loss.node_prediction.para_pos_weight
                )
                node_loss = bce_weight * epitope_bce_loss + bce_weight * paratope_bce_loss
            elif cfg.loss.node_prediction.name == "bce":
                epitope_bce_loss = binary_cross_entropy(
                    p_epi, y_epi,
                    pos_weight=cfg.loss.node_prediction.epi_pos_weight
                )
                paratope_bce_loss = binary_cross_entropy(
                    p_para, y_para,
                    pos_weight=cfg.loss.node_prediction.para_pos_weight
                )
                node_loss = bce_weight * epitope_bce_loss + bce_weight * paratope_bce_loss

            else:
                raise ValueError(f"Unknown node_prediction.name: {cfg.loss.node_prediction.name}")
            
            # Add Dice loss for better imbalanced segmentation 
            if cfg.loss.node_prediction.dice_enabled:
                #    Enhancement: Use per-graph Dice if configured
                per_graph_dice = getattr(cfg.loss.node_prediction, 'dice_per_graph', False)
                ag_batch_indices = batch['ag_res'].batch if hasattr(batch['ag_res'], 'batch') else None
                ab_batch_indices = batch['ab_res'].batch if hasattr(batch['ab_res'], 'batch') else None
                
                epitope_dice_loss = dice_loss(p_epi, y_epi, ag_batch_indices, per_graph_dice)
                paratope_dice_loss = dice_loss(p_para, y_para, ab_batch_indices, per_graph_dice)

                node_loss  = (node_loss + dice_weight * (epitope_dice_loss + paratope_dice_loss))

            # Count regularizer for calibration
            if hasattr(cfg.loss, 'count_regularizer') and cfg.loss.node_prediction.count_regularizer_enabled:
                if cfg.loss.count_regularizer.per_graph_matching and hasattr(batch['ag_res'], 'batch'):
                    epitope_count_loss = per_graph_count_loss(
                        p_epi, y_epi, batch['ag_res'].batch, cfg.loss.count_regularizer.epitope_weight
                    )
                    paratope_count_loss = per_graph_count_loss(
                        p_para, y_para, batch['ab_res'].batch, cfg.loss.count_regularizer.paratope_weight
                    )
                node_loss = (node_loss + epitope_count_loss + paratope_count_loss)
                
            # Smoothness regularizer for contiguous patches
            if (hasattr(cfg.loss.node_prediction, 'smoothness_weight') and 
                cfg.loss.node_prediction.smoothness_weight > 0 and
                ('ag_res', 'r3', 'ag_res') in batch) and cfg.loss.node_prediction.smoothness_enabled:

                ag_edge_index = batch[('ag_res', 'r3', 'ag_res')].edge_index
                ab_edge_index = batch[('ab_res', 'r3', 'ab_res')].edge_index

                epitope_smoothness_loss_val = epitope_smoothness_loss(
                    p_epi, ag_edge_index, cfg.loss.node_prediction.smoothness_weight
                )
                paratope_smoothness_loss_val = epitope_smoothness_loss(
                    p_para, ab_edge_index, cfg.loss.node_prediction.smoothness_weight
                )

                node_loss = (node_loss + epitope_smoothness_loss_val + paratope_smoothness_loss_val)

        else:
            print("please specify the node prediction task: either epi-only or joint..")

        
        #  : Node-edge consistency loss (ties tasks together)
        if ('interaction_matrix' in outputs and 
            hasattr(cfg.loss.node_prediction, 'consistency_weight') and 
            cfg.loss.node_prediction.consistency_weight > 0 and
                cfg.loss.node_prediction.edge_node_consistency_enabled):

            p_para = outputs.get('paratope_prob', torch.zeros_like(p_epi))

            #    Fix: Proper batch access and pass model for learnable parameters
            ag_batch = getattr(batch.get('ag_res', None), 'batch', None) if 'ag_res' in batch else None
            ab_batch = getattr(batch.get('ab_res', None), 'batch', None) if 'ab_res' in batch else None
            
            consistency_loss_val = edge_node_consistency_loss(
                outputs['interaction_matrix'], p_epi, p_para,
                ag_batch, ab_batch, model, cfg.loss.node_prediction.consistency_weight
            )
    
    
            # Add consistency loss to total node loss
            node_loss = node_loss + consistency_loss_val

    # 2) Bipartite edge prediction loss (interactions)
    if 'interaction_matrix' in outputs and cfg.loss.edge_prediction.enabled:
        ag_batch = batch['ag_res'].batch
        ab_batch = batch['ab_res'].batch
        edge_index = batch[('ag_res', 'interacts', 'ab_res')].edge_index

        losses = []
        batch_size = int(ag_batch.max().item()) + 1 if ag_batch.numel() > 0 else 1
        
        pos_w = float(getattr(cfg.loss.edge_prediction, 'pos_weight', 1.0))

        for i in range(batch_size): # iterate over all complexex in the batch
            ag_mask = (ag_batch == i)
            ab_mask = (ab_batch == i)
            n_ag = int(ag_mask.sum().item())
            n_ab = int(ab_mask.sum().item())
            
            ag_global_indices = torch.where(ag_mask)[0]
            ab_global_indices = torch.where(ab_mask)[0]
            
            # Build local GT adjacency
            adj = torch.zeros((n_ag, n_ab), device=device)
            if edge_index.numel() > 0:
                ag_edges_mask = torch.isin(edge_index[0], ag_global_indices)
                ab_edges_mask = torch.isin(edge_index[1], ab_global_indices)
                valid_edges_mask = ag_edges_mask & ab_edges_mask
                if valid_edges_mask.any():
                    local_edges = edge_index[:, valid_edges_mask]
                    ag_global_to_local = {g.item(): l for l, g in enumerate(ag_global_indices)}
                    ab_global_to_local = {g.item(): l for l, g in enumerate(ab_global_indices)}
                    for e in range(local_edges.shape[1]):
                        ag_g = local_edges[0, e].item()
                        ab_g = local_edges[1, e].item()
                        if ag_g in ag_global_to_local and ab_g in ab_global_to_local:
                            adj[ag_global_to_local[ag_g], ab_global_to_local[ab_g]] = 1.0

            
            # Extract predicted submatrix from sparse matrix (same as distance loss)
            # The interaction matrix is 2D sparse: (total_ag, total_ab) with sub-matrices on diagonal
            ag_grid, ab_grid = torch.meshgrid(ag_global_indices, ab_global_indices, indexing='ij')
            pred_submatrix = outputs['interaction_matrix'][ag_grid, ab_grid]  # [n_ag, n_ab]
            
            if pred_submatrix.numel() > 0 and adj.numel() > 0:
                # Per-complex BCE loss with positive class reweighting
                if pos_w != 1.0:
                    w = torch.ones_like(adj, device=device)
                    w = torch.where(adj == 1.0, w * pos_w, w)
                    # graph_loss = F.binary_cross_entropy(pred_submatrix, adj, weight=w)
                    graph_loss = F.binary_cross_entropy_with_logits(pred_submatrix, adj, weight=w)
                else:
                    # graph_loss = F.binary_cross_entropy(pred_submatrix, adj)
                    graph_loss = F.binary_cross_entropy_with_logits(pred_submatrix, adj)
                losses.append(graph_loss)
            
            # print("edge loss per batch", losses, i)
        
        edge_loss = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)
        
        # Edge count regularizer: match predicted vs true interaction counts per graph
        if hasattr(cfg.loss, 'edge_count_regularizer') and cfg.loss.edge_count_regularizer.enabled:
            edge_count_reg_loss = per_graph_edge_count_loss(
                outputs['interaction_matrix'],
                batch[('ag_res', 'interacts', 'ab_res')].edge_index,
                ag_batch, ab_batch,
                cfg.loss.edge_count_regularizer.weight
            )
            edge_loss = edge_loss + edge_count_reg_loss

    else:
        edge_loss = torch.tensor(0.0, device=device)

    # 3) Contrastive loss (weight applied only once later)
    if cfg.loss.contrastive.enabled and cfg.loss.contrastive.name in ["gwnce", "infonce"]:
        ag_emb = outputs['ag_embed']
        ab_emb = outputs['ab_embed']
        y_ag = batch['ag_res'].y.long()
        y_ab = batch['ab_res'].y.long()
        
        if cfg.loss.contrastive.name == "infonce":
            intra_cl_loss = intra_nce_loss(ag_emb, ab_emb, y_ag, y_ab, cfg.loss.contrastive.temperature)
            inter_cl_loss = inter_nce_loss(ag_emb, ab_emb, y_ag, y_ab, cfg.loss.contrastive.temperature)
            # cl_loss = lambda_intra * intra_cl + lambda_inter * inter_cl
            cl_loss = cfg.loss.contrastive.intra_weight * intra_cl_loss + cfg.loss.contrastive.inter_weight * inter_cl_loss
            # Do NOT multiply by any weight here; applied in total_loss
        elif cfg.loss.contrastive.name == "gwnce":
            ei_ag = batch['ag_res', 'r3', 'ag_res'].edge_index 
            ei_ab = batch['ab_res', 'r3', 'ab_res'].edge_index
            ei_ag_ab = batch['ag_res', 'interacts', 'ab_res'].edge_index
            cl_loss = gradient_weighted_nce(
                ag_emb, ab_emb, 
                ei_ag, ei_ag_ab,
                temperature=cfg.loss.contrastive.temperature,
                cutrate=cfg.loss.gwnce.cut_rate,
                cutway=cfg.loss.gwnce.cut_way,
                mean=True
            )
            # Do NOT multiply by gwnce.weight here; use cfg.loss.contrastive.weight outside
        contrastive_loss = cl_loss
    else:
        contrastive_loss = torch.tensor(0.0, device=device)

    # 4) Auxiliary distance prediction loss for improved geometric learning
    auxiliary_dist_loss = torch.tensor(0.0, device=device)

    if (cfg.loss.auxiliary_distance.enabled and 
        'distance_logits' in outputs): 
        
        # Get positions and batch indices from HeteroData batch        
        ag_positions = batch['ag_res'].pos
        ab_positions = batch['ab_res'].pos
        ag_batch_indices = batch['ag_res'].batch
        ab_batch_indices = batch['ab_res'].batch
            
        # print("Computing auxiliary distance loss...")

        auxiliary_dist_loss = auxiliary_distance_prediction_loss(
            outputs['distance_logits'],
            ag_positions,
            ab_positions,
            ag_batch_indices,
            ab_batch_indices,
            cfg
        )

        node_loss = node_loss + ( cfg.loss.auxiliary_distance.weight * auxiliary_dist_loss)
        
        # print(f"Auxiliary distance loss: {auxiliary_dist_loss.item():.6f}")
            

    # Ensure tensors
    if not isinstance(node_loss, torch.Tensor):
        node_loss = torch.tensor(node_loss, device=device)
    if not isinstance(edge_loss, torch.Tensor):
        edge_loss = torch.tensor(edge_loss, device=device)
    if not isinstance(contrastive_loss, torch.Tensor):
        contrastive_loss = torch.tensor(contrastive_loss, device=device)

    loss_components = {
        'node_loss': node_loss,
        'edge_loss': edge_loss,
        'contrastive_loss': contrastive_loss,
        'auxiliary_dist_loss': auxiliary_dist_loss,
        'consistency_loss': consistency_loss_val,
        'epitope_bce_loss': epitope_bce_loss,
        'epitope_dice_loss': epitope_dice_loss,
        'epitope_smoothness_loss': epitope_smoothness_loss_val,
        'epitope_count_loss': epitope_count_loss,
        'paratope_bce_loss': paratope_bce_loss,
        'paratope_dice_loss': paratope_dice_loss,
        'paratope_smoothness_loss': paratope_smoothness_loss_val,
        'paratope_count_loss': paratope_count_loss
    }

    # Simple weighted combination (stable)
    total_loss = (
        cfg.loss.node_prediction.weight * loss_components['node_loss'] +
        cfg.loss.edge_prediction.weight * loss_components['edge_loss'] +
        cfg.loss.contrastive.weight * loss_components['contrastive_loss'] +
        loss_components['auxiliary_dist_loss']  # Already weighted in the function
    )
    
    #    Fix: Avoid L2 regularization double-counting
    # Use weight_decay in optimizer OR manual L2 reg, not both
    weight_decay = getattr(cfg.hparams.train, 'weight_decay', 0.0) if hasattr(cfg.hparams, 'train') else 0.0
    manual_l2_enabled = (hasattr(cfg.hparams, 'train') and 
                        hasattr(cfg.hparams.train, 'regularization') and 
                        cfg.hparams.train.regularization.use_l2_reg)
    
    # Only apply manual L2 if weight_decay is not used
    if manual_l2_enabled and weight_decay == 0.0:
        l2_reg = torch.tensor(0., device=device)
        for param in model.parameters():
            if param.requires_grad:
                l2_reg = l2_reg + param.pow(2).sum()
        reg_loss = cfg.hparams.train.regularization.l2_lambda * l2_reg
        total_loss = total_loss + reg_loss
        loss_components['reg_loss'] = reg_loss
    elif weight_decay > 0.0 and manual_l2_enabled:
        #    Warning: Both weight_decay and manual L2 enabled
        print(f"WARNING: Both weight_decay ({weight_decay}) and manual L2 reg enabled. Using weight_decay only.")
        loss_components['reg_loss'] = torch.tensor(0., device=device)
    else:
        loss_components['reg_loss'] = torch.tensor(0., device=device)
    
    # Force/geometric consistency loss (skip if positions are inputs without grad)
    force_accum = torch.tensor(0., device=device)
    if hasattr(cfg.loss, 'force') and getattr(cfg.loss.force, 'enabled', False):
        # Antigen
        if ('ag_atom' in batch and hasattr(batch['ag_atom'], 'pos') and
            ('ag_atom', 'atom_bond', 'ag_atom') in batch):
            ag_positions = batch['ag_atom'].pos
            ag_bonds = batch[('ag_atom', 'atom_bond', 'ag_atom')].edge_index
            if isinstance(ag_positions, torch.Tensor) and ag_positions.requires_grad:
                ag_force_loss = geometric_consistency_loss(
                    ag_positions, ag_bonds,
                    bond_weight=cfg.loss.force.bond_weight,
                    smooth_weight=cfg.loss.force.smooth_weight,
                    bond_tolerance=cfg.loss.force.bond_tolerance,
                    angle_tolerance=cfg.loss.force.angle_tolerance,
                    smooth_alpha=cfg.loss.force.smooth_alpha
                )
                force_accum = force_accum + ag_force_loss

        # Antibody
        if ('ab_atom' in batch and hasattr(batch['ab_atom'], 'pos') and
            ('ab_atom', 'atom_bond', 'ab_atom') in batch):
            ab_positions = batch['ab_atom'].pos
            ab_bonds = batch[('ab_atom', 'atom_bond', 'ab_atom')].edge_index
            if isinstance(ab_positions, torch.Tensor) and ab_positions.requires_grad:
                ab_force_loss = geometric_consistency_loss(
                    ab_positions, ab_bonds,
                    bond_weight=cfg.loss.force.bond_weight,
                    smooth_weight=cfg.loss.force.smooth_weight,
                    bond_tolerance=cfg.loss.force.bond_tolerance,
                    angle_tolerance=cfg.loss.force.angle_tolerance,
                    smooth_alpha=cfg.loss.force.smooth_alpha
                )
                force_accum = force_accum + ab_force_loss

        if force_accum.item() != 0.0:
            weighted_force_loss = cfg.loss.force.weight * force_accum
            total_loss = total_loss + weighted_force_loss
            loss_components['force_loss'] = weighted_force_loss
        else:
            loss_components['force_loss'] = torch.tensor(0., device=device)
    else:
        loss_components['force_loss'] = torch.tensor(0., device=device)
    
    if return_components:
        return total_loss, loss_components
    return total_loss


# -------------------------
# Supervised Loss Functions
# -------------------------

def binary_cross_entropy(pred: torch.Tensor,
                         target: torch.Tensor,
                         pos_weight=None) -> torch.Tensor:
    """
    BCE with positive class up-weighting on probabilities.

    BCE = - [ w_p * y * log(p) + (1-y) * log(1-p) ]
    
    Args:
        pred: Probabilities in [0,1]
        target: Ground-truth 0/1 labels
        pos_weight: Ratio (#neg / #pos) or weight for positive class
    
    Returns:
        Scalar mean loss
    """
    target = target.float()
    if pos_weight is None or float(pos_weight) == 1.0:
        return F.binary_cross_entropy(pred, target)

    if not torch.is_tensor(pos_weight):
        pos_weight = torch.tensor(float(pos_weight),
                                  dtype=pred.dtype,
                                  device=pred.device)

    w = torch.ones_like(target, dtype=pred.dtype, device=pred.device)
    w = torch.where(target == 1, pos_weight, w)

    return F.binary_cross_entropy(pred, target, weight=w)


# Binary focal loss with logits (not used in hierarchical_loss by default)
def focal_loss(preds, targets, alpha=0.25, gamma=2):
    bce = binary_cross_entropy_with_logits(preds, targets, reduction='none')
    pt = torch.exp(-bce)
    return alpha * (1 - pt) ** gamma * bce


# ---------------------------
# Contrastive Loss Functions
# ---------------------------

def ntxent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = torch.mm(z1, z2.t()) / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(logits, labels)

def intra_nce_loss_graph(h: torch.Tensor, y: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    pos = (y == 1).nonzero(as_tuple=True)[0]
    neg = (y == 0).nonzero(as_tuple=True)[0]
    if pos.numel() == 0:
        return torch.tensor(0., device=h.device)
    
    h_norm = F.normalize(h, dim=1)
    loss = 0.0
    for i in pos:
        anchor = h_norm[i].unsqueeze(0)
        pos_feats = h_norm[pos[pos != i]]
        neg_feats = h_norm[neg]
        
        if pos_feats.numel() == 0:
            continue
            
        sim_pos = torch.exp((anchor @ pos_feats.t()) / tau)
        sim_neg = torch.exp((anchor @ neg_feats.t()) / tau)
        num = sim_pos.sum()
        den = num + sim_neg.sum()
        loss += -torch.log(num / (den + 1e-8))
    
    return loss / pos.numel()

def intra_nce_loss(
    ag_h: torch.Tensor, ab_h: torch.Tensor,
    y_ag: torch.Tensor, y_ab: torch.Tensor,
    tau: float = 0.1
) -> torch.Tensor:
    return intra_nce_loss_graph(ag_h, y_ag, tau) + intra_nce_loss_graph(ab_h, y_ab, tau)

def inter_nce_loss(
    ag_h: torch.Tensor, ab_h: torch.Tensor,
    y_ag: torch.Tensor, y_ab: torch.Tensor,
    tau: float = 0.1
) -> torch.Tensor:
    pos_ag = (y_ag == 1).nonzero(as_tuple=True)[0]
    neg_ag = (y_ag == 0).nonzero(as_tuple=True)[0]
    pos_ab = (y_ab == 1).nonzero(as_tuple=True)[0]
    neg_ab = (y_ab == 0).nonzero(as_tuple=True)[0]

    ag_norm = F.normalize(ag_h, dim=1)
    ab_norm = F.normalize(ab_h, dim=1)

    def _one_direction(anchor_feats, pos_feats, neg_feats):
        if pos_feats.numel() == 0 or anchor_feats.numel() == 0:
            return torch.tensor(0., device=anchor_feats.device)
        loss = 0.0
        for i in range(anchor_feats.size(0)):
            a = anchor_feats[i].unsqueeze(0)
            sims_pos = torch.exp((a @ pos_feats.t()) / tau)
            sims_neg = torch.exp((a @ neg_feats.t()) / tau)
            num = sims_pos.sum()
            den = num + sims_neg.sum()
            loss += -torch.log(num / (den + 1e-8))
        return loss / anchor_feats.size(0)

    A2B = _one_direction(
        ag_norm[pos_ag],
        ab_norm[pos_ab],
        torch.cat([ag_norm[neg_ag], ab_norm[neg_ab]], dim=0)
    )
    B2A = _one_direction(
        ab_norm[pos_ab],
        ag_norm[pos_ag],
        torch.cat([ag_norm[neg_ag], ab_norm[neg_ab]], dim=0)
    )

    return A2B + B2A


# ---------------------------
# ReGCL Loss Implementation
# ---------------------------

def _sim(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    return z1 @ z2.t()

@torch.no_grad()
def _build_adj(edge_index: torch.LongTensor,
               rows: int, cols: int) -> torch.Tensor:
    device = edge_index.device
    if rows == cols:
        A = torch.zeros((rows, rows), dtype=torch.float32, device=device)
        if edge_index.numel() > 0:
            A[edge_index[0], edge_index[1]] = 1.0
        return A
    else:
        A = torch.zeros((rows, cols), dtype=torch.float32, device=device)
        if edge_index.numel() > 0:
            mask = (edge_index[0] < rows) & (edge_index[1] < cols)
            edge_index_masked = edge_index[:, mask]
            A[edge_index_masked[0], edge_index_masked[1]] = 1.0
        return A

def _get_W(z: torch.Tensor, edge_index: torch.LongTensor,
           mode: str, Pm: torch.Tensor,
           tau: float, cutrate: float, cutway: int) -> torch.Tensor:
    n, m = Pm.shape
    device = z.device
    
    A = _build_adj(edge_index, n, m)

    deg = (A.sum(1) + 1).sqrt().view(n, 1)
    P = (Pm / (Pm.sum(1, keepdim=True) + 1e-12)).detach()
    diag = (torch.diag(P) if n == m else (P*A).sum(1))
    P_bf = diag.view(n, 1).expand(n, n).detach()

    P = P / deg
    if n == m:
        P = P / deg.t()

    W = torch.zeros_like(P, device=device)

    if mode == 'between' and n == m:
        P12 = torch.sigmoid(-(P - P.mean())/(P.std()+1e-6)) * A

        sum2 = (A*P).sum(0).view(n,1)
        P3 = (diag.view(n,1)-1)/(deg**2)
        P23 = torch.sigmoid((sum2+P3 - (sum2+P3).mean())/((sum2+P3).std()+1e-6))

        W5k = 1/deg / deg.t()
        P5k = torch.sigmoid((torch.abs((P_bf-1)*W5k) -
                            torch.abs((P_bf-1)*W5k).mean())/
                            (torch.abs((P_bf-1)*W5k).std()+1e-6)) * A
        P5 = P5k.sum(0).view(n,1)

        mask = A @ A
        Pi = P * deg
        Pi = Pi * mask
        sum4 = Pi.sum(0).view(n,1)

        W += 0.5*(P12 + P5k)
        W += 0.5*(torch.diag((sum2+P3).squeeze()) +
                torch.diag((sum4+P5).squeeze()))
    else:
        P12 = torch.sigmoid(-(P - P.mean())/(P.std()+1e-6)) * A
        W += P12

    W = torch.where(W==0, torch.ones_like(W), W)
    return W

def _semi_loss(z1, z2, ei1, ei2, tau, cutrate, cutway):
    f = lambda s: torch.exp(s / tau)
    R_sim = f(_sim(z1, z1))
    B_sim = f(_sim(z1, z2))
    
    R_weighted = R_sim * _get_W(z1, ei1, 'refl', R_sim, tau, cutrate, cutway)
    B_weighted = B_sim * _get_W(z1, ei2, 'between', B_sim, tau, cutrate, cutway)

    N, M = B_sim.shape
    if N == M:
        pos = B_weighted.diag()
    else:
        A = _build_adj(ei2, N, M)
        pos = (B_weighted * A).sum(1)

    denom = R_weighted.sum(1) + B_weighted.sum(1) - (R_weighted.diag() if N==M else 0.)
    ratio = torch.clamp(pos / (denom + 1e-12), min=1e-8)
    return -torch.log(ratio)

def gradient_weighted_nce(z1: torch.Tensor, z2: torch.Tensor,
                          edge_index1: torch.LongTensor,
                          edge_index2: torch.LongTensor,
                          temperature: float,
                          cutrate: float, cutway: int,
                          mean: bool = True) -> torch.Tensor:
    same_size = z1.size(0) == z2.size(0)

    l1 = _semi_loss(z1, z2, edge_index1, edge_index2,
                    temperature, cutrate, cutway)

    if same_size:
        l2 = _semi_loss(z2, z1,
                        edge_index2,
                        edge_index1,
                        temperature, cutrate, cutway)
        loss = 0.5 * (l1 + l2)
    else:
        loss = l1
    return loss.mean() if mean else loss.sum()





# ---------------------------
# Force Loss Functions for Geometric Consistency
# ---------------------------

def bond_length_loss(positions, edge_index, target_lengths=None, tolerance=0.1):
    """
    Regularize bond lengths to maintain geometric consistency
    
    Args:
        positions: Atom positions (N, 3)
        edge_index: Edge indices for bonds (2, E)
        target_lengths: Target bond lengths (E,) or None for current distances
        tolerance: Tolerance for acceptable bond length deviation
    
    Returns:
        Bond length regularization loss
    """
    if edge_index.numel() == 0:
        return torch.tensor(0.0, device=positions.device)
    
    # Calculate current bond lengths
    src, dst = edge_index
    bond_vectors = positions[dst] - positions[src]
    current_lengths = torch.norm(bond_vectors, dim=1)
    
    if target_lengths is None:
        # Use current lengths as targets (maintain current geometry)
        target_lengths = current_lengths.detach()
    
    # Loss for deviations from target lengths
    length_diff = torch.abs(current_lengths - target_lengths)
    # Only penalize deviations beyond tolerance
    penalty = torch.relu(length_diff - tolerance)
    
    return penalty.mean()


def bond_angle_loss(positions, angle_indices, target_angles=None, tolerance=0.1):
    """
    Regularize bond angles for geometric consistency
    
    Args:
        positions: Atom positions (N, 3)
        angle_indices: Indices for angle triplets (3, A) - [i, j, k] for angle i-j-k
        target_angles: Target angles in radians (A,) or None for current angles
        tolerance: Tolerance for acceptable angle deviation in radians
    
    Returns:
        Bond angle regularization loss
    """
    if angle_indices.numel() == 0:
        return torch.tensor(0.0, device=positions.device)
    
    # Get positions for angle calculation
    i, j, k = angle_indices  # j is the central atom
    
    # Calculate vectors
    v1 = positions[i] - positions[j]  # j -> i
    v2 = positions[k] - positions[j]  # j -> k
    
    # Calculate angles using dot product
    v1_norm = F.normalize(v1, dim=1)
    v2_norm = F.normalize(v2, dim=1)
    cos_angles = torch.sum(v1_norm * v2_norm, dim=1)
    
    # Clamp to avoid numerical issues
    cos_angles = torch.clamp(cos_angles, -1.0, 1.0)
    current_angles = torch.acos(cos_angles)
    
    if target_angles is None:
        # Use current angles as targets
        target_angles = current_angles.detach()
    
    # Loss for deviations from target angles
    angle_diff = torch.abs(current_angles - target_angles)
    # Only penalize deviations beyond tolerance
    penalty = torch.relu(angle_diff - tolerance)
    
    return penalty.mean()


def coordinate_smoothness_loss(positions, edge_index, alpha=1.0):
    """
    Encourage smooth coordinate changes along the molecular graph
    
    Args:
        positions: Atom positions (N, 3)
        edge_index: Edge indices (2, E)
        alpha: Smoothness strength
    
    Returns:
        Coordinate smoothness loss
    """
    if edge_index.numel() == 0:
        return torch.tensor(0.0, device=positions.device)
    
    src, dst = edge_index
    
    # Calculate coordinate differences between connected atoms
    coord_diff = positions[dst] - positions[src]
    
    # Encourage smooth transitions (minimize large coordinate jumps)
    smoothness = torch.norm(coord_diff, dim=1) ** 2
    
    return alpha * smoothness.mean()


def geometric_consistency_loss(positions, edge_index, batch_idx=None, 
                             bond_weight=1.0, angle_weight=0.5, smooth_weight=0.1,
                             bond_tolerance=0.1, angle_tolerance=0.1, smooth_alpha=1.0):
    """
    Combined geometric consistency loss with multiple terms
    
    Args:
        positions: Atom positions (N, 3)
        edge_index: Edge indices (2, E)
        batch_idx: Batch indices for atoms (N,)
        bond_weight: Weight for bond length regularization
        angle_weight: Weight for bond angle regularization  
        smooth_weight: Weight for coordinate smoothness
        bond_tolerance: Tolerance for bond length deviations
        angle_tolerance: Tolerance for bond angle deviations
        smooth_alpha: Smoothness regularization strength
    
    Returns:
        Total geometric consistency loss
    """
    total_loss = torch.tensor(0.0, device=positions.device)
    
    # Bond length regularization
    if bond_weight > 0:
        bond_loss = bond_length_loss(positions, edge_index, tolerance=bond_tolerance)
        total_loss += bond_weight * bond_loss
    
    # Coordinate smoothness
    if smooth_weight > 0:
        smooth_loss = coordinate_smoothness_loss(positions, edge_index, alpha=smooth_alpha)
        total_loss += smooth_weight * smooth_loss
    
    # For angle loss, we'd need to construct angle indices from edge_index
    # This is more complex and depends on the molecular structure
    # For now, we'll skip angle loss unless specifically needed
    
    return total_loss


# ---------------------------
#  New Loss Functions for Epitope Prediction
# ---------------------------

def dice_loss(node_probs, node_labels, batch_indices=None, per_graph=False, eps=1e-8):
    """
    Dice loss for imbalanced node segmentation (PRIMARY LOSS for epitope prediction)
    
       Enhancement: Added per-graph option for varying graph sizes
    
    Dice = (2 * Σ p*y + ε) / (Σ p + Σ y + ε)
    
    Args:
        node_probs: Node probabilities (N,) in [0,1]
        node_labels: Ground truth labels (N,) in [0,1]
        batch_indices: Batch indices for nodes (N,) - required if per_graph=True
        per_graph: If True, compute Dice per graph and average (better for varying sizes)
        eps: Small constant for numerical stability
    
    Returns:
        Dice loss (1 - Dice coefficient)
    """
    if node_probs.numel() == 0:
        return torch.tensor(0.0, device=node_probs.device)
    
    if per_graph and batch_indices is not None:
        #    Fix: Per-graph Dice computation for varying graph sizes
        batch_size = batch_indices.max().item() + 1
        device = node_probs.device
        dice_losses = []
        
        for i in range(batch_size):
            mask = (batch_indices == i)
            if mask.sum() == 0:
                continue
                
            graph_probs = node_probs[mask]
            graph_labels = node_labels[mask]
            
            # Compute Dice for this graph
            intersection = 2.0 * (graph_probs * graph_labels).sum()
            union = graph_probs.sum() + graph_labels.sum() + eps
            
            if union > eps:  # Avoid division by zero
                dice_coeff = intersection / union
                dice_losses.append(1.0 - dice_coeff)
        
        if dice_losses:
            return torch.stack(dice_losses).mean()
        else:
            return torch.tensor(0.0, device=device)
    else:
        # Original global Dice computation
        intersection = 2.0 * (node_probs * node_labels).sum()
        union = node_probs.sum() + node_labels.sum() + eps
        dice_coeff = intersection / union
        
        # Return loss (1 - dice)
        return 1.0 - dice_coeff


def epitope_smoothness_loss(node_probs, edge_index, loss_weight=1e-3):
    """
    Laplacian smoothness regularizer on antigen residue graph
    
     : "Encourages contiguous epitope patches by penalizing probability differences"
    L_smooth = mean[(p_i - p_j)²] for connected residues
    
    Args:
        node_probs: Node probabilities (N,)
        edge_index: Edge indices for residue graph (2, E)
        loss_weight: Weight for smoothness loss
    
    Returns:
        Smoothness regularization loss
    """
    if node_probs.numel() == 0 or edge_index.numel() == 0:
        return torch.tensor(0.0, device=node_probs.device)
    
    src, dst = edge_index
    if src.max() >= node_probs.size(0) or dst.max() >= node_probs.size(0):
        return torch.tensor(0.0, device=node_probs.device)
    
    # Probability differences between connected residues
    prob_diff = (node_probs[src] - node_probs[dst]) ** 2
    smoothness_loss = prob_diff.mean()
    
    return smoothness_loss * loss_weight


def per_graph_count_loss(node_probs, node_labels, batch_indices, loss_weight=1.0):
    """
    Per-graph cardinality matching count loss 
    
     : "For each graph g in the batch, penalize deviation between predicted 
    and true positive node counts: |Σ p_i(g) - Σ y_i(g)|"
    
    Args:
        node_probs: Node probabilities (N,)
        node_labels: Ground truth labels (N,)  
        batch_indices: Batch indices for nodes (N,)
        loss_weight: Weight for the count loss
    
    Returns:
        Count regularization loss
    """
    if node_probs.numel() == 0 or batch_indices.numel() == 0:
        return torch.tensor(0.0, device=node_probs.device)
    
    batch_size = batch_indices.max().item() + 1
    device = node_probs.device
    
    # Sum probabilities and labels per graph using index_add

    """
        # Probabilities: Smooth gradients
        ∂L/∂p_i = ∂L/∂(Σp_i) × 1  # Always defined and smooth
        loss = |3.2 + 2.8 + 4.1 + ... - 15.0|  # Smooth changes

        # Binary labels: Problematic gradients  
        ∂L/∂p_i = ∂L/∂(Σ[p_i > t]) × ∂[p_i > t]/∂p_i  # Derivative is 0 almost everywhere
        loss = |0 + 1 + 1 + ... - 15|  # Jumps when predictions cross threshold
    """
    pred_sum = torch.zeros(batch_size, device=device).index_add(0, batch_indices, node_probs.float())
    true_sum = torch.zeros(batch_size, device=device).index_add(0, batch_indices, node_labels.float())
    
    # SmoothL1 loss for robustness (less sensitive to outliers than L2)
    count_loss = F.smooth_l1_loss(pred_sum, true_sum)
    
    return count_loss * loss_weight


def per_graph_edge_count_loss(interaction_matrix, ground_truth_edges, 
                             ag_batch, ab_batch, loss_weight=0.1):
    """
    Edge count regularizer: |Σ M_ij(g) - |E_true(g)|| per graph g
    
    Args:
        interaction_matrix: Sparse predicted interactions (total_ag, total_ab)
        ground_truth_edges: True edge indices for ('ag_res', 'interacts', 'ab_res')
        ag_batch: Batch indices for antigen residues (total_ag,)
        ab_batch: Batch indices for antibody residues (total_ab,)
        loss_weight: Weight for edge count loss
    
    Returns:
        Edge count regularization loss
    """
    if interaction_matrix.numel() == 0:
        return torch.tensor(0.0, device=interaction_matrix.device)
    
    device = interaction_matrix.device
    num_graphs = ag_batch.max().item() + 1
    
    pred_counts = []
    true_counts = []
    
    for i in range(num_graphs):
        # Get indices for this graph
        ag_indices = torch.where(ag_batch == i)[0]
        ab_indices = torch.where(ab_batch == i)[0]
        
        if ag_indices.numel() == 0 or ab_indices.numel() == 0:
            continue
            
        # Extract predicted interaction submatrix
        ag_grid, ab_grid = torch.meshgrid(ag_indices, ab_indices, indexing='ij')
        pred_submatrix = interaction_matrix[ag_grid, ab_grid]  # [n_ag_i, n_ab_i]
        
        # Count predicted edges (sum of interaction probabilities)
        pred_edge_count = pred_submatrix.sum()
        pred_counts.append(pred_edge_count)
        
        # Count true edges for this graph
        if ground_truth_edges.numel() > 0:
            ag_edges_mask = torch.isin(ground_truth_edges[0], ag_indices)
            ab_edges_mask = torch.isin(ground_truth_edges[1], ab_indices)
            valid_edges_mask = ag_edges_mask & ab_edges_mask
            true_edge_count = valid_edges_mask.sum().float()
        else:
            true_edge_count = torch.tensor(0.0, device=device)
            
        true_counts.append(true_edge_count)
    
    if not pred_counts:
        return torch.tensor(0.0, device=device)
    
    pred_counts = torch.stack(pred_counts)
    true_counts = torch.stack(true_counts)
    
    # Use smooth L1 loss (robust to outliers)
    edge_count_loss = F.smooth_l1_loss(pred_counts, true_counts)
    
    return edge_count_loss * loss_weight


def auxiliary_distance_prediction_loss(distance_logits, ag_positions, ab_positions, 
                                      ag_batch, ab_batch, cfg=None):
    """
    Auxiliary distance prediction loss following the hierarchical model structure.
    
    The distance_logits matrix is a sparse matrix with distance sub-matrices on the diagonal.
    We iterate through each graph in the batch, extract the corresponding positions and 
    distance logits, compute the true distance bins, and calculate cross-entropy loss.
    
    Args:
        distance_logits: Sparse distance logits matrix (total_ag, total_ab, 5)
        ag_positions: All antigen positions (total_ag, 3)
        ab_positions: All antibody positions (total_ab, 3)
        ag_batch: Batch indices for antigen residues (total_ag,)
        ab_batch: Batch indices for antibody residues (total_ab,)
        
    Returns:
        Average cross-entropy loss across all complexes in batch
    """
    if distance_logits.numel() == 0 or ag_positions.numel() == 0 or ab_positions.numel() == 0:
        return torch.tensor(0.0, device=distance_logits.device)
    
    device = distance_logits.device
    
    # Distance bin boundaries: <6, 6-8, 8-10, 10-12, >12 Å
    bin_edges = [0.0, 4.0, 8.0, 16.0, 32.0, float('inf')]
    
    # Get number of graphs in batch
    num_graphs = ag_batch.max().item() + 1
    
    total_loss = torch.tensor(0.0, device=device)
    valid_graphs = 0
    
    # Iterate through each graph in the batch 
    for i in range(num_graphs):
        # Get indices for this graph 
        ag_indices = torch.where(ag_batch == i)[0]
        ab_indices = torch.where(ab_batch == i)[0]
        
        # Skip if no residues in either antigen or antibody
        if ag_indices.numel() == 0 or ab_indices.numel() == 0:
            continue
            
        # Get positions for this graph
        ag_pos_i = ag_positions[ag_indices]  # [n_ag_i, 3]
        ab_pos_i = ab_positions[ab_indices]  # [n_ab_i, 3] 
        
        # Extract distance logits sub-matrix from sparse matrix
        ag_grid, ab_grid = torch.meshgrid(ag_indices, ab_indices, indexing='ij')
        dist_logits_i = distance_logits[ag_grid, ab_grid]  # [n_ag_i, n_ab_i, 5]
        
        # Compute true pairwise distances
        ag_pos_expanded = ag_pos_i.unsqueeze(1)  # [n_ag_i, 1, 3]
        ab_pos_expanded = ab_pos_i.unsqueeze(0)  # [1, n_ab_i, 3]
        dists = torch.norm(ag_pos_expanded - ab_pos_expanded, dim=2)  # [n_ag_i, n_ab_i]
        
        # Focus only on close pairs to avoid class imbalance
        max_dist = getattr(cfg.loss.auxiliary_distance, 'max_distance', 12.0)
        close_mask = dists <= max_dist  # Only pairs within max_distance
        
        if not close_mask.any():
            # No close pairs in this complex, skip
            continue
            
        # Filter to only close pairs
        close_dists = dists[close_mask]  # [n_close]
        close_logits = dist_logits_i[close_mask]  # [n_close, 5]
        
        # Convert close distances to bin labels (now only using first 4 bins: <6, 6-8, 8-10, 10-12)
        bin_labels = torch.zeros_like(close_dists, dtype=torch.long, device=device)
        for bin_idx, (low, high) in enumerate(zip(bin_edges[:-2], bin_edges[1:-1])):  # Exclude >12Å bin
            mask = (close_dists >= low) & (close_dists < high)
            bin_labels[mask] = bin_idx
        
        # Distance-aware cross-entropy loss with class balancing
        logits_flat = close_logits.view(-1, 5)[:, :4]  # [n_close, 4] - only use first 4 classes
        labels_flat = bin_labels.view(-1)       # [n_close]
        
        # Calculate class weights for balancing if enabled (only 4 classes now: <6, 6-8, 8-10, 10-12)
        class_weights = None
        if hasattr(cfg.loss.auxiliary_distance, 'class_balancing') and cfg.loss.auxiliary_distance.class_balancing:
            class_counts = torch.bincount(labels_flat, minlength=4).float()  # Only 4 classes now
            total_samples = labels_flat.numel()
            class_weights = total_samples / (4.0 * class_counts + 1e-8)  # +epsilon to avoid division by zero
        
        # Apply distance-aware weighting if enabled
        if hasattr(cfg.loss.auxiliary_distance, 'distance_weighting') and cfg.loss.auxiliary_distance.distance_weighting:
            # Inverse distance weighting: closer pairs get higher importance
            dist_weights = 1.0 / (close_dists + 1.0)  # [n_close], +1 to avoid division by zero
            # Normalize weights to have mean = 1 (preserves loss scale)
            dist_weights = dist_weights / dist_weights.mean()
            weights_flat = dist_weights.view(-1)  # [n_close]
            
            # Distance weighting with optional class balancing
            ce_losses = F.cross_entropy(logits_flat, labels_flat, weight=class_weights, reduction='none')
            graph_loss = (ce_losses * weights_flat).mean()
        else:
            # Standard cross-entropy with optional class balancing
            graph_loss = F.cross_entropy(logits_flat, labels_flat, weight=class_weights, reduction='mean')
        total_loss += graph_loss
        valid_graphs += 1

        """
        TODO:
        - return the total loss rather than averaging over the batch
        """
        
    # return total_loss
    
    # Average over all complexes in the batch
    if valid_graphs > 0:
        avg_loss = total_loss / valid_graphs
        return avg_loss 
    else:
        return torch.tensor(0.0, device=device)


def edge_node_consistency_loss(interaction_matrix, epitope_prob, paratope_prob, 
                             ag_batch=None, ab_batch=None, model=None, loss_weight=0.1):
    """
    Node-edge consistency loss with learnable scale/bias parameters
    
       Fix: "epitope_prob ≈ sigmoid(a * sum_j M_ij + b)" with learnable a,b
    This prevents scale mismatch between raw interaction mass and probabilities
    
    Args:
        interaction_matrix: Predicted interactions (B, N_ag, N_ab) or (N_ag, N_ab)
        epitope_prob: Epitope probabilities (N_ag,)
        paratope_prob: Paratope probabilities (N_ab,)
        ag_batch: Antigen batch indices (N_ag,) 
        ab_batch: Antibody batch indices (N_ab,)
        model: Model containing learnable consistency parameters
        loss_weight: Weight for consistency loss
    
    Returns:
        Consistency loss between edge predictions and node probabilities
    """
    if interaction_matrix.numel() == 0:
        return torch.tensor(0.0, device=interaction_matrix.device)
    
    device = interaction_matrix.device
    
    #    Fix: Initialize learnable scale/bias parameters if they don't exist
    if model is not None:
        if not hasattr(model, 'consist_scale'):
            model.consist_scale = nn.Parameter(torch.tensor(1.0, device=device))
            model.consist_bias = nn.Parameter(torch.tensor(0.0, device=device))
    
    # Handle batch dimension
    if interaction_matrix.dim() == 3:
        if interaction_matrix.shape[0] == 1:
            interaction_matrix = interaction_matrix[0]  # Single graph
            # Single graph case after batch reduction
            ag_interaction_mass = interaction_matrix.sum(dim=1)  # Sum over antibody nodes
            ab_interaction_mass = interaction_matrix.sum(dim=0)  # Sum over antigen nodes
        else:
            # Multiple graphs - flatten for consistency with node probs
            ag_interaction_mass = interaction_matrix.sum(dim=2).flatten()
            ab_interaction_mass = interaction_matrix.sum(dim=1).flatten()
    else:
        # Single graph case
        ag_interaction_mass = interaction_matrix.sum(dim=1)  # Sum over antibody nodes
        ab_interaction_mass = interaction_matrix.sum(dim=0)  # Sum over antigen nodes
    
    # Ensure shapes match
    if ag_interaction_mass.size(0) != epitope_prob.size(0):
        return torch.tensor(0.0, device=device)
    
    #    Fix: Use learnable mapping to probability space
    if model is not None and hasattr(model, 'consist_scale'):
        a = model.consist_scale
        b = model.consist_bias
        # Map interaction mass to probability space using sigmoid
        p_from_edges_ag = torch.sigmoid(a * ag_interaction_mass + b)
        
        # For paratope (if available)
        if ab_interaction_mass.size(0) == paratope_prob.size(0):
            p_from_edges_ab = torch.sigmoid(a * ab_interaction_mass + b)
            ab_consistency_loss = F.mse_loss(p_from_edges_ab, paratope_prob)
        else:
            ab_consistency_loss = torch.tensor(0.0, device=device)
        
        # MSE between mapped probabilities
        ag_consistency_loss = F.mse_loss(p_from_edges_ag, epitope_prob)
    else:
        # Fallback: normalize interaction mass to [0,1] range
        ag_mass_norm = torch.sigmoid(ag_interaction_mass / (ag_interaction_mass.max() + 1e-8))
        ag_consistency_loss = F.mse_loss(ag_mass_norm, epitope_prob)
        
        if ab_interaction_mass.size(0) == paratope_prob.size(0):
            ab_mass_norm = torch.sigmoid(ab_interaction_mass / (ab_interaction_mass.max() + 1e-8))
            ab_consistency_loss = F.mse_loss(ab_mass_norm, paratope_prob)
        else:
            ab_consistency_loss = torch.tensor(0.0, device=device)
    
    total_consistency_loss = ag_consistency_loss + ab_consistency_loss
    return total_consistency_loss * loss_weight

