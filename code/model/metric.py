import torch
import torchmetrics
# FIXME: Removed unused Metric import
from typing import Dict, List

class HierarchicalMetrics(torch.nn.Module):
    def __init__(self, epi_threshold=0.3, para_threshold=0.3, walle_edge_cutoff=3.39):
        super().__init__()
        """
        - Compute classification metrics for node classification task
        - Uses probabilities and true labels to compute AUC and AUPRC
        - For metrics like F1, precision, recall, accuracy, and MCC, converts
          probabilities to binary predictions using configurable thresholds
        - WALLE edge cutoff for epitope classification via edge sum thresholding
        """
        self.epi_threshold = epi_threshold
        self.para_threshold = para_threshold
        self.walle_edge_cutoff = walle_edge_cutoff
        
        # Node-level metrics for epitope
        self.epitope_metrics = torchmetrics.MetricCollection({
            'epitope_auc': torchmetrics.AUROC(task='binary'),
            'epitope_auprc': torchmetrics.AveragePrecision(task='binary'),
            'epitope_f1': torchmetrics.F1Score(task='binary', threshold=epi_threshold),
            'epitope_precision': torchmetrics.Precision(task='binary', threshold=epi_threshold),
            'epitope_recall': torchmetrics.Recall(task='binary', threshold=epi_threshold),
            'epitope_accuracy': torchmetrics.Accuracy(task='binary', threshold=epi_threshold),
            'epitope_mcc': torchmetrics.MatthewsCorrCoef(task='binary', num_classes=2, threshold=epi_threshold),
        })
        
        self.epitope_confmat = torchmetrics.ConfusionMatrix(task='binary', num_classes=2, threshold=epi_threshold)
        
        # Node-level metrics for paratope
        self.paratope_metrics = torchmetrics.MetricCollection({
            'paratope_auc': torchmetrics.AUROC(task='binary'),
            'paratope_auprc': torchmetrics.AveragePrecision(task='binary'),
            'paratope_f1': torchmetrics.F1Score(task='binary', threshold=para_threshold),
            'paratope_precision': torchmetrics.Precision(task='binary', threshold=para_threshold),
            'paratope_recall': torchmetrics.Recall(task='binary', threshold=para_threshold),
            'paratope_accuracy': torchmetrics.Accuracy(task='binary', threshold=para_threshold),
            'paratope_mcc': torchmetrics.MatthewsCorrCoef(task='binary', num_classes=2, threshold=para_threshold),
        })
        
        self.paratope_confmat = torchmetrics.ConfusionMatrix(task='binary', num_classes=2, threshold=para_threshold)
        
        # Edge-level metrics for interaction prediction (fixed 0.3 threshold)
        self.edge_metrics = torchmetrics.MetricCollection({
            'edge_auc': torchmetrics.AUROC(task='binary'),
            'edge_auprc': torchmetrics.AveragePrecision(task='binary'),
            'edge_f1': torchmetrics.F1Score(task='binary', threshold=epi_threshold),
            'edge_precision': torchmetrics.Precision(task='binary', threshold=epi_threshold),
            'edge_recall': torchmetrics.Recall(task='binary', threshold=epi_threshold),
            'edge_accuracy': torchmetrics.Accuracy(task='binary', threshold=epi_threshold),
        })
        self.edge_confmat = torchmetrics.ConfusionMatrix(task='binary', num_classes=2, threshold=epi_threshold)
        
        # WALLE FIX: Add per-complex metric storage
        self.per_complex_epitope_metrics = []
        self.per_complex_paratope_metrics = []
        
        # Initialize all metrics
        # reset metrics when we resume training, torchmetrics loads the initial thresholds by default
        self.reset()

    def to(self, device):
        self.epitope_metrics.to(device)
        self.paratope_metrics.to(device)
        self.edge_metrics.to(device)
        # TODO: Move confusion matrix to device
        self.epitope_confmat.to(device)
        self.paratope_confmat.to(device)
        self.edge_confmat.to(device)
        return self

    # TODO: [WALLE FIX] Implement per-complex MCC calculation as per WALLE approach
    # Current implementation computes global MCC which is biased by large complexes
    def update(self, outputs, batch):
        # Update epitope metrics with probabilities
        self.epitope_metrics.update(
            outputs['epitope_prob'], 
            batch['ag_res'].y.long()
        )
        
        # Update paratope metrics with probabilities
        self.paratope_metrics.update(
            outputs['paratope_prob'], 
            batch['ab_res'].y.long()
        )
        
        # Update confusion matrices with probabilities
        self.epitope_confmat.update(
            outputs['epitope_prob'], 
            batch['ag_res'].y.long()
        )
        self.paratope_confmat.update(
            outputs['paratope_prob'], 
            batch['ab_res'].y.long()
        )
        
        # Update edge metrics
        edge_preds, edge_labels = self._extract_edge_data_from_batch(outputs, batch)
        if edge_preds is not None and edge_labels is not None:
            self.edge_metrics.update(edge_preds, edge_labels.long())
            self.edge_confmat.update(edge_preds, edge_labels.long())
        
        # WALLE FIX: Compute per-complex metrics using PyG batch tensor
        self._update_per_complex_metrics(outputs, batch)
    
    

    def compute(self) -> Dict[str, torch.Tensor]:
        metrics = {}
        metrics.update(self.epitope_metrics.compute())
        metrics.update(self.paratope_metrics.compute())
        
        # Only compute edge metrics if data was collected
        try:
            metrics.update(self.edge_metrics.compute())
        except ValueError:
            # No edge data collected - skip edge metrics
            pass
        
        # Extract confusion matrix components (TP, FP, TN, FN) for epitope, paratope, and edges
        # Confusion matrix format: [[TN, FP], [FN, TP]]
        epitope_cm = self.epitope_confmat.compute()
        paratope_cm = self.paratope_confmat.compute()
        
        # Extract individual components for logging and tracking
        metrics['epitope_tn'] = epitope_cm[0, 0].float()  # True Negatives
        metrics['epitope_fp'] = epitope_cm[0, 1].float()  # False Positives  
        metrics['epitope_fn'] = epitope_cm[1, 0].float()  # False Negatives
        metrics['epitope_tp'] = epitope_cm[1, 1].float()  # True Positives
        
        metrics['paratope_tn'] = paratope_cm[0, 0].float()  # True Negatives
        metrics['paratope_fp'] = paratope_cm[0, 1].float()  # False Positives
        metrics['paratope_fn'] = paratope_cm[1, 0].float()  # False Negatives  
        metrics['paratope_tp'] = paratope_cm[1, 1].float()  # True Positives
        
        # Edge confusion matrix components (only if edge data was collected)
        try:
            edge_cm = self.edge_confmat.compute()
            edge_tp = edge_cm[1, 1].float()  # True Positives
            edge_fp = edge_cm[0, 1].float()  # False Positives
            edge_fn = edge_cm[1, 0].float()  # False Negatives
            edge_tn = edge_cm[0, 0].float()  # True Negatives
            
            metrics['edge_tn'] = edge_tn
            metrics['edge_fp'] = edge_fp
            metrics['edge_fn'] = edge_fn
            metrics['edge_tp'] = edge_tp
            
            # Manual MCC computation to avoid torchmetrics bug
            eps = 1e-8
            mcc_denom = torch.sqrt((edge_tp + edge_fp + eps) * (edge_tp + edge_fn + eps) * 
                                  (edge_tn + edge_fp + eps) * (edge_tn + edge_fn + eps))
            if mcc_denom > eps:
                edge_mcc = (edge_tp * edge_tn - edge_fp * edge_fn) / mcc_denom
            else:
                edge_mcc = torch.tensor(0.0)
            metrics['edge_mcc'] = edge_mcc
            
        except ValueError:
            # No edge data collected - skip edge confusion matrix
            pass
        
        # WALLE FIX: Add per-complex metric averaging (regular threshold-based)
        if self.per_complex_epitope_metrics:
            epitope_avg = self._average_per_complex_metrics(self.per_complex_epitope_metrics)
            paratope_avg = self._average_per_complex_metrics(self.per_complex_paratope_metrics)
            
            # Add per-complex averaged metrics with walle_ prefix for regular threshold-based metrics
            for key, value in epitope_avg.items():
                metrics[f'epitope_walle_{key}'] = value
            for key, value in paratope_avg.items():
                metrics[f'paratope_walle_{key}'] = value
        
        # WALLE-specific metrics using exact paper approach (edge_cutoff-based)
        if hasattr(self, 'per_complex_epitope_walle_metrics') and self.per_complex_epitope_walle_metrics:
            epitope_walle_avg = self._average_per_complex_metrics(self.per_complex_epitope_walle_metrics)
            paratope_walle_avg = self._average_per_complex_metrics(self.per_complex_paratope_walle_metrics)
            
            # Override walle_ metrics with true WALLE approach (edge_cutoff-based)
            for key, value in epitope_walle_avg.items():
                metrics[f'epitope_walle_{key}'] = value
            for key, value in paratope_walle_avg.items():
                metrics[f'paratope_walle_{key}'] = value
        
        return metrics

    def reset(self):
        self.epitope_metrics.reset()
        self.paratope_metrics.reset()
        self.edge_metrics.reset()
        self.epitope_confmat.reset()
        self.paratope_confmat.reset()
        self.edge_confmat.reset()
        # WALLE FIX: Reset per-complex storage
        self.per_complex_epitope_metrics.clear()
        self.per_complex_paratope_metrics.clear()
        # Reset WALLE-specific metrics storage
        if hasattr(self, 'per_complex_epitope_walle_metrics'):
            self.per_complex_epitope_walle_metrics.clear()
            self.per_complex_paratope_walle_metrics.clear()

    def _extract_edge_data_from_batch(self, outputs, batch):
        """Extract flattened edge predictions and ground truth"""
        if 'interaction_matrix' not in outputs:
            return None, None
        
        device = outputs['interaction_matrix'].device
        ag_batch = batch['ag_res'].batch
        ab_batch = batch['ab_res'].batch
        edge_index = batch[('ag_res', 'interacts', 'ab_res')].edge_index
        
        all_edge_preds = []
        all_edge_labels = []
        
        batch_size = int(ag_batch.max().item()) + 1 if ag_batch.numel() > 0 else 1
        
        for i in range(batch_size):
            ag_indices = torch.where(ag_batch == i)[0]
            ab_indices = torch.where(ab_batch == i)[0]
            
            # Extract predicted submatrix
            ag_grid, ab_grid = torch.meshgrid(ag_indices, ab_indices, indexing='ij')
            pred_submatrix = outputs['interaction_matrix'][ag_grid, ab_grid]
            
            # Build ground truth adjacency
            adj = torch.zeros_like(pred_submatrix, device=device)
            if edge_index.numel() > 0:
                ag_edges_mask = torch.isin(edge_index[0], ag_indices)
                ab_edges_mask = torch.isin(edge_index[1], ab_indices)
                valid_edges_mask = ag_edges_mask & ab_edges_mask
                
                if valid_edges_mask.any():
                    local_edges = edge_index[:, valid_edges_mask]
                    ag_global_to_local = {g.item(): l for l, g in enumerate(ag_indices)}
                    ab_global_to_local = {g.item(): l for l, g in enumerate(ab_indices)}
                    
                    for e in range(local_edges.shape[1]):
                        ag_g = local_edges[0, e].item()
                        ab_g = local_edges[1, e].item()
                        if ag_g in ag_global_to_local and ab_g in ab_global_to_local:
                            adj[ag_global_to_local[ag_g], ab_global_to_local[ab_g]] = 1.0
            
            all_edge_preds.append(pred_submatrix.flatten())
            all_edge_labels.append(adj.flatten())
        
        return torch.cat(all_edge_preds) if all_edge_preds else None, \
               torch.cat(all_edge_labels) if all_edge_labels else None

    def _update_per_complex_metrics(self, outputs, batch):
        """WALLE FIX: Compute metrics per complex using WALLE's exact thresholding approach"""
        # Get WALLE edge cutoff from config or use default
        edge_cutoff = getattr(self, 'walle_edge_cutoff', 3.39)
        
        # Check if we have interaction matrix for WALLE evaluation
        if 'interaction_matrix' in outputs:
            self._update_walle_metrics(outputs, batch, edge_cutoff)
        
        # Also compute regular threshold-based metrics
        if hasattr(batch['ag_res'], 'ptr') and hasattr(batch['ab_res'], 'ptr'):
            ag_ptr = batch['ag_res'].ptr
            ab_ptr = batch['ab_res'].ptr
            
            # Process each complex in the batch
            for i in range(len(ag_ptr) - 1):
                # Extract antigen complex data
                ag_start, ag_end = ag_ptr[i], ag_ptr[i+1]
                epi_probs = outputs['epitope_prob'][ag_start:ag_end]
                epi_labels = batch['ag_res'].y[ag_start:ag_end].long()
                
                # Extract antibody complex data
                ab_start, ab_end = ab_ptr[i], ab_ptr[i+1]
                para_probs = outputs['paratope_prob'][ab_start:ab_end]
                para_labels = batch['ab_res'].y[ab_start:ab_end].long()
                
                # Compute per-complex metrics
                epi_metrics = self._compute_complex_metrics(epi_probs, epi_labels, self.epi_threshold)
                para_metrics = self._compute_complex_metrics(para_probs, para_labels, self.para_threshold)
                
                self.per_complex_epitope_metrics.append(epi_metrics)
                self.per_complex_paratope_metrics.append(para_metrics)
        else:
            # Handle single complex case
            epi_metrics = self._compute_complex_metrics(
                outputs['epitope_prob'], batch['ag_res'].y.long(), self.epi_threshold
            )
            para_metrics = self._compute_complex_metrics(
                outputs['paratope_prob'], batch['ab_res'].y.long(), self.para_threshold
            )
            
            self.per_complex_epitope_metrics.append(epi_metrics)
            self.per_complex_paratope_metrics.append(para_metrics)
    
    def _update_walle_metrics(self, outputs, batch, edge_cutoff):
        """WALLE's exact two-threshold approach: 0.3 for edges + edge_cutoff for epitopes"""
        interaction_matrix = outputs['interaction_matrix']
        
        # Get batch information
        ag_batch = batch['ag_res'].batch if hasattr(batch['ag_res'], 'batch') else torch.zeros(len(batch['ag_res'].y), device=interaction_matrix.device, dtype=torch.long)
        ab_batch = batch['ab_res'].batch if hasattr(batch['ab_res'], 'batch') else torch.zeros(len(batch['ab_res'].y), device=interaction_matrix.device, dtype=torch.long)
        batch_size = ag_batch.max().item() + 1 if ag_batch.numel() > 0 else 1
        
        for i in range(batch_size):
            # Get masks for current complex
            ag_mask = (ag_batch == i)
            ab_mask = (ab_batch == i)
            
            # Extract interaction submatrix for this complex
            ag_indices = torch.where(ag_mask)[0]
            ab_indices = torch.where(ab_mask)[0]
            
            if interaction_matrix.dim() == 3:
                # Batch format: [batch_size, n_ag_total, n_ab_total]
                if i < interaction_matrix.shape[0]:
                    submatrix = interaction_matrix[i, ag_indices][:, ab_indices]
                else:
                    continue
            elif interaction_matrix.dim() == 2:
                # Single batch format: [n_ag_total, n_ab_total]
                submatrix = interaction_matrix[ag_indices][:, ab_indices]
            else:
                continue
                
            # WALLE Step 1: Epitope classification via SUM of probabilities > edge_cutoff
            # Paper implementation: sum of interaction probabilities, not count of binary edges
            epitope_prob_scores = submatrix.sum(dim=1)  # SUM of probabilities for each antigen node  
            epitope_predictions_walle = (epitope_prob_scores > edge_cutoff).long()
            
            
            # Get true labels
            epitope_labels = batch['ag_res'].y[ag_mask].long()
            paratope_labels = batch['ab_res'].y[ab_mask].long()
            
            # Compute WALLE-specific metrics using exact paper approach
            epi_walle_metrics = self._compute_complex_metrics(
                epitope_predictions_walle.float(), epitope_labels, threshold=0.3  # Already binary
            )
            
            # For paratope, use similar logic: sum probabilities over antigen nodes
            paratope_prob_scores = submatrix.sum(dim=0)  # SUM of probabilities for each antibody node  
            paratope_predictions_walle = (paratope_prob_scores > edge_cutoff).long()
            
            para_walle_metrics = self._compute_complex_metrics(
                paratope_predictions_walle.float(), paratope_labels, threshold=0.3  # Already binary
            )
            
            # Store WALLE metrics separately
            if not hasattr(self, 'per_complex_epitope_walle_metrics'):
                self.per_complex_epitope_walle_metrics = []
                self.per_complex_paratope_walle_metrics = []
                
            self.per_complex_epitope_walle_metrics.append(epi_walle_metrics)
            self.per_complex_paratope_walle_metrics.append(para_walle_metrics)
    
    def _compute_complex_metrics(self, probs, labels, threshold):
        """WALLE FIX: Compute metrics for a single complex"""
        preds = (probs > threshold).long()
        
        # Compute confusion matrix components
        tp = ((preds == 1) & (labels == 1)).sum().float()
        fp = ((preds == 1) & (labels == 0)).sum().float()
        tn = ((preds == 0) & (labels == 0)).sum().float()
        fn = ((preds == 0) & (labels == 1)).sum().float()
        
        # Compute metrics with safety checks
        eps = 1e-8
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
        
        # MCC computation with safety check for zero denominator
        denom = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn) / (denom + eps) if denom > eps else torch.tensor(0.0, device=probs.device)
        
        return {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'mcc': mcc,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        }
    
    def _average_per_complex_metrics(self, complex_metrics_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """WALLE FIX: Average metrics across complexes"""
        if not complex_metrics_list:
            return {}
        
        keys = complex_metrics_list[0].keys()
        averaged = {}
        
        for key in keys:
            values = [metrics[key] for metrics in complex_metrics_list]
            averaged[key] = torch.mean(torch.stack(values))
        
        return averaged




# OLD IMPLEMENTATION - COMMENTED OUT


# """
# - Comprehensive Metrics for Hierarchical Model Evaluation
# - Now uses model's thresholded predictions
# """
# import torch
# import torchmetrics
# from torchmetrics import Metric
# from typing import Dict

# class HierarchicalMetrics(torch.nn.Module):
#     def __init__(self, epi_threshold=0.3, para_threshold=0.3):
#         super().__init__()
#         """
#         - Compute classification metrics for node classification task
#         - Uses probabilities and true labels to compute AUC and AUPRC
#         - For metrics like F1, precision, recall, accuracy, and MCC, converts
#           probabilities to binary predictions using configurable thresholds
#         """
#         self.epi_threshold = epi_threshold
#         self.para_threshold = para_threshold
        
#         # Node-level metrics for epitope
#         self.epitope_metrics = torchmetrics.MetricCollection({
#             'epitope_auc': torchmetrics.AUROC(task='binary'),
#             'epitope_auprc': torchmetrics.AveragePrecision(task='binary'),
#             'epitope_f1': torchmetrics.F1Score(task='binary', threshold=epi_threshold),
#             'epitope_precision': torchmetrics.Precision(task='binary', threshold=epi_threshold),
#             'epitope_recall': torchmetrics.Recall(task='binary', threshold=epi_threshold),
#             'epitope_accuracy': torchmetrics.Accuracy(task='binary', threshold=epi_threshold),
#             'epitope_mcc': torchmetrics.MatthewsCorrCoef(task='binary', num_classes=2, threshold=epi_threshold),
#         })
        
#         self.epitope_confmat = torchmetrics.ConfusionMatrix(task='binary', num_classes=2, threshold=epi_threshold)
        
#         # Node-level metrics for paratope
#         self.paratope_metrics = torchmetrics.MetricCollection({
#             'paratope_auc': torchmetrics.AUROC(task='binary'),
#             'paratope_auprc': torchmetrics.AveragePrecision(task='binary'),
#             'paratope_f1': torchmetrics.F1Score(task='binary', threshold=para_threshold),
#             'paratope_precision': torchmetrics.Precision(task='binary', threshold=para_threshold),
#             'paratope_recall': torchmetrics.Recall(task='binary', threshold=para_threshold),
#             'paratope_accuracy': torchmetrics.Accuracy(task='binary', threshold=para_threshold),
#             'paratope_mcc': torchmetrics.MatthewsCorrCoef(task='binary', num_classes=2, threshold=para_threshold),
#         })
        
#         self.paratope_confmat = torchmetrics.ConfusionMatrix(task='binary', num_classes=2, threshold=para_threshold)
        
#         # Initialize all metrics
#         # reset metrics when we resume training, torchmetrics loads the initial thresholds by default
#         self.reset()

#     def to(self, device):
#         self.epitope_metrics.to(device)
#         self.paratope_metrics.to(device)
#         # TODO: Move confusion matrix to device
#         self.epitope_confmat.to(device)
#         self.paratope_confmat.to(device)
#         return self

#     def update(self, outputs, batch):








# """
# - Comprehensive Metrics for Hierarchical Model Evaluation
# - Includes MCC and all essential classification metrics
# - Includes metrics computation for node classification and edge prediction tasks
# """

# import torch
# import torchmetrics
# from torchmetrics import Metric
# import numpy as np
# from typing import Dict

# class HierarchicalMetrics(torch.nn.Module):
#     def __init__(self, epi_threshold=0.3, para_threshold=0.3):
#         super().__init__()
#         """
#         - compute the classification metrics for the node classification task
#         - uses the probabilities and the true labels to compute AUC and AUPRC
#         - for metrics like F1, precision, recall, accuracy, and MCC, we convert 
#         the probabilities to binary predictions using a threshold

#         """

#         self.epi_threshold = epi_threshold
#         self.para_threshold = para_threshold

        
#         # Node-level metrics for epitope
#         self.epitope_metrics = torchmetrics.MetricCollection({
#             'epitope_auc': torchmetrics.AUROC(task='binary'),
#             'epitope_auprc': torchmetrics.AveragePrecision(task='binary'), # auprc
#             'epitope_f1': torchmetrics.F1Score(task='binary', threshold=epi_threshold),
#             'epitope_precision': torchmetrics.Precision(task='binary', threshold=epi_threshold),
#             'epitope_recall': torchmetrics.Recall(task='binary', threshold=epi_threshold),
#             'epitope_accuracy': torchmetrics.Accuracy(task='binary', threshold=epi_threshold),
#             'epitope_mcc': torchmetrics.MatthewsCorrCoef(task='binary', num_classes=2, threshold=epi_threshold),
#         })
        
#         # Node-level metrics for paratope
#         self.paratope_metrics = torchmetrics.MetricCollection({
#             'paratope_auc': torchmetrics.AUROC(task='binary'),
#             'paratope_auprc': torchmetrics.AveragePrecision(task='binary'),
#             'paratope_f1': torchmetrics.F1Score(task='binary', threshold=para_threshold),
#             'paratope_precision': torchmetrics.Precision(task='binary', threshold=para_threshold),
#             'paratope_recall': torchmetrics.Recall(task='binary', threshold=para_threshold),
#             'paratope_accuracy': torchmetrics.Accuracy(task='binary', threshold=para_threshold),
#             'paratope_mcc': torchmetrics.MatthewsCorrCoef(task='binary', num_classes=2, threshold=para_threshold),
#         })
        
#         # Interaction-level metrics
#         self.interaction_metrics = torchmetrics.MetricCollection({
#             'interaction_auc': torchmetrics.AUROC(task='binary'),
#             'interaction_auprc': torchmetrics.AveragePrecision(task='binary'),
#             'interaction_f1': torchmetrics.F1Score(task='binary'),
#             'interaction_precision': torchmetrics.Precision(task='binary'),
#             'interaction_recall': torchmetrics.Recall(task='binary'),
#             'interaction_accuracy': torchmetrics.Accuracy(task='binary'),
#             'interaction_mcc': torchmetrics.MatthewsCorrCoef(task='binary', num_classes=2),
#         })
        
#         # Geometric metrics
#         self.coord_metrics = torchmetrics.MetricCollection({
#             'coord_rmsd': torchmetrics.MeanSquaredError(squared=False),
#         })
        
#         # Initialize all metrics
#         self.reset()

#     def to(self, device):
#         """Move all metrics to specified device"""
#         self.epitope_metrics.to(device)
#         self.paratope_metrics.to(device)
#         self.interaction_metrics.to(device)
#         self.coord_metrics.to(device)
#         return self

#     def update(self, outputs, batch):
#         # Update epitope metrics
#         self.epitope_metrics.update(
#             outputs['epitope_prob'], 
#             batch['ag_res'].y.long()
#         )
        
#         # Update paratope metrics
#         self.paratope_metrics.update(
#             outputs['paratope_prob'], 
#             batch['ab_res'].y.long()
#         )
        
#         # Update interaction metrics
#         if 'interaction_probs' in outputs:
#             edge_index = batch[('ag_res', 'interacts', 'ab_res')].edge_index
#             adj = torch.zeros(
#                 (batch['ag_res'].num_nodes, batch['ab_res'].num_nodes),
#                 device=outputs['interaction_probs'].device
#             )
#             adj[edge_index[0], edge_index[1]] = 1.0
#             self.interaction_metrics.update(
#                 outputs['interaction_probs'].flatten(),
#                 adj.flatten().long()
#             )
        
#         # Update coordinate metrics
#         if 'ag_coords' in outputs:
#             self.coord_metrics.update(
#                 outputs['ag_coords'],
#                 batch['ag_res'].pos
#             )

#     def compute(self) -> Dict[str, torch.Tensor]:
#         metrics = {}
#         metrics.update(self.epitope_metrics.compute())
#         metrics.update(self.paratope_metrics.compute())
#         # metrics.update(self.interaction_metrics.compute())
#         # metrics.update(self.coord_metrics.compute())
#         return metrics

#     def reset(self):
#         self.epitope_metrics.reset()
#         self.paratope_metrics.reset()
#         self.interaction_metrics.reset()
#         self.coord_metrics.reset()


# class EdgePredictionMetrics(Metric):
#     """Custom metrics for bipartite edge prediction"""
#     def __init__(self):
#         super().__init__()
#         self.add_state("correct_edges", default=torch.tensor(0), dist_reduce_fx="sum")
#         self.add_state("total_edges", default=torch.tensor(0), dist_reduce_fx="sum")
#         self.add_state("true_positives", default=torch.tensor(0), dist_reduce_fx="sum")
#         self.add_state("false_positives", default=torch.tensor(0), dist_reduce_fx="sum")
#         self.add_state("false_negatives", default=torch.tensor(0), dist_reduce_fx="sum")
        
#     def update(self, preds, targets):
#         # Threshold predictions

#         # convert the probabilities to binary predictions using a threshold
#         preds_binary = (preds > 0.3).float()
        
#         # Calculate metrics
#         self.correct_edges += torch.sum(preds_binary == targets)
#         self.total_edges += targets.numel()
#         self.true_positives += torch.sum((preds_binary == 1) & (targets == 1))
#         self.false_positives += torch.sum((preds_binary == 1) & (targets == 0))
#         self.false_negatives += torch.sum((preds_binary == 0) & (targets == 1))
        
#     def compute(self):
#         precision = self.true_positives / (self.true_positives + self.false_positives + 1e-8)
#         recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-8)
#         f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
#         return {
#             'edge_accuracy': self.correct_edges.float() / self.total_edges,
#             'edge_precision': precision,
#             'edge_recall': recall,
#             'edge_f1': f1
#         }

