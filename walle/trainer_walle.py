"""
V3 WALLE Trainer: Hierarchical Model with WALLE Loss
Implements the WALLE training strategy from the AsEP paper while maintaining
the V3 hierarchical architecture. Focuses on bipartite edge prediction and
epitope node classification using WALLE loss function.
"""

import os
import logging
os.environ["WANDB_SILENT"] = "true"
logging.getLogger("wandb").setLevel(logging.ERROR)

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
import gc
import numpy as np
from sklearn.model_selection import KFold, train_test_split as sk_train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
from torch_geometric.loader import DataLoader as PygDataLoader
from torch_geometric.data import Data
from tqdm import tqdm
import time
import csv
from pathlib import Path
import warnings
import random
warnings.filterwarnings("ignore")

# Core model components
from model.hierarchical_model import HierarchicalModel
from model.loss import walle_loss  # TODO: Use WALLE loss instead of hierarchical loss
from model.metric import HierarchicalMetrics
from model.callbacks import EarlyStopping, ModelCheckpoint

# Utilities
from utils import seed_everything, get_device, load_data, initialize_wandb, train_test_split
from data.data_splits import get_dataset_splits, apply_splits_to_dataset, validate_split_compatibility

torch.set_float32_matmul_precision("high")


class WALLEMetrics:
    """
    WALLE-specific metrics computation based on AsEP paper
    Focuses on bipartite edge prediction and epitope node classification
    """
    
    def __init__(self, edge_cutoff=3.0, device='cpu'):
        self.edge_cutoff = edge_cutoff
        self.device = device
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics"""
        # Per-batch metric storage
        self.edge_auprc = []
        self.edge_auroc = []
        self.edge_mcc = []
        self.edge_tp = []
        self.edge_fp = []
        self.edge_tn = []
        self.edge_fn = []
        
        self.epitope_auprc = []
        self.epitope_auroc = []
        self.epitope_mcc = []
        self.epitope_tp = []
        self.epitope_fp = []
        self.epitope_tn = []
        self.epitope_fn = []
    
    def update(self, outputs, batch, cfg):
        """
        Update metrics with batch predictions
        
        Args:
            outputs: Model outputs containing interaction_matrix
            batch: Batch data
            cfg: Configuration object
        """
        if 'interaction_matrix' not in outputs:
            return
        
        # Extract bipartite edge predictions and detach from computation graph
        edge_probs = outputs['interaction_matrix'].detach().cpu()
        
        # Get ground truth interactions and construct bipartite adjacency
        edge_labels = self._construct_edge_labels(batch, edge_probs.shape).detach().cpu()
        
        # Get edge threshold from config or use default
        edge_threshold = getattr(cfg.loss.walle, 'edge_threshold', 0.1)
        
        # Process each graph in the batch separately
        batch_size = edge_probs.shape[0]
        for i in range(batch_size):
            # Process edge metrics for this graph
            edge_probs_i = edge_probs[i].flatten().numpy()
            edge_labels_i = edge_labels[i].flatten().numpy()
            
            # Handle edge metrics
            if len(np.unique(edge_labels_i)) > 1:
                edge_auprc_i = average_precision_score(edge_labels_i, edge_probs_i)
                edge_auroc_i = roc_auc_score(edge_labels_i, edge_probs_i)
            else:
                edge_auprc_i = 0.0
                edge_auroc_i = 0.5
            
            # Binarize predictions
            edge_preds_i = (edge_probs_i > edge_threshold).astype(int)
            
            # Compute confusion matrix
            edge_tp_i = np.sum((edge_preds_i == 1) & (edge_labels_i == 1))
            edge_fp_i = np.sum((edge_preds_i == 1) & (edge_labels_i == 0))
            edge_tn_i = np.sum((edge_preds_i == 0) & (edge_labels_i == 0))
            edge_fn_i = np.sum((edge_preds_i == 0) & (edge_labels_i == 1))
            
            # Compute MCC
            if edge_tp_i + edge_fp_i + edge_tn_i + edge_fn_i == 0:
                edge_mcc_i = 0.0
            else:
                edge_mcc_i = matthews_corrcoef(edge_labels_i, edge_preds_i)
            
            # Store edge metrics for this graph
            self.edge_auprc.append(edge_auprc_i)
            self.edge_auroc.append(edge_auroc_i)
            self.edge_mcc.append(edge_mcc_i)
            self.edge_tp.append(edge_tp_i)
            self.edge_fp.append(edge_fp_i)
            self.edge_tn.append(edge_tn_i)
            self.edge_fn.append(edge_fn_i)
            
            # Process epitope metrics for this graph
            n_ag = edge_probs[i].shape[0]
            n_ab = edge_probs[i].shape[1]
            edge_probs_i_mat = edge_probs[i].numpy().reshape(n_ag, n_ab)
            edge_labels_i_mat = edge_labels[i].numpy().reshape(n_ag, n_ab)
            
            # Compute epitope scores and labels
            epitope_prob_scores = edge_probs_i_mat.sum(axis=1)
            edge_count_true = edge_labels_i_mat.sum(axis=1)
            epitope_labels_i = (edge_count_true > 0).astype(float)
            
            # Handle epitope metrics
            if len(np.unique(epitope_labels_i)) > 1:
                epitope_auprc_i = average_precision_score(epitope_labels_i, epitope_prob_scores)
                epitope_auroc_i = roc_auc_score(epitope_labels_i, epitope_prob_scores)
            else:
                epitope_auprc_i = 0.0
                epitope_auroc_i = 0.5
            
            # Binarize epitope predictions
            epitope_preds_binary = (epitope_prob_scores > self.edge_cutoff).astype(int)
            
            # Compute confusion matrix
            epitope_tp_i = np.sum((epitope_preds_binary == 1) & (epitope_labels_i == 1))
            epitope_fp_i = np.sum((epitope_preds_binary == 1) & (epitope_labels_i == 0))
            epitope_tn_i = np.sum((epitope_preds_binary == 0) & (epitope_labels_i == 0))
            epitope_fn_i = np.sum((epitope_preds_binary == 0) & (epitope_labels_i == 1))
            
            # Compute MCC
            if epitope_tp_i + epitope_fp_i + epitope_tn_i + epitope_fn_i == 0:
                epitope_mcc_i = 0.0
            else:
                epitope_mcc_i = matthews_corrcoef(epitope_labels_i, epitope_preds_binary)
            
            # Store epitope metrics for this graph
            self.epitope_auprc.append(epitope_auprc_i)
            self.epitope_auroc.append(epitope_auroc_i)
            self.epitope_mcc.append(epitope_mcc_i)
            self.epitope_tp.append(epitope_tp_i)
            self.epitope_fp.append(epitope_fp_i)
            self.epitope_tn.append(epitope_tn_i)
            self.epitope_fn.append(epitope_fn_i)
    
    def _construct_edge_labels(self, batch, pred_shape):
        """Construct ground truth bipartite adjacency matrix"""
        # NOTE: This is similar to the logic in walle_loss function
        ag_batch = batch['ag_res'].batch if hasattr(batch['ag_res'], 'batch') else torch.zeros(len(batch['ag_res'].y), device=self.device, dtype=torch.long)
        ab_batch = batch['ab_res'].batch if hasattr(batch['ab_res'], 'batch') else torch.zeros(len(batch['ab_res'].y), device=self.device, dtype=torch.long)
        edge_index = batch[('ag_res', 'interacts', 'ab_res')].edge_index
        
        if len(pred_shape) == 3:
            batch_size, n_ag_total, n_ab_total = pred_shape
        else:
            batch_size = 1
            n_ag_total, n_ab_total = pred_shape
        
        edge_labels = torch.zeros(pred_shape, device=self.device)
        
        for i in range(batch_size):
            ag_mask = (ag_batch == i)
            ab_mask = (ab_batch == i)
            ag_global_indices = torch.where(ag_mask)[0]
            ab_global_indices = torch.where(ab_mask)[0]
            
            if edge_index.numel() > 0:
                ag_edges_mask = torch.isin(edge_index[0], ag_global_indices)
                ab_edges_mask = torch.isin(edge_index[1], ab_global_indices)
                valid_edges_mask = ag_edges_mask & ab_edges_mask
                
                if valid_edges_mask.any():
                    local_edges = edge_index[:, valid_edges_mask]
                    ag_global_to_local = {global_idx.item(): local_idx for local_idx, global_idx in enumerate(ag_global_indices)}
                    ab_global_to_local = {global_idx.item(): local_idx for local_idx, global_idx in enumerate(ab_global_indices)}
                    
                    for edge_idx in range(local_edges.shape[1]):
                        ag_global = local_edges[0, edge_idx].item()
                        ab_global = local_edges[1, edge_idx].item()
                        
                        if ag_global in ag_global_to_local and ab_global in ab_global_to_local:
                            ag_local = ag_global_to_local[ag_global]
                            ab_local = ab_global_to_local[ab_global]
                            if len(pred_shape) == 3:
                                edge_labels[i, ag_local, ab_local] = 1.0
                            else:
                                edge_labels[ag_local, ab_local] = 1.0
        
        return edge_labels
    
    def compute(self, cfg):
        """
        Compute final metrics from all accumulated predictions
        Returns metrics similar to WALLE paper evaluation
        """
        # If no graphs were processed, return zeros
        if not self.edge_auprc:
            return {
                'edge_auprc': 0.0,
                'edge_auroc': 0.5,
                'edge_mcc': 0.0,
                'edge_tp': 0.0,
                'edge_fp': 0.0,
                'edge_tn': 0.0,
                'edge_fn': 0.0,
                'epitope_auprc': 0.0,
                'epitope_auroc': 0.5,
                'epitope_mcc': 0.0,
                'epitope_tp': 0.0,
                'epitope_fp': 0.0,
                'epitope_tn': 0.0,
                'epitope_fn': 0.0,
            }
        
        return {
            # Edge-level metrics (averaged per graph)
            'edge_auprc': np.mean(self.edge_auprc),
            'edge_auroc': np.mean(self.edge_auroc),
            'edge_mcc': np.mean(self.edge_mcc),
            'edge_tp': np.mean(self.edge_tp),
            'edge_fp': np.mean(self.edge_fp),
            'edge_tn': np.mean(self.edge_tn),
            'edge_fn': np.mean(self.edge_fn),
            
            # Epitope node-level metrics (averaged per graph)
            'epitope_auprc': np.mean(self.epitope_auprc),
            'epitope_auroc': np.mean(self.epitope_auroc),
            'epitope_mcc': np.mean(self.epitope_mcc),
            'epitope_tp': np.mean(self.epitope_tp),
            'epitope_fp': np.mean(self.epitope_fp),
            'epitope_tn': np.mean(self.epitope_tn),
            'epitope_fn': np.mean(self.epitope_fn),
        }


def create_dataloader(dataset, cfg, shuffle=True):
    """Create PyG DataLoader with proper batching"""
    return PygDataLoader(
        dataset,
        batch_size=cfg.hparams.train.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_threads,
        pin_memory=False,
        follow_batch=['ag_res', 'ab_res']
    )


def train_epoch(model, loader, optimizer, device, metrics, cfg, epoch):
    """Training epoch with WALLE loss"""
    model.train()
    total_loss = 0
    metrics.reset()
    loss_components = {}

    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        outputs = model(batch)
        
        # TODO: Use WALLE loss instead of hierarchical loss
        loss, comp = walle_loss(
            outputs, 
            batch, 
            device, 
            cfg,
            model=model,
            epoch=epoch,
            return_components=True
        )
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.hparams.train.grad_clip)
        optimizer.step()

        total_loss += loss.item()
        for k, v in comp.items():
            loss_components[k] = loss_components.get(k, 0) + (v.item() if hasattr(v, 'item') else v)
        
        # Update WALLE metrics - pass cfg to update
        metrics.update(outputs, batch, cfg)

    for k in loss_components:
        loss_components[k] /= len(loader)
    return total_loss / len(loader), metrics.compute(cfg), loss_components


def validate_epoch(model, loader, device, metrics, cfg, epoch, update_threshold=False):
    """Validation epoch with WALLE loss and metrics"""
    model.eval()
    total_loss = 0
    metrics.reset()
    loss_components = {}
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            batch = batch.to(device)
            outputs = model(batch)
            
            # TODO: Use WALLE loss for validation
            loss, comp = walle_loss(
                outputs, 
                batch, 
                device, 
                cfg,
                model=model,
                epoch=epoch,
                return_components=True
            )
            
            total_loss += loss.item()
            for k, v in comp.items():
                loss_components[k] = loss_components.get(k, 0) + (v.item() if hasattr(v, 'item') else v)
            
            # Update WALLE metrics - pass cfg to update
            metrics.update(outputs, batch, cfg)
        
        for k in loss_components:
            loss_components[k] /= len(loader)
        
    return total_loss / len(loader), metrics.compute(cfg), loss_components


def get_run_id(cfg):
    """Generate unique run ID"""
    if cfg.get("run_id"):
        return cfg.run_id
    return f"v3_walle_{cfg.model.name}_{time.strftime('%Y%m%d-%H%M%S')}"


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Main training function with WALLE strategy"""
    start_time = time.time()
    device = get_device(cfg)
    
    run_id = get_run_id(cfg)
    cfg.run_id = run_id
    
    print(f"V3 WALLE Training Pipeline - Run ID: {run_id}")
    print(f"Using device: {device}")
    print("NOTE: Using WALLE loss function and metrics from AsEP paper")
    
    seed_everything(cfg.seed)
    
    # TODO: Enable WALLE loss in configuration
    if not getattr(cfg.loss.walle, 'enabled', False):
        print("WARNING: WALLE loss is not enabled in configuration. Set loss.walle.enabled=true")
        cfg.loss.walle.enabled = True
        print("→ Automatically enabled WALLE loss for this run")
    
    # Load dataset
    proj_dir = Path(hydra.utils.get_original_cwd()) / "../../../../"
    dataset_path = proj_dir / "data/asep/m3epi" / cfg.dataset.tensor
    print(f"Loading dataset from: {dataset_path}")
    full_data = load_data(str(dataset_path))
    print(f"Dataset contains {len(full_data)} complexes")
    
    # Development mode subset
    if cfg.mode.mode == "dev":
        full_data = full_data[:cfg.mode.data.dev_subset]
    
    # Data splitting
    print("\n=== Data Split ===")
    print(f"Split method: {cfg.dataset.split.method}")
    
    splits = get_dataset_splits(len(full_data), cfg.dataset.split)
    validate_split_compatibility(len(full_data), splits, cfg.dataset.split.method)
    train_data, valid_data, test_data = apply_splits_to_dataset(full_data, splits)

    del full_data
    gc.collect()
    
    print(f"Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")
    
    # For backward compatibility, merge train+valid for cross-validation if no separate validation
    if cfg.mode.mode == "train" and len(valid_data) > 0:
        print("Train mode: Using train+valid for cross-validation, test set held out")
        cv_data = train_data + valid_data
    else:
        cv_data = train_data
    
    # Initialize wandb
    if cfg.logging_method == "wandb":
        initialize_wandb(cfg)
    
    # Initialize checkpointing
    checkpoint_dir = Path(cfg.callbacks.model_checkpoint.dirpath) / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Cross-validation setup
    if cfg.mode.mode == "train":
        print("\n=== Training Mode (K-Fold Cross-Validation) ===")
        kf = KFold(n_splits=cfg.hparams.train.kfolds, shuffle=True, random_state=cfg.seed)
        cv_splits = list(kf.split(cv_data))
    elif cfg.mode.mode == "test" or cfg.mode.mode == "dev":
        print("\n=== Test Mode (Single Train-Valid Split) ===")
        if len(valid_data) == 0:
            from sklearn.model_selection import train_test_split as sk_split
            train_subset, valid_subset = sk_split(train_data, test_size=0.2, random_state=cfg.seed)
            cv_splits = [(list(range(len(train_subset))), list(range(len(train_subset), len(train_subset) + len(valid_subset))))]
            cv_data = train_subset + valid_subset
        else:
            train_indices = list(range(len(train_data)))
            valid_indices = list(range(len(train_data), len(train_data) + len(valid_data)))
            cv_splits = [(train_indices, valid_indices)]
            cv_data = train_data + valid_data
    # else:
    #     print(f"\n=== Development Mode ===")
    #     kf = KFold(n_splits=cfg.hparams.train.kfolds, shuffle=True, random_state=cfg.seed)
        # cv_splits = list(kf.split(cv_data))
    
    all_results = []
    
    # FIX: Initialize WALLE metrics with edge cutoff from config (use same as loss function)
    edge_cutoff = getattr(cfg.loss.walle, 'edge_cutoff', 2.5)  # FIX: Match loss function default
    print(f"WALLE Edge Cutoff for Epitope Classification: {edge_cutoff}")
    metrics = WALLEMetrics(edge_cutoff=edge_cutoff, device=device)
    
    model_ckpt = ModelCheckpoint(
        dirpath=cfg.callbacks.model_checkpoint.dirpath,
        run_id=run_id,
        filename=cfg.model.name + "_walle",  # Add walle suffix
        monitor="val_loss",
        mode="min",
        save_top_k=cfg.callbacks.model_checkpoint.save_top_k,
        config=cfg
    )
    
    # Cross-validation training with WALLE approach
    num_folds = len(cv_splits)
    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        if cfg.mode.mode == "test":
            print(f"\n▶︎ Training on predefined train/valid split (WALLE approach)")
        else:
            print(f"\n▶︎ Fold {fold_idx+1}/{num_folds} (WALLE approach)")
        
        # Create fold datasets
        train_subset = [cv_data[i] for i in train_idx]
        val_subset = [cv_data[i] for i in val_idx]
        
        train_loader = create_dataloader(train_subset, cfg, shuffle=True)
        val_loader = create_dataloader(val_subset, cfg, shuffle=False)
        
        # Initialize model
        model = HierarchicalModel(cfg).to(device)
        
        optimizer = Adam(
            model.parameters(),
            lr=cfg.hparams.train.learning_rate,
            weight_decay=cfg.hparams.train.weight_decay
        )
        
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=True
        )
        
        # Initialize callbacks
        es = EarlyStopping(**cfg.callbacks.early_stopping)
        best_val_loss = float('inf')
        best_val_metrics = None
        
        # Training loop
        for epoch in range(1, cfg.hparams.train.num_epochs + 1):
            epoch_start = time.time()
            
            # FIX: Reset metrics for each epoch to avoid tensor state issues
            metrics.reset()
            
            train_loss, train_metrics, train_loss_comp = train_epoch(
                model, train_loader, optimizer, device, metrics, cfg, epoch
            )
            
            # FIX: Reset metrics again before validation
            metrics.reset()
            
            val_loss, val_metrics, val_loss_comp = validate_epoch(
                model, val_loader, device, metrics, cfg, epoch
            )
            
            epoch_time = time.time() - epoch_start
            
            # Log to wandb
            if cfg.logging_method == "wandb":
                log_data = {
                    "run_id": run_id,
                    "fold": fold_idx,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "epoch_time": epoch_time,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                    **{f"train_{k}": v for k, v in train_loss_comp.items()},
                    **{f"val_{k}": v for k, v in val_loss_comp.items()}
                }
                wandb.log(log_data)
            
            # Print epoch summary (WALLE-style metrics)
            print(f"Epoch {epoch:03d} | Time: {epoch_time:.1f}s | LR: {optimizer.param_groups[0]['lr']:.2e}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # FIX: Add detailed loss component logging
            print("=== Loss Components ===")
            print(f"Train: " + " | ".join([f"{k}: {v:.4f}" for k, v in train_loss_comp.items()]))
            print(f"Val: " + " | ".join([f"{k}: {v:.4f}" for k, v in val_loss_comp.items()]))
            
            print("Train Edge Metrics: " + " | ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items() if k.startswith('edge_')]))
            print("Train Epitope Metrics: " + " | ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items() if k.startswith('epitope_')]))
            print("Val Edge Metrics: " + " | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items() if k.startswith('edge_')]))
            print("Val Epitope Metrics: " + " | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items() if k.startswith('epitope_')]))
            
            # Save checkpoint
            model_ckpt.save(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                fold=fold_idx,
                value=val_loss,
                other_states={
                    'early_stopping': es.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_val_metrics': best_val_metrics,
                    'walle_edge_cutoff': edge_cutoff
                }
            )
            
            # Update best model and early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_metrics = val_metrics
            
            scheduler.step(val_loss)
            
            # Early stopping
            if es(val_loss):
                print(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Store best results from this fold
        if best_val_metrics:
            final_metrics = {
                k: (v.item() if hasattr(v, 'item') else v)
                for k, v in best_val_metrics.items()
            }
            all_results.append(final_metrics)
    
    # Results reporting
    if all_results:
        avg = {k: np.mean([m[k] for m in all_results]) for k in all_results[0]}
        std = {k: np.std([m[k] for m in all_results]) for k in all_results[0]}
        
        if cfg.mode.mode == "train":
            print("\n=== WALLE Cross-Validation Results ===")
            for k in avg:
                print(f"{k}: {avg[k]:.4f} ± {std[k]:.4f}")
        elif cfg.mode.mode == "test":
            print("\n=== WALLE Train-Valid Results ===")
            for k in avg:
                print(f"{k}: {avg[k]:.4f}")
        else:
            print("\n=== WALLE Development Results ===")
            for k in avg:
                print(f"{k}: {avg[k]:.4f} ± {std[k]:.4f}")

        if cfg.logging_method == "wandb":
            wandb.log({f"cv_{k}": avg[k] for k in avg})
            if cfg.mode.mode != "test":
                wandb.log({f"cv_{k}_std": std[k] for k in std})
    
    # Final test evaluation
    print("\n=== Final Test Evaluation (WALLE) ===")
    
    # Load best model
    model, optimizer, scheduler = model_ckpt.load_best_model(model, optimizer, scheduler, device)
    
    test_loader = create_dataloader(test_data, cfg, shuffle=False)
    test_loss, test_metrics, test_loss_comp = validate_epoch(
        model, test_loader, device, metrics, cfg, epoch=999
    )
    
    print(f"WALLE Edge Cutoff: {edge_cutoff}")
    print(f"Test Loss: {test_loss:.4f}")
    print("=== Edge-level Metrics (Bipartite Graph Link Prediction) ===")
    for k, v in test_metrics.items():
        if k.startswith('edge_'):
            print(f"{k}: {v:.4f}")
    print("=== Epitope Node-level Metrics ===")
    for k, v in test_metrics.items():
        if k.startswith('epitope_'):
            print(f"{k}: {v:.4f}")
    
    # Log test results to wandb
    if cfg.logging_method == "wandb":
        wandb.log({
            "walle_edge_cutoff": edge_cutoff,
            "test_loss": test_loss,
            **{f"test_{k}": v for k, v in test_metrics.items()},
            **{f"test_{k}": v for k, v in test_loss_comp.items()}
        })
    
    # Save test results summary
    summary_dir = checkpoint_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    test_summary_file = summary_dir / f"{cfg.model.name}_walle_test_results.csv"
    
    all_metrics = {}
    for k, v in test_metrics.items():
        if hasattr(v, 'item'):
            all_metrics[k] = v.item()
        else:
            all_metrics[k] = v
    
    all_metrics["total_time_s"] = time.time() - start_time
    all_metrics["walle_edge_cutoff"] = edge_cutoff
    all_metrics["walle_loss_enabled"] = True
    
    with open(test_summary_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(all_metrics.keys())
        writer.writerow([f"{v:.4f}" if isinstance(v, float) else v 
                        for v in all_metrics.values()])
    
    print(f"→ WALLE test summary saved to {test_summary_file}")
    
    # Cleanup
    if cfg.logging_method == "wandb":
        wandb.finish()
    
    print(f"\n✓ V3 WALLE training pipeline completed in {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()