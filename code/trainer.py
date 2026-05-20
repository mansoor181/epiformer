"""
Implements unified pre-training of the entire model (AG encoder + AB encoder + decoder)
with projection heads from 3D-EMGP, GearNet, and CL-GNN, then fine-tuning after discarding heads.
"""
import os
import gc
import csv
import time
import random
import logging
import warnings
from pathlib import Path

os.environ["WANDB_SILENT"] = "true"
logging.getLogger("wandb").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold, train_test_split as sk_train_test_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, OneCycleLR
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PygDataLoader
from tqdm import tqdm

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# Core model components
from model.loss import hierarchical_loss
from model.metric import HierarchicalMetrics
from model.callbacks import EarlyStopping, ModelCheckpoint


# Utilities
from utils import (seed_everything, get_device, load_data, initialize_wandb, train_test_split,
                  get_data_splits_by_mode, save_embedding_plots, compute_simple_debug_stats)
# TODO: Added data splitting utilities for AsEP paper compatibility
from data.data_splits import get_dataset_splits, apply_splits_to_dataset, validate_split_compatibility


from model.epiformer import EpiformerModel


torch.set_float32_matmul_precision("high")


import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"



# def create_dataloader(dataset, cfg, batch_size, shuffle=True):
#     """Create PyG DataLoader with proper batching"""
#     return PygDataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         num_workers=cfg.num_threads,
#         pin_memory=False,
#         follow_batch=['ag_res', 'ab_res']
#     )


def seed_worker(worker_id):
    import numpy as np, random, torch
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_dataloader(dataset, cfg, batch_size, shuffle=True):
    g = torch.Generator()
    g.manual_seed(cfg.seed)  # fixed across runs
    return PygDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_threads,
        pin_memory=False,
        follow_batch=['ag_res', 'ab_res'],
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=False,  # optional: disable to simplify determinism
    )



def pretrain_epoch(model, loader, optimizer, device, cfg, epoch):
    """
    pre-training loop with joint loss computation
    [Inference] Single ResMP task + joint decoder loss per batch
    """
    model.train()
    total_loss = 0
    loss_components = {}

    for batch in tqdm(loader, desc=f"Pre-training Epoch {epoch}"):
        batch = batch.to(device)
        
        # Compute joint loss: encoder_loss + decoder_loss
        batch_loss, batch_components = compute_joint_pretrain_loss(model, batch, cfg, device)
        
        # Backward pass
        optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.hparams.train.grad_clip)
        optimizer.step()
        
        total_loss += batch_loss.item()
        for k, v in batch_components.items():
            loss_components[k] = loss_components.get(k, 0) + v

    for k in loss_components:
        loss_components[k] /= len(loader)
    return total_loss / len(loader), loss_components


def train_epoch(model, loader, optimizer, device, metrics, cfg, epoch):
    """Fine-tuning epoch with optional debug monitoring"""
    model.train()
    total_loss = 0
    metrics.reset()
    loss_components = {}

    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        outputs = model(batch)
        
        loss, comp = hierarchical_loss(
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
            loss_components[k] = loss_components.get(k, 0) + v
        metrics.update(outputs, batch)
        

    for k in loss_components:
        loss_components[k] /= len(loader)
    return total_loss / len(loader), metrics.compute(), loss_components


def validate_epoch(model, loader, device, metrics, cfg, epoch):
    """Validation epoch with fixed thresholds"""
    model.eval()
    total_loss = 0
    metrics.reset()
    loss_components = {}
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            batch = batch.to(device)
            outputs = model(batch)

            loss, comp = hierarchical_loss(
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
                loss_components[k] = loss_components.get(k, 0) + v
            
            # Update metrics
            metrics.update(outputs, batch)
        
        # Compute metrics with fixed thresholds
        final_metrics = metrics.compute()
        
        for k in loss_components:
            loss_components[k] /= len(loader)
        
    return total_loss / len(loader), final_metrics, loss_components





def get_run_id(cfg):
    """Generate unique run ID"""
    # if cfg.get("run_id"):
    #     return cfg.run_id
    return f"{cfg.dataset.graph_type}_{time.strftime('%Y%m%d-%H%M%S')}"


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Main training function with end-to-end pre-training"""
    start_time = time.time()
    device = get_device(cfg)
    # print(cfg)
    
    ### need three paths: data tensor path, results checkpoint path, and splits path
    proj_dir = Path(hydra.utils.get_original_cwd()) / "../../../../.." / cfg.base_dir

    # Handle resume functionality
    checkpoint = None
    if cfg.resume:
        if cfg.get("run_id"):
            run_id = cfg.run_id
        checkpoint_dir = proj_dir / Path(cfg.callbacks.model_checkpoint.dirpath) / run_id

        checkpoint_path = checkpoint_dir / f"{cfg.model.name}_last.pt"
        
        if checkpoint_path.exists():
            print(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = ModelCheckpoint.load_checkpoint(checkpoint_path, device)
            # Use seed from checkpoint for reproducibility
            seed = checkpoint['config'].get('seed', cfg.seed)
        else:
            print(f"No checkpoint found for run {run_id}, starting from scratch")
            seed = cfg.seed
    else:
        seed = cfg.seed
        run_id = get_run_id(cfg)
        cfg.run_id = run_id

    
    print(f"V3 End-to-End Pre-training Pipeline - Run ID: {run_id}")
    print(f"Using device: {device}")
    
    seed_everything(seed)  # Use the correct seed variable

    # print(proj_dir)
    
    # Select dataset based on model graph type
    graph_type = cfg.dataset.graph_type
    
    if graph_type == "raad-plm":
        # RAAD with PLM-specific embedding
        plm_type = cfg.dataset.plm_type
        dataset_filename = getattr(cfg.dataset, plm_type)
        print(f"Using RAAD dataset with {plm_type}: {dataset_filename}")
    
    dataset_path = proj_dir / "data/asep/m3epi" / dataset_filename

    print(f"Loading dataset from: {dataset_path}")
    full_data = load_data(str(dataset_path))
    print(f"Dataset contains {len(full_data)} complexes")
    
    # Mode-specific data handling
    if cfg.mode.mode == "dev":
        full_data = full_data[:cfg.mode.data.dev_subset]
    
    # TODO: STEP 1: Enhanced Data Split with AsEP paper compatibility
    print("\n=== STEP 1: Data Split ===")
    print(f"Split method: {cfg.dataset.split.method}")
    
    # Get splits based on configuration
    split_dict_path = proj_dir / cfg.dataset.split.split_dict_path
    splits = get_dataset_splits(split_dict_path, len(full_data), cfg.dataset.split)
    
    # Validate splits
    validate_split_compatibility(len(full_data), splits, cfg.dataset.split.method)
    
    # Apply splits to dataset
    train_data, valid_data, test_data = apply_splits_to_dataset(full_data, splits)

    del full_data  # Delete the reference to the original full_data
    gc.collect()  # Explicitly trigger garbage collection
    
    # In test mode: combine train+val for training, use test for testing
    if cfg.mode.mode == "test":
        if len(valid_data) > 0:
            print("Test mode: Combining train+valid for training")
            train_data = train_data + valid_data  # Combine train+val
            valid_data = []  # No validation set
        # test_data remains unchanged
    
    print(f"Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")
    
    # For backward compatibility, merge train+valid for cross-validation if no separate validation
    if cfg.mode.mode == "train" and len(valid_data) > 0:
        print("Train mode: Using train+valid for cross-validation, test set held out")
        cv_data = train_data + valid_data
    else:
        cv_data = train_data
    
    # Initialize wandb (optional)
    use_wandb = cfg.logging_method == "wandb" and WANDB_AVAILABLE
    if use_wandb:
        initialize_wandb(cfg)
    elif cfg.logging_method == "wandb" and not WANDB_AVAILABLE:
        print("Warning: wandb requested but not installed. Disabling wandb logging.")


    # Initialize checkpointing
    checkpoint_dir = proj_dir / Path(cfg.callbacks.model_checkpoint.dirpath) / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)


    if cfg.model.name == "epiformer":
        model = EpiformerModel(cfg).to(device)
    
    # TODO: STEP 3: Mode-aware Fine-tuning Training
    if cfg.mode.mode == "train":
        print("\n=== STEP 3: Training Mode (K-Fold Cross-Validation) ===")
        
        # Cross-validation setup
        kf = KFold(n_splits=cfg.hparams.train.kfolds, shuffle=True, random_state=cfg.seed)
        cv_splits = list(kf.split(cv_data))
        
    elif cfg.mode.mode == "test":
        print("\n=== STEP 3: Test Mode (Train/Test only, no validation) ===")
        
        # Train/test split only (no validation) - train_data already combined with valid_data above
        train_indices = list(range(len(train_data)))
        cv_splits = [(train_indices, [])]
        cv_data = train_data
        
    elif cfg.mode.mode in ["val", "debug"]:
        mode_name = "Validation Mode (Train/Val/Test)" if cfg.mode.mode == "val" else "Debug Mode (Train/Val/Test with monitoring)"
        print(f"\n=== STEP 3: {mode_name} ===")
        
        # Use predefined train/valid split
    
        train_indices = list(range(len(train_data)))
        valid_indices = list(range(len(train_data), len(train_data) + len(valid_data)))
        cv_splits = [(train_indices, valid_indices)]
        cv_data = train_data + valid_data
        
    else:
        print(f"\n=== STEP 3: Development Mode ===")
        # For dev mode, use simple split
        kf = KFold(n_splits=cfg.hparams.train.kfolds, shuffle=True, random_state=cfg.seed)
        cv_splits = list(kf.split(cv_data))
    
    all_results = []
    # Get WALLE edge cutoff from loss config if available
    walle_edge_cutoff = getattr(cfg.loss.walle, 'edge_cutoff', 3.39) if hasattr(cfg.loss, 'walle') else 3.39
    metrics = HierarchicalMetrics(cfg.model.epi_threshold, cfg.model.para_threshold, walle_edge_cutoff).to(device)
    
    
    model_ckpt = ModelCheckpoint(
        dirpath=checkpoint_dir,
        run_id=run_id,
        filename=cfg.model.name,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        config=cfg
    )


    # Resume state if applicable
    start_fold = 0
    start_epoch = 1
    if checkpoint:
        start_fold = checkpoint.get('fold', 0)
        start_epoch = checkpoint.get('epoch', 1) + 1
        print(f"Resuming at fold {start_fold}, epoch {start_epoch}")
    
    # TODO: Mode-aware cross-validation training
    num_folds = len(cv_splits)
    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        # Skip folds that are already completed during resume
        if checkpoint and fold_idx < start_fold:
            print(f"\n⏭️ Skipping completed fold {fold_idx+1}/{num_folds}")
            continue
            
        if cfg.mode.mode == "test":
            print(f"\n▶︎ Training on predefined train/valid split")
        else:
            print(f"\n▶︎ Fold {fold_idx+1}/{num_folds}")
        
        # Create fold datasets
        train_subset = [cv_data[i] for i in train_idx]
        val_subset = [cv_data[i] for i in val_idx]
        
        train_loader = create_dataloader(train_subset, cfg, cfg.hparams.train.batch_size, shuffle=True)
        val_loader = create_dataloader(val_subset, cfg, cfg.hparams.train.batch_size, shuffle=False)
        
        # Reinitialize model for each fold (load pre-trained weights if available)
        # fold_model = HierarchicalModelWithHeads(cfg).to(device)
        # fold_model.disable_pretraining()

        """
        initialize the epiformer model
        """
        if cfg.model.name == "epiformer":
            fold_model = EpiformerModel(cfg).to(device)
        
        optimizer = Adam(
            fold_model.parameters(),
            lr=cfg.hparams.train.learning_rate,
            weight_decay=cfg.hparams.train.weight_decay
        )

        if cfg.hparams.train.scheduler=="reduce_lr_on_plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=True
            )

        elif cfg.hparams.train.scheduler=="cosine_annealing":
            # Cosine Annealing with Warm Restarts
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10,  # Initial restart period
                T_mult=1,  # Period multiplication factor
                eta_min=1e-7  # Minimum learning rate
            )
        
        elif cfg.hparams.train.scheduler=="one_cycle_lr":
            # One Cycle LR 
            scheduler = OneCycleLR(
                optimizer,
                max_lr=cfg.hparams.train.learning_rate,
                epochs=cfg.hparams.train.num_epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.3  # Spend 30% of training ramping up
            )
        
        # Initialize callbacks
        es = EarlyStopping(**cfg.callbacks.early_stopping)
        best_val_loss = float('inf')
        best_val_metrics = None
        
        # Resume training if checkpoint exists and we're on the right fold
        if checkpoint and fold_idx == start_fold:
            fold_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            # Load thresholds
            # fold_model.hierarchical_model.epi_threshold = checkpoint['epi_threshold']
            # fold_model.hierarchical_model.para_threshold = checkpoint['para_threshold']
            # Restore early stopping state if available
            if 'early_stopping' in checkpoint:
                es.load_state_dict(checkpoint['early_stopping'])
            # Restore best validation metrics
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            best_val_metrics = checkpoint.get('best_val_metrics', None)
            print(f"✓ Resumed model, optimizer, scheduler, thresholds, and callbacks from checkpoint")

        print(fold_model)
        
        # Fine-tuning loop - start from correct epoch if resuming
        start_epoch_fold = start_epoch if (checkpoint and fold_idx == start_fold) else 1
        for epoch in range(start_epoch_fold, cfg.hparams.train.num_epochs + 1):
            epoch_start = time.time()
            
            train_loss, train_metrics, train_loss_comp = train_epoch(
                fold_model, train_loader, optimizer, device, metrics, cfg, epoch
            )
            
            # Validate epoch only if not in test mode 
            if cfg.mode.mode != "test":
                val_loss, val_metrics, val_loss_comp = validate_epoch(
                    fold_model, val_loader, device, metrics, cfg, epoch
                )
                # Fixed thresholds (0.5) used for validation
                
            else:
                # In test mode: skip validation completely 
                val_loss = 0.0
                val_metrics = {}
                val_loss_comp = {}
            
            epoch_time = time.time() - epoch_start
            
            # Log to wandb
            if use_wandb:
                log_data = {
                    "run_id": run_id,
                    "fold": fold_idx,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "epoch_time": epoch_time,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    **{f"train_{k}": v for k, v in train_loss_comp.items()}
                }
                
                # Add validation metrics only if not in test mode
                if cfg.mode.mode != "test":
                    log_data["val_loss"] = val_loss
                    log_data.update({f"val_{k}": v for k, v in val_metrics.items()})
                    log_data.update({f"val_{k}": v for k, v in val_loss_comp.items()})
                
                wandb.log(log_data)
            
            # Print epoch summary
            if cfg.mode.mode != "test":
                print(f"Epoch {epoch:03d} | Time: {epoch_time:.1f}s | LR: {optimizer.param_groups[0]['lr']:.2e}")
                print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch:03d} | Time: {epoch_time:.1f}s | LR: {optimizer.param_groups[0]['lr']:.2e}")
                print(f"Train Loss: {train_loss:.4f}")

            # Print loss components only if not in test mode
            if cfg.mode.mode != "test":
                print("=== Loss Components ===")
                print(f"Train: " + " | ".join([f"{k}: {v:.4f}" for k, v in train_loss_comp.items()]))
                print(f"Val: " + " | ".join([f"{k}: {v:.4f}" for k, v in val_loss_comp.items()]))

            print("Train Edge Metrics: " + " | ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items() if k.startswith('edge_')]))
            print("Train Epitope Metrics: " + " | ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items() if k.startswith('epitope_')]))
            print("Train Paratope Metrics: " + " | ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items() if k.startswith('paratope_')]))
            
            # Print validation metrics only if not in test mode
            if cfg.mode.mode != "test":
                print("Val Edge Metrics: " + " | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items() if k.startswith('edge_')]))
                print("Val Epitope Metrics: " + " | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items() if k.startswith('epitope_')]))
                print("Val Paratope Metrics: " + " | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items() if k.startswith('paratope_')]))
            
            # Print confusion matrix components
            if cfg.mode.mode != "test":
                print("=== Confusion Matrix Components ===")  
                print(f"Train Epitope - TP: {train_metrics.get('epitope_tp', 0):.0f} | FP: {train_metrics.get('epitope_fp', 0):.0f} | TN: {train_metrics.get('epitope_tn', 0):.0f} | FN: {train_metrics.get('epitope_fn', 0):.0f}")
                print(f"Train Paratope - TP: {train_metrics.get('paratope_tp', 0):.0f} | FP: {train_metrics.get('paratope_fp', 0):.0f} | TN: {train_metrics.get('paratope_tn', 0):.0f} | FN: {train_metrics.get('paratope_fn', 0):.0f}")
                print(f"Val Epitope - TP: {val_metrics.get('epitope_tp', 0):.0f} | FP: {val_metrics.get('epitope_fp', 0):.0f} | TN: {val_metrics.get('epitope_tn', 0):.0f} | FN: {val_metrics.get('epitope_fn', 0):.0f}")
                print(f"Val Paratope - TP: {val_metrics.get('paratope_tp', 0):.0f} | FP: {val_metrics.get('paratope_fp', 0):.0f} | TN: {val_metrics.get('paratope_tn', 0):.0f} | FN: {val_metrics.get('paratope_fn', 0):.0f}")
          
            
            # Save checkpoint and update best model only if not in test mode
            if cfg.mode.mode != "test":
                model_ckpt.save(
                    model=fold_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    fold=fold_idx,
                    value=val_loss,
                    other_states={
                        'early_stopping': es.state_dict(),
                        'best_val_loss': best_val_loss,
                        'best_val_metrics': best_val_metrics,
                        'best_epi_threshold': cfg.model.epi_threshold,
                        'best_para_threshold': cfg.model.para_threshold,
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
            else:
                # In test mode: keep the last model, no early stopping 
                scheduler.step(train_loss)  # Use training loss for scheduler
        
        # Store best results from this fold only if not in test mode
        if cfg.mode.mode != "test":
            if best_val_metrics:
                final_metrics = {
                    k: (v.cpu().item() if isinstance(v, torch.Tensor) else v)
                    for k, v in best_val_metrics.items()
                }
                all_results.append(final_metrics)
        else:
            # In test mode: use the last trained model for testing 
            model = fold_model

    
    # Report CV results only if not in test mode 
    if cfg.mode.mode != "test" and all_results:
        avg = {k: np.mean([m[k] for m in all_results]) for k in all_results[0]}
        std = {k: np.std([m[k] for m in all_results]) for k in all_results[0]}
        
        if cfg.mode.mode == "train":
            print("\n=== Cross-Validation Results ===")
            for k in avg:
                print(f"{k}: {avg[k]:.4f} ± {std[k]:.4f}")
            
            if use_wandb:
                wandb.log({f"cv_{k}": avg[k] for k in avg})
                wandb.log({f"cv_{k}_std": std[k] for k in std})
        else:
            print("\n=== Development Results ===")
            for k in avg:
                print(f"{k}: {avg[k]:.4f} ± {std[k]:.4f}")
    
    # Test on hold-out set if in test mode 
    if cfg.mode.mode == "test":
        # After training completes
        model_ckpt.save(
            model=fold_model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            fold=fold_idx,
            value=val_loss,
            other_states={
                'best_epi_threshold': cfg.model.epi_threshold,
                'best_para_threshold': cfg.model.para_threshold,
            }
        )

        test_loader = create_dataloader(test_data, cfg, cfg.hparams.train.batch_size, shuffle=False)
        
        # Use fixed thresholds (0.5) for test mode
        print("\n=== Using Default Thresholds (Test Mode) ===")
        print(f"Epitope threshold: {cfg.model.epi_threshold:.3f}")
        print(f"Paratope threshold: {cfg.model.para_threshold:.3f}")
        
        test_loss, test_metrics, test_loss_comp = validate_epoch(
            model, test_loader, device, metrics, cfg, epoch=999
        )
        
        print("\n=== Test Results ===")
        print(f"Test Loss: {test_loss:.4f}")
        for k, v in test_metrics.items():
            print(f"{k}: {v:.4f}")
        
        if use_wandb:
            wandb.log({
                "test_loss": test_loss,
                **{f"test_{k}": v for k, v in test_metrics.items()},
                **{f"test_{k}": v for k, v in test_loss_comp.items()}
            })
        return  # Skip the final test evaluation for test mode
    
    # TODO: STEP 5: Final Test Evaluation (for other modes)
    print("\n=== STEP 5: Final Test Evaluation ===")
    
    # Load best model (only for non-test modes)
    model, optimizer, scheduler = model_ckpt.load_best_model(model, optimizer, scheduler, device)
    
    test_loader = create_dataloader(test_data, cfg, cfg.hparams.train.batch_size, shuffle=False)
    
    # Use fixed thresholds (0.5) for test evaluation
    best_epi_threshold = cfg.model.epi_threshold
    best_para_threshold = cfg.model.para_threshold
    
    print(f"\n=== Using Fixed Thresholds ===")
    print(f"Epitope threshold: {best_epi_threshold:.3f}")
    print(f"Paratope threshold: {best_para_threshold:.3f}")
    
    # Create fresh metrics object with fixed thresholds for test
    walle_edge_cutoff = getattr(cfg.loss.walle, 'edge_cutoff', 3.39) if hasattr(cfg.loss, 'walle') else 3.39
    test_metrics_obj = HierarchicalMetrics(best_epi_threshold, best_para_threshold, walle_edge_cutoff).to(device)
    
    # Evaluate on test set with fixed thresholds
    test_loss, test_metrics, test_loss_comp = validate_epoch(
        model, test_loader, device, test_metrics_obj, cfg, epoch=999
    )
    
    print(f"\nTest Loss: {test_loss:.4f}")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")
    
    # TODO: Add confusion matrix components to final test reporting
    print("=== Final Test Confusion Matrix ===")
    print(f"Epitope - TP: {test_metrics.get('epitope_tp', 0):.0f} | FP: {test_metrics.get('epitope_fp', 0):.0f} | TN: {test_metrics.get('epitope_tn', 0):.0f} | FN: {test_metrics.get('epitope_fn', 0):.0f}")
    print(f"Paratope - TP: {test_metrics.get('paratope_tp', 0):.0f} | FP: {test_metrics.get('paratope_fp', 0):.0f} | TN: {test_metrics.get('paratope_tn', 0):.0f} | FN: {test_metrics.get('paratope_fn', 0):.0f}")
    
    # Log test results to wandb
    if use_wandb:
        wandb.log({
            # "threshold": model.hierarchical_model.epi_threshold,
            "test_loss": test_loss,
            **{f"test_{k}": v for k, v in test_metrics.items()},
            **{f"test_{k}": v for k, v in test_loss_comp.items()}
        })
    
    # Save test results summary
    summary_dir = checkpoint_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    test_summary_file = summary_dir / f"{cfg.model.name}_test_results.csv"
    
    all_metrics = {}
    for k, v in test_metrics.items():
        if hasattr(v, 'item'):
            all_metrics[k] = v.item()
        else:
            all_metrics[k] = v
    
    all_metrics["total_time_s"] = time.time() - start_time
    
    with open(test_summary_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(all_metrics.keys())
        writer.writerow([f"{v:.4f}" if isinstance(v, float) else v 
                        for v in all_metrics.values()])
    
    print(f"→ Test summary saved to {test_summary_file}")
    
    # Generate embedding visualizations for debug mode
    if cfg.mode.mode == "debug" and getattr(cfg.mode, 'save_embeddings', True):
        print("\n=== Generating Embedding Visualizations ===")
        
        try:
            # Get embeddings from test set with best model
            model.eval()
            ag_embeddings, ab_embeddings = [], []
            epi_labels, para_labels = [], []
            
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    
                    # Get encoder outputs (before decoder)
                    ag_out, _ = model.hierarchical_model.ag_encoder(batch, 'ag')
                    ab_out, _ = model.hierarchical_model.ab_encoder(batch, 'ab')
                    
                    ag_embeddings.append(ag_out.cpu())
                    ab_embeddings.append(ab_out.cpu())
                    
                    # Get labels
                    epi_labels.append(batch['ag_res'].y.cpu())
                    para_labels.append(batch['ab_res'].y.cpu())
            
            
            # Concatenate all embeddings
            ag_embeddings = torch.cat(ag_embeddings, dim=0)
            ab_embeddings = torch.cat(ab_embeddings, dim=0)
            epi_labels = torch.cat(epi_labels, dim=0)
            para_labels = torch.cat(para_labels, dim=0)
            
            
            # Save visualizations
            embedding_dir = checkpoint_dir / "embeddings"
            save_embedding_plots(
                ag_embeddings, ab_embeddings, epi_labels, para_labels,
                str(embedding_dir), "final"
            )
            
            print(f"✓ Embedding visualizations saved to {embedding_dir}")
            
        except Exception as e:
            print(f"Warning: Could not generate embedding visualizations: {e}")
    
    # Cleanup
    if use_wandb:
        wandb.finish()
    
    print(f"\n✓ V3 End-to-End training pipeline completed in {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()



