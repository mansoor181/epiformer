#!/usr/bin/env python3
"""
EpiFormer Evaluation Script

Evaluates a trained EpiFormer model checkpoint on both dataset splits.
Computes AUROC, AUPRC, F1, MCC, Precision, and Recall for epitope prediction.

Usage:
    python evaluate.py --checkpoint ../checkpoints/best-glamorous-sweep-37/epiformer_best.pt
    python evaluate.py --checkpoint path/to/checkpoint.pt --data_dir path/to/data --gpu_id 0
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
from torch_geometric.data import Batch
from omegaconf import OmegaConf
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score
)

# Add code directory to path
CODE_DIR = Path(__file__).parent
sys.path.insert(0, str(CODE_DIR))

from model.epiformer import EpiformerModel
from utils import load_data, seed_everything
from data.data_splits import get_asep_splits


def load_checkpoint(checkpoint_path, device):
    """Load model checkpoint and reconstruct config."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Reconstruct config
    cfg = OmegaConf.create(checkpoint['config'])


    # Add missing keys that the model expects
    if 'epiformer' in cfg.model:
        if 'ag_resmp_type' not in cfg.model.epiformer:
            cfg.model.epiformer.ag_resmp_type = 'egnn'
        if 'ab_resmp_type' not in cfg.model.epiformer:
            cfg.model.epiformer.ab_resmp_type = 'egnn'
        if 'geo_dim' not in cfg.model.epiformer:
            cfg.model.epiformer.geo_dim = cfg.model.get('geo_dim', 105)

    # Ensure dataset config has required fields
    if 'graph_num_relations' not in cfg.dataset:
        cfg.dataset.graph_num_relations = 4
    if 'plm_type' not in cfg.dataset:
        cfg.dataset.plm_type = 'esm2_650m'

    # Initialize model
    model = EpiformerModel(cfg).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded successfully")
    print(f"  Encoder blocks: {len(model.epiformer_encoder.epiformer_blocks)}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, cfg


def compute_metrics(y_true, y_prob, threshold=0.3):
    """Compute all evaluation metrics."""
    y_pred = (y_prob > threshold).astype(int)

    # Handle edge case of single class
    if len(np.unique(y_true)) < 2:
        return {
            'auroc': 0.5,
            'auprc': float(y_true.mean()),
            'f1': 0.0,
            'mcc': 0.0,
            'precision': 0.0,
            'recall': 0.0,
        }

    return {
        'auroc': roc_auc_score(y_true, y_prob),
        'auprc': average_precision_score(y_true, y_prob),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
    }


def evaluate_split(model, data_list, device, threshold=0.3, desc="Evaluating"):
    """Evaluate model on a dataset split."""
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for sample in tqdm(data_list, desc=desc):
            batch = Batch.from_data_list([sample]).to(device)
            outputs = model(batch)

            all_probs.append(outputs['epitope_prob'].cpu().numpy())
            all_labels.append(batch['ag_res'].y.cpu().numpy())

    # Concatenate all predictions
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)

    # Compute metrics
    metrics = compute_metrics(y_true, y_prob, threshold)

    # Add dataset statistics
    metrics['n_samples'] = len(data_list)
    metrics['n_residues'] = len(y_true)
    metrics['epitope_ratio'] = float(y_true.mean())

    return metrics


def print_metrics(metrics, split_name):
    """Print metrics in a formatted table."""
    print(f"\n{'='*60}")
    print(f"Results for {split_name}")
    print(f"{'='*60}")
    print(f"  Samples:       {metrics['n_samples']}")
    print(f"  Residues:      {metrics['n_residues']}")
    print(f"  Epitope ratio: {metrics['epitope_ratio']:.4f}")
    print(f"  {'-'*40}")
    print(f"  AUROC:         {metrics['auroc']:.4f}")
    print(f"  AUPRC:         {metrics['auprc']:.4f}")
    print(f"  F1:            {metrics['f1']:.4f}")
    print(f"  MCC:           {metrics['mcc']:.4f}")
    print(f"  Precision:     {metrics['precision']:.4f}")
    print(f"  Recall:        {metrics['recall']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate EpiFormer checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (e.g., ../checkpoints/best-glamorous-sweep-37/epiformer_best.pt)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to data directory (default: ../data/asep)"
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU ID to use (default: 0)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Classification threshold (default: 0.3)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    # Set seed
    seed_everything(args.seed)

    # Set device
    if torch.cuda.is_available() and args.gpu_id >= 0:
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Set paths
    proj_dir = CODE_DIR.parent
    args.checkpoint = Path(args.checkpoint)

    if args.data_dir is None:
        args.data_dir = proj_dir / "data" / "asep"
    else:
        args.data_dir = Path(args.data_dir)

    # Verify paths exist
    if not args.checkpoint.exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Load model
    model, cfg = load_checkpoint(args.checkpoint, device)

    # Load dataset
    dataset_path = args.data_dir / "epiformer" / "res_graph_tensor_esm2_650m.pkl"
    split_path = args.data_dir / "split" / "split_dict_corrected.pt"

    print(f"\nLoading dataset from: {dataset_path}")
    full_data = load_data(str(dataset_path))
    print(f"Loaded {len(full_data)} samples")

    # Evaluate on both splits
    results = {}

    for split_method in ["epitope_ratio", "epitope_group"]:
        print(f"\n{'#'*60}")
        print(f"# Split method: {split_method}")
        print(f"{'#'*60}")

        # Load splits
        splits = get_asep_splits(str(split_path), split_method)

        # Get test data
        test_indices = splits['test'].tolist()
        test_data = [full_data[i] for i in test_indices]

        # Get validation data
        val_indices = splits['valid'].tolist()
        val_data = [full_data[i] for i in val_indices]

        print(f"Test set:  {len(test_data)} samples")
        print(f"Valid set: {len(val_data)} samples")

        # Evaluate test set
        test_metrics = evaluate_split(
            model, test_data, device,
            threshold=args.threshold,
            desc=f"Test ({split_method})"
        )
        print_metrics(test_metrics, f"Test Set ({split_method})")

        # Evaluate validation set
        val_metrics = evaluate_split(
            model, val_data, device,
            threshold=args.threshold,
            desc=f"Valid ({split_method})"
        )
        print_metrics(val_metrics, f"Validation Set ({split_method})")

        results[split_method] = {
            'test': test_metrics,
            'valid': val_metrics
        }

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Split':<20} {'Set':<10} {'AUROC':<10} {'AUPRC':<10} {'F1':<10} {'MCC':<10}")
    print(f"{'-'*80}")

    for split_method in ["epitope_ratio", "epitope_group"]:
        for set_name in ["test", "valid"]:
            m = results[split_method][set_name]
            print(f"{split_method:<20} {set_name:<10} {m['auroc']:.4f}     {m['auprc']:.4f}     {m['f1']:.4f}     {m['mcc']:.4f}")

    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Threshold:  {args.threshold}")


if __name__ == "__main__":
    main()
