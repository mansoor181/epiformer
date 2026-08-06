"""
Dataset splitting utilities for AsEP paper compatibility
TODO: Added comprehensive dataset splitting methods including AsEP paper splits
"""

import os
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


def get_asep_splits(split_dict_path: str, split_method: str = "epitope_ratio") -> Dict[str, torch.Tensor]:
    """
    Load corrected AsEP paper dataset splits
    
    TODO: Simplified AsEP split loading using corrected split files
    
    Args:
        split_dict_path: Path to corrected split_dict_corrected.pt file
        split_method: Either "epitope_ratio" or "epitope_group"
        
    Returns:
        Dictionary with keys ['train', 'val', 'test'], each value is a tensor of indices
    """

    # NOTE: the split files (epitope_ratio and epitope_group) the indices for train, val, and test sets

    assert split_method in {"epitope_ratio", "epitope_group"}, \
        f"split_method={split_method} not supported, valid options: ['epitope_ratio', 'epitope_group']"
    
    if not os.path.exists(split_dict_path):
        raise FileNotFoundError(f"Split dict file not found: {split_dict_path}")
    
    logger.info(f"Loading corrected AsEP {split_method} splits from {split_dict_path}")
    split_dict = torch.load(split_dict_path, weights_only=True)
    
    if split_method not in split_dict:
        raise ValueError(f"Split method {split_method} not found in split dict. Available: {list(split_dict.keys())}")
    
    splits = split_dict[split_method]
    
    # Convert 'val' to 'valid' for consistency if needed
    if 'val' in splits and 'valid' not in splits:
        splits['valid'] = splits.pop('val')
    
    logger.info(f"Loaded {split_method} splits: train={len(splits['train'])}, "
                f"valid={len(splits.get('valid', []))}, test={len(splits['test'])}")
    
    return splits


def get_random_split(dataset_size: int, seed: Optional[int] = None, 
                    train_ratio: float = 0.8, val_ratio: float = 0.1, 
                    test_ratio: float = 0.1) -> Dict[str, torch.Tensor]:
    """
    Generate random dataset split
    
    TODO: Added configurable random split generation
    
    Args:
        dataset_size: Total number of samples
        seed: Random seed for reproducibility
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        
    Returns:
        Dictionary with keys ['train', 'valid', 'test'], each value is a tensor of indices
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"
    
    if seed is not None:
        torch.manual_seed(seed)
    
    # Generate random permutation of indices
    idx = torch.randperm(dataset_size)
    
    # Calculate split sizes
    train_size = int(dataset_size * train_ratio)
    val_size = int(dataset_size * val_ratio)
    
    # Split indices
    splits = {
        'train': idx[:train_size],
        'valid': idx[train_size:train_size + val_size],
        'test': idx[train_size + val_size:]
    }
    
    logger.info(f"Generated random splits: train={len(splits['train'])}, "
                f"valid={len(splits['valid'])}, test={len(splits['test'])}")
    
    return splits


def get_dataset_splits(split_dict_path, dataset_size: int, split_config: dict) -> Dict[str, torch.Tensor]:
    """
    Get dataset splits based on configuration
    
    TODO: Added unified dataset splitting interface
    
    Args:
        dataset_size: Total number of samples in dataset
        split_config: Configuration dictionary with split parameters
        
    Returns:
        Dictionary with train/valid/test splits
    """
    split_method = split_config.get('method', 'random')
    
    if split_method == 'random':
        return get_random_split(
            dataset_size=dataset_size,
            seed=split_config.get('seed', None),
            train_ratio=split_config.get('train_ratio', 0.8),
            val_ratio=split_config.get('val_ratio', 0.1),
            test_ratio=split_config.get('test_ratio', 0.1)
        )
    
    elif split_method in ['epitope_ratio', 'epitope_group']:
        # split_dict_path = split_config.get('split_dict_path')
        if not split_dict_path:
            raise ValueError(f"split_dict_path required for {split_method} split method")
        
        # TODO: Added dataset size validation for AsEP splits
        expected_size = 1721  # After excluding 5nj6_0P and 5ies_0P
        if dataset_size != expected_size:
            if dataset_size < 100:  # Likely test dataset
                logger.warning(f"Using test dataset ({dataset_size} samples) with AsEP splits - "
                             f"falling back to random split for testing")
                return get_random_split(
                    dataset_size=dataset_size,
                    seed=split_config.get('seed', 42),
                    train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
                )
            else:
                logger.warning(f"Dataset size {dataset_size} doesn't match expected AsEP size {expected_size}")
        
        # TODO: Using corrected split files (no index mapping needed)
        return get_asep_splits(split_dict_path, split_method)
    
    else:
        raise ValueError(f"Unknown split method: {split_method}. "
                        f"Supported methods: ['random', 'epitope_ratio', 'epitope_group']")


def apply_splits_to_dataset(dataset: List, splits: Dict[str, torch.Tensor]) -> Tuple[List, List, List]:
    """
    Apply split indices to dataset
    
    TODO: Added dataset splitting application utility
    
    Args:
        dataset: List of data samples
        splits: Dictionary with train/valid/test indices
        
    Returns:
        Tuple of (train_data, valid_data, test_data)
    """
    train_data = [dataset[i] for i in splits['train']]
    valid_data = [dataset[i] for i in splits.get('valid', [])]
    test_data = [dataset[i] for i in splits['test']]
    
    logger.info(f"Applied splits: train={len(train_data)}, valid={len(valid_data)}, test={len(test_data)}")
    
    return train_data, valid_data, test_data




def validate_split_compatibility(dataset_size: int, splits: Dict[str, torch.Tensor], 
                                split_method: str) -> bool:
    """
    Validate that splits are compatible with dataset
    
    TODO: Added split validation for data integrity
    
    Args:
        dataset_size: Actual dataset size
        splits: Split indices
        split_method: Method used for splitting
        
    Returns:
        True if splits are valid, raises exception otherwise
    """
    # Check all indices are within dataset bounds
    all_indices = torch.cat([splits['train'], splits.get('valid', torch.tensor([])), splits['test']])
    
    if len(all_indices) == 0:
        raise ValueError("No indices found in splits")
    
    if all_indices.max() >= dataset_size:
        raise ValueError(f"Split indices exceed dataset size: max_index={all_indices.max()}, "
                        f"dataset_size={dataset_size}")
    
    if all_indices.min() < 0:
        raise ValueError(f"Split indices contain negative values: min_index={all_indices.min()}")
    
    # Check for overlaps
    train_set = set(splits['train'].tolist())
    valid_set = set(splits.get('valid', torch.tensor([])).tolist())
    test_set = set(splits['test'].tolist())
    
    if train_set & test_set:
        raise ValueError("Train and test sets overlap")
    
    if valid_set and (train_set & valid_set):
        raise ValueError("Train and validation sets overlap")
    
    if valid_set and (valid_set & test_set):
        raise ValueError("Validation and test sets overlap")
    
    # Check coverage (all indices should be used exactly once)
    expected_total = len(splits['train']) + len(splits.get('valid', [])) + len(splits['test'])
    unique_indices = len(set(all_indices.tolist()))
    
    if unique_indices != expected_total:
        raise ValueError(f"Duplicate indices found: expected {expected_total} unique, got {unique_indices}")
    
    if split_method in ['epitope_ratio', 'epitope_group']:
        # For AsEP splits, we expect to use all available samples in hierarchical dataset
        if unique_indices != dataset_size:
            logger.warning(f"AsEP split uses {unique_indices} samples, dataset has {dataset_size}")
    
    logger.info(f"Split validation passed for {split_method}: {unique_indices} unique indices")
    return True