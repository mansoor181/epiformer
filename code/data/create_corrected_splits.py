#!/usr/bin/env python3
"""
Create corrected AsEP split files by excluding the two removed complexes
TODO: Added script to generate corrected split files for hierarchical dataset
"""

import torch
import os
from typing import Dict, List, Set
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_original_asep_files(asep_data_dir: str) -> tuple[List[str], Dict]:
    """Load original AsEP PDB IDs and split dictionary"""
    
    # Load original PDB IDs
    pdb_ids_path = os.path.join(asep_data_dir, "asepv1-AbDb-IDs.txt")
    if not os.path.exists(pdb_ids_path):
        raise FileNotFoundError(f"Original PDB IDs file not found: {pdb_ids_path}")
    
    with open(pdb_ids_path, 'r') as f:
        original_pdb_ids = [line.strip() for line in f if line.strip()]
    
    # Load original split dict
    split_dict_path = os.path.join(asep_data_dir, "split", "split_dict.pt")
    if not os.path.exists(split_dict_path):
        raise FileNotFoundError(f"Original split dict not found: {split_dict_path}")
    
    original_splits = torch.load(split_dict_path, weights_only=True)
    
    logger.info(f"Loaded {len(original_pdb_ids)} original PDB IDs")
    logger.info(f"Split methods available: {list(original_splits.keys())}")
    
    return original_pdb_ids, original_splits

def create_index_mapping(original_pdb_ids: List[str], excluded_complexes: Set[str]) -> Dict[int, int]:
    """Create mapping from original indices to corrected indices"""
    
    mapping = {}
    corrected_idx = 0
    
    for original_idx, pdb_id in enumerate(original_pdb_ids):
        if pdb_id not in excluded_complexes:
            mapping[original_idx] = corrected_idx
            corrected_idx += 1
        else:
            logger.info(f"Excluding complex {pdb_id} at original index {original_idx}")
    
    logger.info(f"Created mapping: {len(original_pdb_ids)} -> {len(mapping)} indices")
    return mapping

def correct_splits(original_splits: Dict, index_mapping: Dict[int, int]) -> Dict:
    """Apply index mapping to create corrected splits"""
    
    corrected_splits = {}
    
    for split_method, splits in original_splits.items():
        logger.info(f"Processing {split_method} split")
        
        corrected_method_splits = {}
        
        for split_name, indices in splits.items():
            original_indices = indices.tolist()
            corrected_indices = []
            
            for idx in original_indices:
                if idx in index_mapping:
                    corrected_indices.append(index_mapping[idx])
                else:
                    logger.info(f"Skipping excluded index {idx} from {split_name} set")
            
            corrected_method_splits[split_name] = torch.tensor(corrected_indices, dtype=torch.long)
            
            logger.info(f"  {split_name}: {len(original_indices)} -> {len(corrected_indices)} indices")
        
        corrected_splits[split_method] = corrected_method_splits
    
    return corrected_splits

def save_corrected_files(corrected_pdb_ids: List[str], corrected_splits: Dict, output_dir: str):
    """Save corrected PDB IDs and split files"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save corrected PDB IDs
    corrected_pdb_path = os.path.join(output_dir, "asepv1-AbDb-IDs-corrected.txt")
    with open(corrected_pdb_path, 'w') as f:
        for pdb_id in corrected_pdb_ids:
            f.write(f"{pdb_id}\n")
    logger.info(f"Saved corrected PDB IDs to: {corrected_pdb_path}")
    
    # Save corrected split dict
    corrected_split_path = os.path.join(output_dir, "split_dict_corrected.pt")
    torch.save(corrected_splits, corrected_split_path)
    logger.info(f"Saved corrected splits to: {corrected_split_path}")
    
    # Print summary
    print("\n=== Corrected Split Summary ===")
    for method, splits in corrected_splits.items():
        train_size = len(splits['train'])
        val_size = len(splits.get('val', splits.get('valid', [])))
        test_size = len(splits['test'])
        total_size = train_size + val_size + test_size
        
        print(f"{method}:")
        print(f"  Train: {train_size}")
        print(f"  Valid: {val_size}")  
        print(f"  Test: {test_size}")
        print(f"  Total: {total_size}")
    
    return corrected_pdb_path, corrected_split_path

def main():
    """Main function to create corrected split files"""
    
    print("=== Creating Corrected AsEP Split Files ===")
    
    # Configuration
    excluded_complexes = {'5nj6_0P', '5ies_0P'}
    asep_data_dir = "../../data/asep"
    output_dir = "../../data/asep/split"
    
    try:
        # Load original files
        print(f"Loading original files from: {asep_data_dir}")
        original_pdb_ids, original_splits = load_original_asep_files(asep_data_dir)
        
        # Create index mapping
        print(f"Creating index mapping (excluding {excluded_complexes})")
        index_mapping = create_index_mapping(original_pdb_ids, excluded_complexes)
        
        # Create corrected PDB IDs list
        corrected_pdb_ids = [pdb_id for pdb_id in original_pdb_ids if pdb_id not in excluded_complexes]
        print(f"Corrected PDB IDs: {len(corrected_pdb_ids)} (was {len(original_pdb_ids)})")
        
        # Apply corrections to splits
        print("Applying corrections to splits")
        corrected_splits = correct_splits(original_splits, index_mapping)
        
        # Save corrected files
        print(f"Saving corrected files to: {output_dir}")
        corrected_pdb_path, corrected_split_path = save_corrected_files(
            corrected_pdb_ids, corrected_splits, output_dir
        )
        
        print("\n=== Success! ===")
        print("To use the corrected splits, update your config:")
        print(f"  pdb_ids_path: {corrected_pdb_path}")
        print(f"  split_dict_path: {corrected_split_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        logger.exception("Failed to create corrected splits")

if __name__ == "__main__":
    main()