"""
Create dict_pre_cal.pt from asepv1_interim_graphs/*.pt files.

This script converts the individual interim graph files (downloaded from Zenodo)
into the dictionary format used by baselines and construct_res_graphs_tensor.py.

The interim graph files contain pre-computed embeddings:
- Antibody: IgFold (512D) or ESM-2 (480D)
- Antigen: ESM-2 (480D)

Input: data/asep/asepv1_interim_graphs/*.pt (1,723 files from Zenodo)
Output: data/asep/processed/dict_pre_cal.pt

Output format:
{
    'pdb_id': {
        'x_g': [n_surface, emb_dim],   # Antigen embeddings (surface residues only)
        'x_b': [n_cdr, emb_dim],       # Antibody embeddings (CDR residues only)
        'edge_index_g': [2, E_ag],     # Antigen edges
        'edge_index_b': [2, E_ab],     # Antibody edges
        'edge_index_bg': [2, E_cross], # Cross-chain bipartite edges
        'y_g': [n_surface],            # Epitope labels (0/1)
        'y_b': [n_cdr],                # Paratope labels (0/1)
    },
    ...
}

Usage:
    cd code/data
    python create_dict_pre_cal.py [--interim_dir PATH] [--output_dir PATH]
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
from torch import Tensor
from tqdm import tqdm

# Excluded samples (alignment errors)
EXCLUDED_SAMPLES = {'5nj6_0P', '5ies_0P'}


def load_interim_graph(pdb_id: str, interim_dir: str) -> Dict[str, Any]:
    """Load interim graph data from a .pt file.

    Returns dict with keys:
    - embedding: {ab: {igfold, esm2}, ag: {esm2}}
    - edges: {ab, ag, bipartite} (sparse COO tensors)
    - mapping: {ab: {seqres2cdr}, ag: {seqres2surf}}
    - seqres: {ab: {H, L}, ag: {...}}
    - stats, Nb, Ng, abdbid
    """
    pt_path = os.path.join(interim_dir, f"{pdb_id}.pt")
    if not os.path.exists(pt_path):
        return None
    return torch.load(pt_path, map_location='cpu', weights_only=False)


def filter_embeddings_by_mask(
    embeddings: Tensor,
    mask: Any
) -> Tensor:
    """Filter embeddings using a binary mask (seqres2surf or seqres2cdr)."""
    if hasattr(mask, 'astype'):
        # numpy array
        mask_bool = mask.astype(bool)
    else:
        # tensor
        mask_bool = mask.bool()
    return embeddings[mask_bool]


def get_labels_from_bipartite_edges(
    edge_index_bg: Tensor,
    n_ab: int,
    n_ag: int
) -> Tuple[Tensor, Tensor]:
    """
    Extract epitope/paratope labels from bipartite edge indices.

    edge_index_bg: [2, E] where row 0 is AB indices, row 1 is AG indices
    Returns: (y_b, y_g) - binary label tensors
    """
    y_b = torch.zeros(n_ab, dtype=torch.long)
    y_g = torch.zeros(n_ag, dtype=torch.long)

    # Nodes connected by bipartite edges are labeled as epitope/paratope
    y_b[edge_index_bg[0].unique(sorted=True)] = 1
    y_g[edge_index_bg[1].unique(sorted=True)] = 1

    return y_b, y_g


def convert_interim_to_dict_entry(
    data: Dict[str, Any],
    ab_embedding_key: str = 'igfold',
    ag_embedding_key: str = 'esm2'
) -> Dict[str, Any]:
    """
    Convert a single interim graph to dict_pre_cal format.

    Args:
        data: Interim graph data loaded from .pt file
        ab_embedding_key: Which antibody embedding to use ('igfold' or 'esm2')
        ag_embedding_key: Which antigen embedding to use ('esm2')

    Returns:
        Dict with x_b, x_g, edge_index_b, edge_index_g, edge_index_bg, y_b, y_g
    """
    # Get mappings for filtering
    seqres2cdr = data['mapping']['ab']['seqres2cdr']
    seqres2surf = data['mapping']['ag']['seqres2surf']

    # Get embeddings and filter to CDR/surface residues
    ab_emb_full = data['embedding']['ab'][ab_embedding_key]
    ag_emb_full = data['embedding']['ag'][ag_embedding_key]

    x_b = filter_embeddings_by_mask(ab_emb_full, seqres2cdr)
    x_g = filter_embeddings_by_mask(ag_emb_full, seqres2surf)

    # Get edge indices (already filtered in interim graphs)
    edge_index_b = data['edges']['ab'].coalesce().indices()
    edge_index_g = data['edges']['ag'].coalesce().indices()
    edge_index_bg = data['edges']['bipartite'].coalesce().indices()

    # Get labels from bipartite edges
    n_ab = x_b.size(0)
    n_ag = x_g.size(0)
    y_b, y_g = get_labels_from_bipartite_edges(edge_index_bg, n_ab, n_ag)

    return {
        'x_b': x_b,
        'x_g': x_g,
        'edge_index_b': edge_index_b,
        'edge_index_g': edge_index_g,
        'edge_index_bg': edge_index_bg,
        'y_b': y_b,
        'y_g': y_g,
    }


def create_dict_pre_cal(
    interim_dir: str,
    output_path: str,
    ab_embedding: str = 'igfold',
    ag_embedding: str = 'esm2',
    id_list_file: str = None
) -> None:
    """
    Create dict_pre_cal.pt from interim graph files.

    Args:
        interim_dir: Directory containing asepv1_interim_graphs/*.pt files
        output_path: Path to save output dict_pre_cal.pt file
        ab_embedding: Antibody embedding type ('igfold' or 'esm2')
        ag_embedding: Antigen embedding type ('esm2')
        id_list_file: Optional file with PDB ID list (one per line)
    """
    # Get list of PDB IDs
    if id_list_file and os.path.exists(id_list_file):
        with open(id_list_file, 'r') as f:
            pdb_ids = [line.strip() for line in f if line.strip()]
    else:
        # Get from directory listing
        pdb_ids = [
            f.replace('.pt', '')
            for f in os.listdir(interim_dir)
            if f.endswith('.pt')
        ]

    print(f"Found {len(pdb_ids)} interim graph files")

    # Process each complex
    dict_pre_cal = {}
    skipped = []

    for pdb_id in tqdm(pdb_ids, desc="Converting interim graphs"):
        # Skip excluded samples
        if pdb_id in EXCLUDED_SAMPLES:
            skipped.append((pdb_id, "Excluded sample"))
            continue

        # Load interim graph
        data = load_interim_graph(pdb_id, interim_dir)
        if data is None:
            skipped.append((pdb_id, "File not found"))
            continue

        # Check required keys exist
        required_keys = ['embedding', 'edges', 'mapping']
        if not all(k in data for k in required_keys):
            skipped.append((pdb_id, "Missing required keys"))
            continue

        # Check embeddings exist
        if ab_embedding not in data['embedding']['ab']:
            skipped.append((pdb_id, f"Missing {ab_embedding} antibody embedding"))
            continue
        if ag_embedding not in data['embedding']['ag']:
            skipped.append((pdb_id, f"Missing {ag_embedding} antigen embedding"))
            continue

        # Convert to dict format
        entry = convert_interim_to_dict_entry(
            data,
            ab_embedding_key=ab_embedding,
            ag_embedding_key=ag_embedding
        )
        dict_pre_cal[pdb_id] = entry

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save
    print(f"\nSaving {len(dict_pre_cal)} entries to {output_path}")
    torch.save(dict_pre_cal, output_path)

    # Report
    print(f"\nProcessed: {len(dict_pre_cal)}")
    print(f"Skipped: {len(skipped)}")
    if skipped:
        print("\nSkipped samples:")
        for pdb_id, reason in skipped[:10]:
            print(f"  {pdb_id}: {reason}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")

    # Verify
    sample_key = list(dict_pre_cal.keys())[0]
    sample = dict_pre_cal[sample_key]
    print(f"\nSample entry '{sample_key}':")
    for key, value in sample.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)}")


def main():
    parser = argparse.ArgumentParser(
        description="Create dict_pre_cal.pt from asepv1_interim_graphs/*.pt files"
    )
    parser.add_argument(
        "--interim_dir",
        type=str,
        default="../../data/asep/asepv1_interim_graphs",
        help="Directory containing interim graph .pt files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../../data/asep/processed",
        help="Output directory for dict_pre_cal.pt"
    )
    parser.add_argument(
        "--ab_embedding",
        type=str,
        default="igfold",
        choices=["igfold", "esm2"],
        help="Antibody embedding type (default: igfold)"
    )
    parser.add_argument(
        "--ag_embedding",
        type=str,
        default="esm2",
        choices=["esm2"],
        help="Antigen embedding type (default: esm2)"
    )
    parser.add_argument(
        "--id_list",
        type=str,
        default=None,
        help="Optional file with PDB ID list (e.g., asepv1-AbDb-IDs.txt)"
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="",
        help="Suffix for output filename (e.g., '_esm2_esm2' for dict_pre_cal_esm2_esm2.pt)"
    )

    args = parser.parse_args()

    # Construct output path
    output_filename = f"dict_pre_cal{args.output_suffix}.pt"
    output_path = os.path.join(args.output_dir, output_filename)

    print(f"Input directory: {args.interim_dir}")
    print(f"Output file: {output_path}")
    print(f"Antibody embedding: {args.ab_embedding}")
    print(f"Antigen embedding: {args.ag_embedding}")
    print()

    create_dict_pre_cal(
        interim_dir=args.interim_dir,
        output_path=output_path,
        ab_embedding=args.ab_embedding,
        ag_embedding=args.ag_embedding,
        id_list_file=args.id_list
    )


if __name__ == "__main__":
    main()
