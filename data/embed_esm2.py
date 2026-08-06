"""
Generate ESM2 embeddings (650M and 3B) for antigen sequences in AsEP dataset
"""
import os
import warnings

import numpy as np
import torch
import esm
from Bio import SeqIO
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Configuration
# Note: Adjust proj_dir based on your directory structure
# Default assumes running from code/data/ with data at ../../data/asep/
proj_dir = os.path.join(os.getcwd(), '../../')
dataset_dir = os.path.join(proj_dir, "data/")
asep_data_dir = os.path.join(dataset_dir, "asep/")
asep_graphs_dir = os.path.join(asep_data_dir, "asepv1_interim_graphs/")
asep_sequences_dir = os.path.join(asep_data_dir, "sequences/")
asep_ab_ag_sequences_fasta_path = os.path.join(asep_sequences_dir, "asep_ab_ag_seqres_1723.fasta")
output_dir = os.path.join(asep_data_dir, "antigen/plm_embeddings")
os.makedirs(output_dir, exist_ok=True)

# Load both ESM-2 models
print("Loading ESM2 650M model...")
model_650m, alphabet_650m = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter_650m = alphabet_650m.get_batch_converter()
model_650m.eval()

print("Loading ESM2 3B model...")
model_3b, alphabet_3b = esm.pretrained.esm2_t36_3B_UR50D()
batch_converter_3b = alphabet_3b.get_batch_converter()
model_3b.eval()

def embed_esm2_650m(model, batch_converter, sequence: str) -> torch.Tensor:
    """
    Generate ESM2 650M embeddings for a given protein sequence.
    Args:
        model: ESM2 650M model
        batch_converter: batch converter for 650M model
        sequence (str): The protein sequence to embed.
    Returns:
        torch.Tensor: A tensor of shape (L, 1280) where L is the length of the sequence.
    """
    data = [("protein1", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33])  # Layer 33 is the final layer
        residue_embeddings = results["representations"][33]  # Shape: (1, sequence_length, 1280)
        # Remove the batch dimension and the start/end tokens
        residue_embeddings = residue_embeddings[0, 1:-1, :]
    return residue_embeddings

def embed_esm2_3b(model, batch_converter, sequence: str) -> torch.Tensor:
    """
    Generate ESM2 3B embeddings for a given protein sequence.
    Args:
        model: ESM2 3B model
        batch_converter: batch converter for 3B model
        sequence (str): The protein sequence to embed.
    Returns:
        torch.Tensor: A tensor of shape (L, 2560) where L is the length of the sequence.
    """
    data = [("protein1", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[36])  # Layer 36 is the final layer for 3B model
        residue_embeddings = results["representations"][36]  # Shape: (1, sequence_length, 2560)
        # Remove the batch dimension and the start/end tokens
        residue_embeddings = residue_embeddings[0, 1:-1, :]
    return residue_embeddings

# Process FASTA file
fasta_sequences = list(SeqIO.parse(open(asep_ab_ag_sequences_fasta_path), 'fasta'))

# Initialize embedding dictionaries with nested structure
ag_plm_embeddings = {
    'esm2_650m': {},
    'esm2_3b': {}
}

counter = 0
for fasta in tqdm(fasta_sequences, desc="Processing complexes"):
    name, sequence = fasta.id, str(fasta.seq)
    pdb_id = name.split("|")[0]

    # Skip known problematic complexes
    if pdb_id in ["5ies_0P"]:
        continue

    # Check sequence format
    sequence_parts = sequence.split(":")
    if len(sequence_parts) < 3:
        print(f"Skipping {pdb_id}: invalid sequence format (expected H:L:AG)")
        continue

    # Extract antigen chain
    ag_chain = sequence_parts[2]

    # Check graph file exists
    graph_path = os.path.join(asep_graphs_dir, f"{pdb_id}.pt")
    if not os.path.exists(graph_path):
        print(f"Skipping {pdb_id}: graph file not found")
        continue

    print(f"Processing {pdb_id}: {ag_chain[:50]}...")

    # Generate embeddings with both models
    ag_embeddings_650m = embed_esm2_650m(model_650m, batch_converter_650m, ag_chain)
    ag_embeddings_3b = embed_esm2_3b(model_3b, batch_converter_3b, ag_chain)

    # Load graph file to get surface mapping
    asep_graphs_file = torch.load(graph_path)
    seqres2surf_mask = torch.tensor(asep_graphs_file["mapping"]["ag"]["seqres2surf"]).bool()

    # Apply surface mapping and store embeddings
    ag_plm_embeddings['esm2_650m'][pdb_id] = ag_embeddings_650m[seqres2surf_mask].numpy()
    ag_plm_embeddings['esm2_3b'][pdb_id] = ag_embeddings_3b[seqres2surf_mask].numpy()

    counter += 1
    if counter % 10 == 0:
        print(f"{counter} samples processed...")

# Save embeddings
output_path = os.path.join(output_dir, "ag_esm2_embeddings_asep.pt")
torch.save(ag_plm_embeddings, output_path)

print(f"Saved ESM2 embeddings for {len(ag_plm_embeddings['esm2_650m'])} complexes to {output_path}")
print(f"ESM2 650M embedding shape example: {list(ag_plm_embeddings['esm2_650m'].values())[0].shape}")
print(f"ESM2 3B embedding shape example: {list(ag_plm_embeddings['esm2_3b'].values())[0].shape}")
