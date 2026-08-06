"""
Generate AntiBERTy embeddings for antibody sequences in AsEP dataset
"""
import os
import warnings

import numpy as np
import torch
from Bio import SeqIO
from tqdm import tqdm
from antiberty import AntiBERTyRunner

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
output_dir = os.path.join(asep_data_dir, "antibody/antiberty_embeddings")
os.makedirs(output_dir, exist_ok=True)

# Initialize AntiBERTy
antiberty = AntiBERTyRunner()

def embed_antiberty(protein_sequence: str) -> torch.Tensor:
    """
    Generate AntiBERTy embeddings for a given protein sequence.

    Args:
        protein_sequence (str): The protein sequence to embed.

    Returns:
        torch.Tensor: A tensor of shape (L, 512) where L is the length of the sequence.
    """
    sequences = [protein_sequence]
    embeddings = antiberty.embed(sequences)
    residue_embeddings = embeddings[0][1:-1, :]  # Remove CLS and SEP tokens
    return residue_embeddings

# Process FASTA file
fasta_sequences = list(SeqIO.parse(open(asep_ab_ag_sequences_fasta_path), 'fasta'))
antiberty_embeddings = {}

counter = 0
for fasta in tqdm(fasta_sequences, desc="Processing complexes"):
    name, sequence = fasta.id, str(fasta.seq)
    pdb_id = name.split("|")[0]

    # Skip known problematic complexes (seqres2cdr_seq and atmseq2cdr have different lengths)
    if pdb_id in ["5ies_0P"]:
        continue

    # Check sequence format
    sequence_parts = sequence.split(":")
    if len(sequence_parts) < 2:
        print(f"Skipping {pdb_id}: invalid sequence format (expected H:L:AG)")
        continue

    # Check graph file exists
    graph_path = os.path.join(asep_graphs_dir, f"{pdb_id}.pt")
    if not os.path.exists(graph_path):
        print(f"Skipping {pdb_id}: graph file not found")
        continue

    # Extract heavy and light chains
    H_chain = sequence_parts[0]
    L_chain = sequence_parts[1]
    full_ab_sequence = H_chain + L_chain

    # Generate AntiBERTy embeddings
    ab_embeddings = embed_antiberty(full_ab_sequence)

    # Load graph file to get CDR mapping
    asep_graphs_file = torch.load(graph_path)
    seqres2cdr_mask = torch.tensor(asep_graphs_file["mapping"]["ab"]["seqres2cdr"]).bool()

    antiberty_embeddings[pdb_id] = ab_embeddings[seqres2cdr_mask].numpy()

    counter += 1
    if counter % 100 == 0:
        print(f"{counter} samples processed...")

# Save embeddings
output_path = os.path.join(output_dir, "asep_antiberty_embeddings.pt")
torch.save(antiberty_embeddings, output_path)
print(f"Saved AntiBERTy embeddings for {len(antiberty_embeddings)} complexes to {output_path}")




"""
python data/embed_antiberty.py

Saved AntiBERTy embeddings for 1721 complexes to 
"""