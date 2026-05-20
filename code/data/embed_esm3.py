"""
Generate ESM3 small embeddings and combine with ESM2 embeddings
"""
import os
import sys
import torch
import numpy as np
from Bio import SeqIO
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Force ESM3 to use GPU 1 by making it appear as GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ESM3 imports
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../../walle')))

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

# Output paths
esm2_embeddings_path = os.path.join(output_dir, "ag_esm2_embeddings_asep.pt")
esm3_embeddings_path = os.path.join(output_dir, "ag_esm3_embeddings_asep.pt")
combined_embeddings_path = os.path.join(output_dir, "ag_esm_embeddings_asep.pt")

# Load ESM3 small model
print("Loading ESM3 small model on GPU 1...")
try:
    client_esm3_small = ESM3.from_pretrained("esm3_sm_open_v1")
    print(f"✓ ESM3 small model loaded successfully on GPU {torch.cuda.current_device()}")
except Exception as e:
    print(f"✗ Failed to load ESM3 model: {e}")
    exit(1)

def embed_esm3_small(client, sequence: str) -> torch.Tensor:
    """Generate ESM3 small embeddings for a given protein sequence."""
    protein = ESMProtein(sequence=sequence)
    protein_tensor = client.encode(protein)
    
    output = client.forward_and_sample(
        protein_tensor,
        SamplingConfig(return_per_residue_embeddings=True)
    )

    residue_embeddings = output.per_residue_embedding

    residue_embeddings = residue_embeddings[ 1:-1, :]
    
    return residue_embeddings

# STEP 1: Generate ESM3 embeddings
print("=" * 50)
print("STEP 1: Generating ESM3 embeddings")
print("=" * 50)

# Check if ESM3 embeddings already exist
if os.path.exists(esm3_embeddings_path):
    print("Loading existing ESM3 embeddings...")
    try:
        ag_esm3_embeddings = torch.load(esm3_embeddings_path, weights_only=False)
        print(f"✓ Found {len(ag_esm3_embeddings['esm3_small'])} existing ESM3 embeddings")
    except:
        print("Failed to load existing ESM3 embeddings, starting fresh...")
        ag_esm3_embeddings = {'esm3_small': {}}
else:
    ag_esm3_embeddings = {'esm3_small': {}}

# Process FASTA file for ESM3
fasta_sequences = list(SeqIO.parse(open(asep_ab_ag_sequences_fasta_path), 'fasta'))

# Create mapping from PDB ID to sequence
pdb_to_sequence = {}
for fasta in fasta_sequences:
    name, sequence = fasta.id, str(fasta.seq)
    pdb_id = name.split("|")[0]
    if pdb_id not in ["5ies_0P"]:
        try:
            ag_chain = sequence.split(":")[2]
            pdb_to_sequence[pdb_id] = ag_chain
        except IndexError:
            continue

# Get PDB IDs that need ESM3 processing
processed_esm3_ids = set(ag_esm3_embeddings['esm3_small'].keys())
pdb_ids_to_process = set(pdb_to_sequence.keys()) - processed_esm3_ids

print(f"Total sequences available: {len(pdb_to_sequence)}")
print(f"Already processed ESM3: {len(processed_esm3_ids)}")
print(f"Need to process ESM3: {len(pdb_ids_to_process)}")

# Generate ESM3 embeddings
counter = 0
for pdb_id in tqdm(pdb_ids_to_process, desc="Generating ESM3 embeddings"):
    try:
        ag_chain = pdb_to_sequence[pdb_id]
        
        if counter % 10 == 0:
            print(f"Processing {pdb_id}: {ag_chain[:50]}...")
        
        # Generate ESM3 embeddings
        ag_embeddings_esm3 = embed_esm3_small(client_esm3_small, ag_chain)
        
        # Load graph file to get surface mapping
        asep_graphs_file = torch.load(os.path.join(asep_graphs_dir, f"{pdb_id}.pt"), weights_only=False)
        seqres2surf_mask = torch.tensor(asep_graphs_file["mapping"]["ag"]["seqres2surf"]).bool()
        
        # Apply surface mapping and store embeddings
        ag_esm3_embeddings['esm3_small'][pdb_id] = ag_embeddings_esm3[seqres2surf_mask].cpu().numpy()
        
        counter += 1
        if counter % 50 == 0:
            print(f"{counter} ESM3 embeddings processed...")
            # Save intermediate results
            torch.save(ag_esm3_embeddings, esm3_embeddings_path)
            print("Intermediate ESM3 embeddings saved")
            
    except Exception as e:
        print(f"Error processing {pdb_id} for ESM3: {str(e)}")
        continue

# Save ESM3 embeddings
print(f"Saving ESM3 embeddings to {esm3_embeddings_path}")
torch.save(ag_esm3_embeddings, esm3_embeddings_path)
print(f"✓ Saved {len(ag_esm3_embeddings['esm3_small'])} ESM3 embeddings")

# STEP 2: Load ESM2 embeddings and combine
print("=" * 50)
print("STEP 2: Combining with ESM2 embeddings")
print("=" * 50)

# Load ESM2 embeddings
try:
    print("Loading ESM2 embeddings...")
    ag_esm2_embeddings = torch.load(esm2_embeddings_path, weights_only=False)
    print(f"✓ Loaded ESM2 embeddings with keys: {list(ag_esm2_embeddings.keys())}")
    print(f"✓ Found {len(ag_esm2_embeddings['esm2_650m'])} ESM2 complexes")
except FileNotFoundError:
    print("✗ ESM2 embeddings file not found! Please run ESM2 script first.")
    exit(1)

# Create combined embeddings dictionary
combined_embeddings = {
    'esm2_650m': ag_esm2_embeddings['esm2_650m'],
    'esm2_3b': ag_esm2_embeddings['esm2_3b'],
    'esm3_small': ag_esm3_embeddings['esm3_small']
}

# Find common PDB IDs across all models
esm2_ids = set(combined_embeddings['esm2_650m'].keys())
esm3_ids = set(combined_embeddings['esm3_small'].keys())
common_ids = esm2_ids.intersection(esm3_ids)

print(f"ESM2 complexes: {len(esm2_ids)}")
print(f"ESM3 complexes: {len(esm3_ids)}")
print(f"Common complexes: {len(common_ids)}")

# Filter to only include common complexes
final_embeddings = {
    'esm2_650m': {pdb_id: combined_embeddings['esm2_650m'][pdb_id] for pdb_id in common_ids},
    'esm2_3b': {pdb_id: combined_embeddings['esm2_3b'][pdb_id] for pdb_id in common_ids},
    'esm3_small': {pdb_id: combined_embeddings['esm3_small'][pdb_id] for pdb_id in common_ids}
}

# Save combined embeddings
print(f"Saving combined embeddings to {combined_embeddings_path}")
torch.save(final_embeddings, combined_embeddings_path)

# Summary
print("=" * 50)
print("FINAL SUMMARY")
print("=" * 50)
print(f"Combined embeddings for {len(common_ids)} complexes")
print(f"Saved to: {combined_embeddings_path}")

if len(common_ids) > 0:
    example_pdb = list(common_ids)[0]
    print(f"\nExample embedding shapes for {example_pdb}:")
    print(f"  ESM2 650M: {final_embeddings['esm2_650m'][example_pdb].shape}")
    print(f"  ESM2 3B: {final_embeddings['esm2_3b'][example_pdb].shape}")
    print(f"  ESM3 Small: {final_embeddings['esm3_small'][example_pdb].shape}")

print(f"\nFiles created:")
print(f"  ESM3 only: {esm3_embeddings_path}")
print(f"  Combined: {combined_embeddings_path}")
