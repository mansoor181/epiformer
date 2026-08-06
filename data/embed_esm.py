"""
Generate ESM2 (via transformers) and ESM3 embeddings together
"""
import os
import torch
import numpy as np
from Bio import SeqIO
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ESM3 imports
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig

# ESM2 via transformers (avoids fair-esm conflict)
from transformers import EsmModel, AutoTokenizer

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

# Load models
print("Loading ESM2 650M via transformers...")
tokenizer_650m = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model_650m = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
model_650m.eval()

print("Loading ESM2 3B via transformers...")
tokenizer_3b = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
model_3b = EsmModel.from_pretrained("facebook/esm2_t36_3B_UR50D")
model_3b.eval()

print("Loading ESM3 small...")
client_esm3 = ESM3.from_pretrained("esm3_sm_open_v1")

def embed_esm2_650m_hf(model, tokenizer, sequence: str):
    inputs = tokenizer(sequence, return_tensors="pt", padding=True, truncation=False)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[0, 1:-1, :]  # Remove CLS/SEP
    return embeddings

def embed_esm2_3b_hf(model, tokenizer, sequence: str):
    inputs = tokenizer(sequence, return_tensors="pt", padding=True, truncation=False)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[0, 1:-1, :]  # Remove CLS/SEP
    return embeddings

def embed_esm3_small(client, sequence: str):
    protein = ESMProtein(sequence=sequence)
    protein_tensor = client.encode(protein)
    output = client.forward_and_sample(
        protein_tensor,
        SamplingConfig(return_per_residue_embeddings=True)
    )
    return output.per_residue_embedding[1:-1, :]  # Remove start/end tokens

# Initialize embeddings storage
ag_plm_embeddings = {
    'esm2_650m': {},
    'esm2_3b': {},
    'esm3_small': {}
}

# Process sequences
fasta_sequences = list(SeqIO.parse(open(asep_ab_ag_sequences_fasta_path), 'fasta'))

counter = 0
for fasta in tqdm(fasta_sequences, desc="Processing complexes"):
    name, sequence = fasta.id, str(fasta.seq)
    pdb_id = name.split("|")[0]
    
    if pdb_id in ["5ies_0P"]:
        continue
    
    ag_chain = sequence.split(":")[2]

    if counter % 10 == 0:
        print(f"Processing {pdb_id}: {ag_chain[:50]}...")

    # Generate all embeddings
    ag_embeddings_650m = embed_esm2_650m_hf(model_650m, tokenizer_650m, ag_chain)
    ag_embeddings_3b = embed_esm2_3b_hf(model_3b, tokenizer_3b, ag_chain)
    ag_embeddings_esm3 = embed_esm3_small(client_esm3, ag_chain)

    # Load graph and apply surface mapping
    asep_graphs_file = torch.load(os.path.join(asep_graphs_dir, f"{pdb_id}.pt"), weights_only=False)
    seqres2surf_mask = torch.tensor(asep_graphs_file["mapping"]["ag"]["seqres2surf"]).bool()

    # Store embeddings
    ag_plm_embeddings['esm2_650m'][pdb_id] = ag_embeddings_650m[seqres2surf_mask].cpu().numpy()
    ag_plm_embeddings['esm2_3b'][pdb_id] = ag_embeddings_3b[seqres2surf_mask].cpu().numpy()
    ag_plm_embeddings['esm3_small'][pdb_id] = ag_embeddings_esm3[seqres2surf_mask].cpu().numpy()

    counter += 1
    if counter % 100 == 0:
        print(f"{counter} samples processed...")
        output_path = os.path.join(output_dir, "ag_esm_plm_embeddings.pt")
        torch.save(ag_plm_embeddings, output_path)

    # if counter == 5:  # Remove for full processing
    #     break
       

# Save results
output_path = os.path.join(output_dir, "ag_esm_plm_embeddings.pt")
torch.save(ag_plm_embeddings, output_path)
print(f"Saved combined embeddings to {output_path}")



print(f"\n=== Final Summary ===")
print(f"Saved embeddings for {len(ag_plm_embeddings['esm2_650m'])} complexes to {output_path}")
print(f"Generated embedding types: {list(ag_plm_embeddings.keys())}")

# Show embedding dimensions
if len(ag_plm_embeddings['esm2_650m']) > 0:
    print(f"\nEmbedding dimensions:")
    print(f"  ESM2 650M: {list(ag_plm_embeddings['esm2_650m'].values())[0].shape} (1280 dims)")
    print(f"  ESM2 3B: {list(ag_plm_embeddings['esm2_3b'].values())[0].shape} (2560 dims)")
    if 'esm3_small' in ag_plm_embeddings and len(ag_plm_embeddings['esm3_small']) > 0:
        print(f"  ESM3 Small: {list(ag_plm_embeddings['esm3_small'].values())[0].shape} (1536 dims)")
