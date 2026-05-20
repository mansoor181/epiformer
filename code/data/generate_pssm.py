import os
import numpy as np
from Bio import SeqIO
from Bio.Blast.Applications import NcbipsiblastCommandline
from multiprocessing import Pool
import pickle
from tqdm import tqdm
import re

# Configuration
BLAST_DB = "blastdb/uniref50_db"
NUM_CORES = os.cpu_count() - 16  # Leave one core free
OUTPUT_PKL = "ab_ag_pssm_dict.pkl"
TEMP_DIR = "pssm_temp"
os.makedirs(TEMP_DIR, exist_ok=True)

def parse_sequence_id(header):
    """Extract PDB ID and chain types from header"""
    pdb_id = header.split("|")[0]
    chains = header.split("|")[1].split(":")
    return pdb_id, chains

def run_psiblast(fasta_file):
    """Run PSI-BLAST on a single sequence file"""
    pdb_id = os.path.basename(fasta_file).split(".")[0]
    pssm_file = os.path.join(TEMP_DIR, f"{pdb_id}.pssm")
    
    try:
        cline = NcbipsiblastCommandline(
            cmd="psiblast",
            query=fasta_file,
            db=BLAST_DB,
            num_iterations=3,
            out_ascii_pssm=pssm_file,
            outfmt=6  # Also save tabular output
        )
        stdout, stderr = cline()
        return pdb_id, pssm_file
    except Exception as e:
        print(f"Failed on {pdb_id}: {str(e)}")
        return pdb_id, None

def parse_pssm(pssm_file):
    """Parse PSSM file into (seq_len, 20) numpy array"""
    with open(pssm_file) as f:
        lines = [l.strip() for l in f if l.strip()]
    
    # Find matrix start
    start_idx = 0
    while start_idx < len(lines) and not lines[start_idx].startswith("Last"):
        start_idx += 1
    start_idx += 1
    
    pssm = []
    for line in lines[start_idx:]:
        if not line[0].isdigit():
            continue
        scores = list(map(int, line.split()[2:22]))  # AA scores columns
        pssm.append(scores)
    
    return np.array(pssm)

def process_sequence(record):
    """Process a single sequence record"""
    pdb_id, chains = parse_sequence_id(record.id)
    
    # Save temporary FASTA files for each chain
    chain_files = {}
    for chain_type in ['H', 'L', 'A', 'B', 'C', 'D', 'E', 'F']:  # Common chain types
        if chain_type in chains:
            chain_seq = str(record.seq).split(":")[chains.index(chain_type)]
            chain_file = os.path.join(TEMP_DIR, f"{pdb_id}_{chain_type}.fasta")
            with open(chain_file, "w") as f:
                f.write(f">{pdb_id}_{chain_type}\n{chain_seq}\n")
            chain_files[chain_type] = chain_file
    
    # Run PSI-BLAST for each chain in parallel
    with Pool(NUM_CORES) as pool:
        results = pool.map(run_psiblast, chain_files.values())
    
    # Organize results
    pssm_data = {}
    for chain_type, (_, pssm_file) in zip(chain_files.keys(), results):
        if pssm_file:
            pssm = parse_pssm(pssm_file)
            if 'H' in chain_type or 'L' in chain_type:
                pssm_data.setdefault('ab', []).append((chain_type, pssm))
            else:
                pssm_data.setdefault('ag', []).append((chain_type, pssm))
    
    # Combine antibody chains (H first, then L)
    ab_pssm = None
    if 'ab' in pssm_data:
        ab_pssm = np.concatenate([pssm for _, pssm in sorted(pssm_data['ab'], 
                                key=lambda x: x[0])])  # Sort by chain type
    
    # Combine antigen chains
    ag_pssm = None
    if 'ag' in pssm_data:
        ag_pssm = np.concatenate([pssm for _, pssm in pssm_data['ag']])
    
    # Clean up temporary files
    for f in chain_files.values():
        if os.path.exists(f):
            os.remove(f)
    
    return pdb_id, {'ab': ab_pssm, 'ag': ag_pssm}

def main():
    # Load all sequences
    records = list(SeqIO.parse("ab_ag_seq.fasta", "fasta"))
    
    # Process in parallel with progress bar
    results = {}
    with Pool(NUM_CORES) as pool:
        for pdb_id, data in tqdm(pool.imap(process_sequence, records), total=len(records)):
            if data['ab'] is not None or data['ag'] is not None:
                results[pdb_id] = data
    
    # Save final dictionary
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(results, f)
    
    print(f"\nSaved PSSM data for {len(results)} complexes to {OUTPUT_PKL}")

if __name__ == "__main__":
    main()


"""
Example usage:
    python generate_pssm.py
"""