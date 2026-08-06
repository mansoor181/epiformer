import os
import argparse
import logging
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from Bio import AlignIO, SeqIO
from Bio.Align.Applications import ClustalOmegaCommandline
from Bio.PDB import PDBIO, PDBParser, Select
from Bio.Seq import Seq
from Bio.SeqIO import SeqRecord
from biopandas.pdb import PandasPdb

warnings.filterwarnings("ignore")
CLUSTAL_OMEGA_EXECUTABLE = shutil.which("clustalo")

AA_MAP = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G",
    "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
    "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V",
    "TRP": "W", "TYR": "Y"
}



# align seq using ClustalOmega
def run_align_clustalomega(clustal_omega_executable: str,
                           seq1: str = None, seq2: str = None,
                           seqs: List[str] = None) -> List[SeqRecord]:
    """

    Args:
        seq1: sequence of a chain e.g. seqres sequence
        seq2: sequence of a chain e.g. atmseq sequence
        or you can provide a list of strings using seqs
        seqs: e.g. ["seq1", "seq2", ...]
        clustal_omega_executable: (str) path to clustal omega executable
            e.g. "/usr/local/bin/clustal-omega"
    Returns:
        aln_seq_records: (List)
    """
    # assert input
    if seqs is None and (seq1 is None or seq2 is None):
        raise NotImplemented(f"Provide either List of seqs as `seqs` OR a pair of seqs as `seq1` and `seq2`.")

    # generate seq_recs
    seq_rec = [None]
    if seqs:
        seq_rec = [SeqRecord(id=f"seq{i + 1}", seq=Seq(seqs[i]), description="")
                   for i in range(len(seqs))]
    elif seq1 is not None and seq2 is not None:
        seq_rec = [SeqRecord(id=f"seq{1}", seq=Seq(seq1), description=""),
                   SeqRecord(id=f"seq{2}", seq=Seq(seq2), description="")]

    with tempfile.TemporaryDirectory() as tmpdir:
        # executable
        cmd = clustal_omega_executable

        # create input seq fasta file and output file for clustal-omega
        in_file = os.path.join(tmpdir, "seq.fasta")
        out_file = os.path.join(tmpdir, f"aln.fasta")
        with open(in_file, "w") as f:
            SeqIO.write(seq_rec, f, "fasta")
        # create Clustal-Omega commands
        clustalomega_cline = ClustalOmegaCommandline(cmd=cmd, infile=in_file, outfile=out_file, verbose=True, auto=True)

        # run Clustal-Omega
        stdout, stderr = clustalomega_cline()

        # read aln
        aln_seq_records = []
        with open(out_file, "r") as f:
            for record in AlignIO.read(f, "fasta"):
                aln_seq_records.append(record)

        return aln_seq_records
    
# align ATOMSEQ to SEQRES
"""
FIXME: 
- keep log of the antigen seqres with alignment error
"""

def get_seqres2atmseq_mask(seqres, atmseq, pdbid):
    try:
        aln = run_align_clustalomega(
            clustal_omega_executable=CLUSTAL_OMEGA_EXECUTABLE,
            seq1=seqres,
            seq2=atmseq,
        )

        # Check if seqres contains dash
        if "-" in str(aln[0].seq):
            raise ValueError("Error: seqres contains dash")

        aln1 = str(aln[1].seq)  # atmseq in aln may contain "-"
        seqres2atmseq = [
            1 if i != "-" else 0 for i in aln1
        ]  # 1 => in atmseq; 0 => not in atmseq

        # Ensure the lengths match
        if len(seqres2atmseq) != len(seqres):
            raise ValueError("Error: Length mismatch between seqres2atmseq and seqres")

        return seqres2atmseq
    
    except Exception as e:
        # Log the error with the PDB ID
        logging.error(f"PDB ID {pdbid}: {e}")
        return None  # Return None or an empty list to indicate failure




def split_complex_reindex_antibody_chains(pdb_path, pt_graphs_dir, pdb_id, ab_out_dir):
    
    ppdb = PandasPdb().read_pdb(pdb_path)
    atomic_df = ppdb.get_model(1).df["ATOM"]

    mask_data = torch.load(f"{pt_graphs_dir}/{pdb_id}.pt")
    output_path = os.path.join(ab_out_dir, f"{pdb_id}_ab.pdb")

    antibody_chains = {"H", "L"}

    # Process heavy and light chains separately
    chain_data = {}

    # Create a copy of the original DataFrame for antibody chains only
    ab_df = ppdb.df["ATOM"][ppdb.df["ATOM"]["chain_id"].isin(antibody_chains)].copy()
 

    for chain_type in ["H", "L"]:
        if chain_type not in antibody_chains:
            continue

        chain_df = ab_df[ab_df["chain_id"] == chain_type]

        # chain_df = atomic_df[atomic_df["chain_id"] == chain_type]

        if chain_df.empty:
            continue

        # Get SEQRES and ATMSEQ for the chain
        seqres = str(np.array(mask_data["seqres"]["ab"][chain_type]))

        atmseq_df = atomic_df[atomic_df["chain_id"] == chain_type]  # NEW LINE
        atmseq_df = atmseq_df[["residue_number", "residue_name"]].drop_duplicates()

        """
        BUG: 
        - incorrect atmseq (didn't include alternate residues) which lead to incorrect alignment
        - the following code is for correct atmseq filtering
        """

        # Process ATMSEQ with alternates preserved
        # First get ALL residues in original order (including alternates)
        atmseq_full = chain_df.assign(
            full_residue=chain_df["residue_number"].astype(str) + chain_df["insertion"].fillna('')
        )

        # Get ordered unique residues (with alternates)
        residues_ordered = atmseq_full["full_residue"].unique()

        # Now get ATMSEQ string with original residues (including alternates)
        atmseq_df = atmseq_full.drop_duplicates("full_residue")
        atmseq = "".join(atmseq_df["residue_name"].map(AA_MAP))

        # Generate alignment mask
        mask = get_seqres2atmseq_mask(seqres, atmseq, pdb_id)

        # Create full residue identifiers including insertion codes
        chain_df["full_residue"] = chain_df["residue_number"].astype(str) + \
                                chain_df["insertion"].fillna('')

        # Create 1-based consecutive indices for all residues
        new_indices_list = [i for i, bit in enumerate(mask) if bit == 1]
        new_indices = {res: new_index for res, new_index in zip(residues_ordered, new_indices_list)}

        # Apply mapping directly to the DataFrame
        chain_df["new_residue_number"] = chain_df["full_residue"].map(new_indices)

        ab_df.loc[chain_df.index, "residue_number"] = chain_df["new_residue_number"]
        ab_df.loc[chain_df.index, "insertion"] = ""  # Clear insertion codes
     
        # ppdb.df["ATOM"].loc[chain_df.index, "residue_number"] = chain_df["new_residue_number"]

        chain_data[chain_type] = (seqres, atmseq, mask)


    # Save only antibody chains with new numbering
    ppdb_ab = PandasPdb()
    ppdb_ab.df["ATOM"] = ab_df
    ppdb_ab.to_pdb(path=output_path, 
                  records=["ATOM"],
                  gz=False,
                  append_newline=True)


    return chain_data



def main():
    parser = argparse.ArgumentParser(description="Process antibody PDB files with index reset")
    parser.add_argument("input_dir", type=Path, help="Input PDB directory")
    parser.add_argument("pt_graphs_dir", type=Path, help="PyTorch graphs directory")
    parser.add_argument("output_dir", type=Path, help="Output directory for processed PDBs")
    parser.add_argument("metadata_dir", type=Path, help="Output directory for alignment metadata")
    args = parser.parse_args()

    logging.basicConfig(filename=args.metadata_dir/'alignment_errors.log', 
                        level=logging.ERROR,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    metadata_list = []
    for pdb_file in args.input_dir.glob("*.pdb"):
        # print(pdb_file)
        pdb_id = pdb_file.stem.split(".")[0]

        chain_data = split_complex_reindex_antibody_chains(str(pdb_file), args.pt_graphs_dir,
                                 pdb_id, args.output_dir)
        
        if chain_data:
            metadata_entry = {
                "pdb_id": pdb_id,
                "heavy_seqres": chain_data.get("H", (None, None, None))[0],
                "heavy_atmseq": chain_data.get("H", (None, None, None))[1],
                "heavy_seqres2atmseq_mask": chain_data.get("H", (None, None, None))[2],
                "light_seqres": chain_data.get("L", (None, None, None))[0],
                "light_atmseq": chain_data.get("L", (None, None, None))[1],
                "light_seqres2atmseq_mask": chain_data.get("L", (None, None, None))[2],
                "seqres2atmseq_mask":  chain_data.get("H", (None, None, None))[2] +
                                        chain_data.get("L", (None, None, None))[2]
            }
            metadata_list.append(metadata_entry)

    # Save metadata
    pd.DataFrame(metadata_list).to_csv(args.metadata_dir/"seqres2atmseq_mask_ab_HL_chain.csv", index=False)
    print(f"Processed {len(metadata_list)} antibody structures")

if __name__ == "__main__":
    main()






"""
python3 reindex_ab_split_complex.py  \
     \
     \
    
"""

"""
python3 reindex_ab_split_complex.py  \
     \
     \
    
"""




# ************************ old script using biopython for reindexing ******************************************** #



# import os
# import argparse, torch
# import pandas as pd
# import numpy as np
# from pathlib import Path
# from biopandas.pdb import PandasPdb
# from Bio.PDB import PDBIO, PDBParser, Select
# import shutil
# import tempfile
# from typing import List
# from Bio import AlignIO, SeqIO
# from Bio.Align.Applications import ClustalOmegaCommandline
# from Bio.Seq import Seq
# from Bio.SeqIO import SeqRecord
# import logging
# import warnings

# warnings.filterwarnings("ignore")
# CLUSTAL_OMEGA_EXECUTABLE = shutil.which("clustalo")

# AA_MAP = {
#     "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G",
#     "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
#     "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V",
#     "TRP": "W", "TYR": "Y"
# }

# class ChainSelect(Select):
#     def __init__(self, chains):
#         self.chains = set(chains)
    
#     def accept_chain(self, chain):
#         return chain.get_id() in self.chains
       
#     """
#     TODO: 
#     - this method ensures that heteratoms are removed from the chain
#     - also remove alternate conformations (A, B, etc.) and keep main conformation only
#         - causes mismatch in seqres and atmseq, even after seqres2atmseq masking
#     """
#     def accept_residue(self, residue):
#         # Ensure only standard residues (no heteroatoms) and no alternate conformations (A, B, etc.)
#         return residue.id[0] == " "  # and residue.id[2] == " "  # Keep only residues with an empty insertion code

   

# # align seq using ClustalOmega
# def run_align_clustalomega(clustal_omega_executable: str,
#                            seq1: str = None, seq2: str = None,
#                            seqs: List[str] = None) -> List[SeqRecord]:
#     """

#     Args:
#         seq1: sequence of a chain e.g. seqres sequence
#         seq2: sequence of a chain e.g. atmseq sequence
#         or you can provide a list of strings using seqs
#         seqs: e.g. ["seq1", "seq2", ...]
#         clustal_omega_executable: (str) path to clustal omega executable
#             e.g. "/usr/local/bin/clustal-omega"
#     Returns:
#         aln_seq_records: (List)
#     """
#     # assert input
#     if seqs is None and (seq1 is None or seq2 is None):
#         raise NotImplemented(f"Provide either List of seqs as `seqs` OR a pair of seqs as `seq1` and `seq2`.")

#     # generate seq_recs
#     seq_rec = [None]
#     if seqs:
#         seq_rec = [SeqRecord(id=f"seq{i + 1}", seq=Seq(seqs[i]), description="")
#                    for i in range(len(seqs))]
#     elif seq1 is not None and seq2 is not None:
#         seq_rec = [SeqRecord(id=f"seq{1}", seq=Seq(seq1), description=""),
#                    SeqRecord(id=f"seq{2}", seq=Seq(seq2), description="")]

#     with tempfile.TemporaryDirectory() as tmpdir:
#         # executable
#         cmd = clustal_omega_executable

#         # create input seq fasta file and output file for clustal-omega
#         in_file = os.path.join(tmpdir, "seq.fasta")
#         out_file = os.path.join(tmpdir, f"aln.fasta")
#         with open(in_file, "w") as f:
#             SeqIO.write(seq_rec, f, "fasta")
#         # create Clustal-Omega commands
#         clustalomega_cline = ClustalOmegaCommandline(cmd=cmd, infile=in_file, outfile=out_file, verbose=True, auto=True)

#         # run Clustal-Omega
#         stdout, stderr = clustalomega_cline()

#         # read aln
#         aln_seq_records = []
#         with open(out_file, "r") as f:
#             for record in AlignIO.read(f, "fasta"):
#                 aln_seq_records.append(record)

#         return aln_seq_records
    
# # align ATOMSEQ to SEQRES
# """
# FIXME: 
# - keep log of the antigen seqres with alignment error
# """

# def get_seqres2atmseq_mask(seqres, atmseq, pdbid):
#     try:
#         aln = run_align_clustalomega(
#             clustal_omega_executable=CLUSTAL_OMEGA_EXECUTABLE,
#             seq1=seqres,
#             seq2=atmseq,
#         )

#         # Check if seqres contains dash
#         if "-" in str(aln[0].seq):
#             raise ValueError("Error: seqres contains dash")

#         aln1 = str(aln[1].seq)  # atmseq in aln may contain "-"
#         seqres2atmseq = [
#             1 if i != "-" else 0 for i in aln1
#         ]  # 1 => in atmseq; 0 => not in atmseq

#         # Ensure the lengths match
#         if len(seqres2atmseq) != len(seqres):
#             raise ValueError("Error: Length mismatch between seqres2atmseq and seqres")

#         return seqres2atmseq
    
#     except Exception as e:
#         # Log the error with the PDB ID
#         logging.error(f"PDB ID {pdbid}: {e}")
#         return None  # Return None or an empty list to indicate failure

# def get_pdb_metadata(pdb_path, selected_chains):
#     with open(pdb_path, "r") as f:
#         lines = f.readlines()
    
#     metadata = []
#     for line in lines:
#         if line.startswith(("HEADER", "TITLE", "REMARK", "MODRES")):
#             chain_id = line.split()[2] if len(line.split()) > 2 else None
#             if chain_id in selected_chains:
#                 metadata.append(line)
#         elif line.startswith("SEQRES"):
#             chain_id = line.split()[2]
#             if chain_id in selected_chains:
#                 metadata.append(line)
    
#     return "".join(metadata)




# def process_antibody_chains(pdb_path, structure, antibody_chains, pt_graphs_dir, pdb_id):
#     atomic_df = PandasPdb().read_pdb(pdb_path).get_model(1).df["ATOM"]
#     mask_data = torch.load(f"{pt_graphs_dir}/{pdb_id}.pt")

#     # Process heavy and light chains separately
#     chain_data = {}
#     for chain_type in ["H", "L"]:
#         if chain_type not in antibody_chains:
#             continue

#         chain_df = atomic_df[atomic_df["chain_id"] == chain_type]
#         if chain_df.empty:
#             continue

#         # Get SEQRES and ATMSEQ for the chain
#         seqres = str(np.array(mask_data["seqres"]["ab"][chain_type]))

#         atmseq_df = atomic_df[atomic_df["chain_id"] == chain_type]  # NEW LINE
#         atmseq_df = atmseq_df[["residue_number", "residue_name"]].drop_duplicates()

#         """
#         BUG: 
#         - wrong atmseq (didn't include alternate residues) which lead to incorrect alignment
#         - the following code is for correct atmseq filtering
#         """
#         # atmseq = "".join(atmseq_df["residue_name"].map(AA_MAP))

#         # *********************************************

#         # Process ATMSEQ with alternates preserved
#         # First get ALL residues in original order (including alternates)
#         atmseq_full = chain_df.assign(
#             full_residue=chain_df["residue_number"].astype(str) + chain_df["insertion"].fillna('')
#         )
        
#         # Now get ATMSEQ string with original residues (including alternates)
#         atmseq_df = atmseq_full.drop_duplicates("full_residue")
#         atmseq = "".join(atmseq_df["residue_name"].map(AA_MAP))

#         # *********************************************

#         """
#         TODO: 
#         - save seqres2atmseq_mask 
#         - reset indices by including the alternate conformations of residues
#         - inflate the residue indices based on the alternate conformations
#         - for example: [70, 70A, 70B, 72] => [70, 71, 72, 74]
#         """
        
#         # Generate alignment mask seqres2atmseq
#         mask = get_seqres2atmseq_mask(seqres, atmseq, pdb_id)
#         if mask is None:
#             continue


#         # Create residue mapping
#         old_indices = chain_df["residue_number"].unique()

#         # old_indices = np.arange(len(seqres))

#         offset = 1000 if chain_type == "H" else 2000  # Prevent index overlap between chains

#         # Two-step remapping:
#         # 1. First, offset all residue numbers to avoid collisions
#         temp_mapping = {old: old + offset for old in old_indices}

#         # 2. Then map to final indices based on the alignment mask
#         #    (only residues where mask==1 are kept, others are discarded)
#         new_indices = [i for i, bit in enumerate(mask) if bit == 1]
#         final_mapping = {temp: idx for temp, idx in zip(temp_mapping.values(), new_indices)}

#         # Apply mapping to structure
#         structure = apply_chain_mapping(structure, chain_type, temp_mapping)
#         structure = apply_chain_mapping(structure, chain_type, final_mapping)

#         chain_data[chain_type] = (seqres, atmseq, mask)
    
    
#     return structure, chain_data

# def apply_chain_mapping(structure, chain_id, mapping):
#     """
#     Updates residue numbers in a specific chain of a structure according to a mapping.
    
#     Args:
#         structure: The biopython Structure object to modify
#         chain_id: The ID of the chain to modify (e.g., "H" or "L")
#         mapping: Dictionary {old_residue_number: new_residue_number}
    
#     Returns:
#         The modified structure with updated residue numbers
#     """
#     # Iterate through all models in the structure (typically just 1)
#     for model in structure:
#         # Find the specific chain we want to modify
#         for chain in model:
#             if chain.get_id() == chain_id:
#                 # Update each residue in the chain
#                 for residue in chain:
#                     # Get the current residue number (second element of the ID tuple)
#                     old_id = residue.get_id()[1]
                    
#                     # If this residue is in our mapping, update its number
#                     if old_id in mapping:
#                         # Residue ID is a tuple of (hetero flag, number, insertion code)
#                         # We preserve the original hetero flag and insertion code
#                         residue.id = (residue.id[0], mapping[old_id], residue.id[2])
    
#     return structure

# def write_pdb_with_metadata(output_path, metadata, structure, select):
#     with open(output_path, "w") as f:
#         f.write(metadata)
#         io = PDBIO()
#         io.set_structure(structure)
#         io.save(f, select=select)

# def split_complex(pdb_path, pt_graphs_dir, pdb_id, output_dir):
#     structure = PDBParser(QUIET=True).get_structure("complex", pdb_path)
#     antibody_chains = {"H", "L"}
    
#     # Process antibody chains
#     modified_structure, chain_data = process_antibody_chains(
#         pdb_path, structure, antibody_chains, pt_graphs_dir, pdb_id
#     )
    
#     if modified_structure:
#         metadata = get_pdb_metadata(pdb_path, antibody_chains)
#         output_path = os.path.join(output_dir, f"{pdb_id}_ab.pdb")
#         write_pdb_with_metadata(output_path, metadata, modified_structure, ChainSelect(antibody_chains))
#         return chain_data

# def main():
#     parser = argparse.ArgumentParser(description="Process antibody PDB files with index reset")
#     parser.add_argument("input_dir", type=Path, help="Input PDB directory")
#     parser.add_argument("pt_graphs_dir", type=Path, help="PyTorch graphs directory")
#     parser.add_argument("output_dir", type=Path, help="Output directory for processed PDBs")
#     parser.add_argument("metadata_dir", type=Path, help="Output directory for alignment metadata")
#     args = parser.parse_args()

#     logging.basicConfig(filename=args.metadata_dir/'alignment_errors.log', 
#                         level=logging.ERROR,
#                         format='%(asctime)s - %(levelname)s - %(message)s')

#     metadata_list = []
#     for pdb_file in args.input_dir.glob("*.pdb"):
#         # print(pdb_file)
#         pdb_id = pdb_file.stem.split(".")[0]

#         chain_data = split_complex(str(pdb_file), args.pt_graphs_dir,
#                                  pdb_id, args.output_dir)
        
#         if chain_data:
#             metadata_entry = {
#                 "pdb_id": pdb_id,
#                 "heavy_seqres": chain_data.get("H", (None, None, None))[0],
#                 "heavy_atmseq": chain_data.get("H", (None, None, None))[1],
#                 "heavy_seqres2atmseq_mask": chain_data.get("H", (None, None, None))[2],
#                 "light_seqres": chain_data.get("L", (None, None, None))[0],
#                 "light_atmseq": chain_data.get("L", (None, None, None))[1],
#                 "light_seqres2atmseq_mask": chain_data.get("L", (None, None, None))[2],
#                 "seqres2atmseq_mask":  chain_data.get("H", (None, None, None))[2] +
#                                         chain_data.get("L", (None, None, None))[2]
#             }
#             metadata_list.append(metadata_entry)

#     # Save metadata
#     pd.DataFrame(metadata_list).to_csv(args.metadata_dir/"seqres2atmseq_mask_ab_HL_chain.csv", index=False)
#     print(f"Processed {len(metadata_list)} antibody structures")

# if __name__ == "__main__":
#     main()









# ******************************************************************** #


# import os
# import argparse, torch
# import pandas as pd
# import numpy as np
# from pathlib import Path
# from biopandas.pdb import PandasPdb
# from Bio.PDB import PDBIO, PDBParser, Select
# import shutil
# import tempfile
# from typing import List
# from Bio import AlignIO, SeqIO
# from Bio.Align.Applications import ClustalOmegaCommandline
# from Bio.Seq import Seq
# from Bio.SeqIO import SeqRecord
# import logging


# import warnings
# warnings.filterwarnings("ignore")

# CLUSTAL_OMEGA_EXECUTABLE = shutil.which("clustalo")

# # Dictionary for mapping three-letter codes to one-letter amino acid codes
# AA_MAP = {
#     "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G",
#     "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
#     "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V",
#     "TRP": "W", "TYR": "Y"
# }

# class ChainSelect(Select):
#     """Select specific chains from a PDB structure."""
#     def __init__(self, chains):
#         self.chains = set(chains)
    
#     def accept_chain(self, chain):
#         return chain.get_id() in self.chains
    
#     """
#     TODO: 
#     - this method ensures that heteratoms are removed from the chain
#     - also remove alternate conformations (A, B, etc.) and keep main conformation only
#         - causes mismatch in seqres and atmseq, even after seqres2atmseq masking
#     """
#     def accept_residue(self, residue):
#         # Ensure only standard residues (no heteroatoms) and no alternate conformations (A, B, etc.)
#         return residue.id[0] == " " and residue.id[2] == " "  # Keep only residues with an empty insertion code

    
# def get_atmseq_seqres(pdb_structures_dir, pt_graphs_dir, pdb_id):
#     """
#     - loads a pdb and pytorch file and returns seqres and atmseq
#     """
#     atomic_df = PandasPdb().read_pdb(pdb_structures_dir + f"{pdb_id}_ab.pdb").get_model(1).df["ATOM"]
#     mask_data = torch.load(pt_graphs_dir + f'{pdb_id}.pt')

#     ag_chain = atomic_df["chain_id"].unique()[0]
#     seqres = np.array(mask_data["seqres"]["ab"][ag_chain])

#     # filtered_antigen_df  = atomic_df[atomic_df["residue_number"].map(seqres2surf_mask) == 1]
#     atomic_df = atomic_df[["residue_number", "residue_name"]].drop_duplicates()
#     atmseq = "".join(atomic_df["residue_name"].map(AA_MAP))

#     return seqres.item(), atmseq


# # align seq using ClustalOmega
# def run_align_clustalomega(clustal_omega_executable: str,
#                            seq1: str = None, seq2: str = None,
#                            seqs: List[str] = None) -> List[SeqRecord]:
#     """

#     Args:
#         seq1: sequence of a chain e.g. seqres sequence
#         seq2: sequence of a chain e.g. atmseq sequence
#         or you can provide a list of strings using seqs
#         seqs: e.g. ["seq1", "seq2", ...]
#         clustal_omega_executable: (str) path to clustal omega executable
#             e.g. "/usr/local/bin/clustal-omega"
#     Returns:
#         aln_seq_records: (List)
#     """
#     # assert input
#     if seqs is None and (seq1 is None or seq2 is None):
#         raise NotImplemented(f"Provide either List of seqs as `seqs` OR a pair of seqs as `seq1` and `seq2`.")

#     # generate seq_recs
#     seq_rec = [None]
#     if seqs:
#         seq_rec = [SeqRecord(id=f"seq{i + 1}", seq=Seq(seqs[i]), description="")
#                    for i in range(len(seqs))]
#     elif seq1 is not None and seq2 is not None:
#         seq_rec = [SeqRecord(id=f"seq{1}", seq=Seq(seq1), description=""),
#                    SeqRecord(id=f"seq{2}", seq=Seq(seq2), description="")]

#     with tempfile.TemporaryDirectory() as tmpdir:
#         # executable
#         cmd = clustal_omega_executable

#         # create input seq fasta file and output file for clustal-omega
#         in_file = os.path.join(tmpdir, "seq.fasta")
#         out_file = os.path.join(tmpdir, f"aln.fasta")
#         with open(in_file, "w") as f:
#             SeqIO.write(seq_rec, f, "fasta")
#         # create Clustal-Omega commands
#         clustalomega_cline = ClustalOmegaCommandline(cmd=cmd, infile=in_file, outfile=out_file, verbose=True, auto=True)

#         # run Clustal-Omega
#         stdout, stderr = clustalomega_cline()

#         # read aln
#         aln_seq_records = []
#         with open(out_file, "r") as f:
#             for record in AlignIO.read(f, "fasta"):
#                 aln_seq_records.append(record)

#         return aln_seq_records
    
# # align ATOMSEQ to SEQRES
# """
# FIXME: 
# - keep log of the antigen seqres with alignment error
# """




# # *********************************************

# def parse_residue(res):
#     """Extracts the base residue number from a residue identifier (e.g., '29A' → 29)."""
#     base_str = ''.join([c for c in str(res) if c.isdigit()])
#     return int(base_str) if base_str else None

# def group_residues(residues):
#     """Groups consecutive residues with the same base number."""
#     groups = []
#     current_group = []
#     prev_base = None
    
#     for res in residues:
#         base = parse_residue(res)
#         if base != prev_base:
#             if current_group:
#                 groups.append(current_group)
#                 current_group = []
#             prev_base = base
#         current_group.append(res)
    
#     if current_group:
#         groups.append(current_group)
    
#     return groups

# def convert_residues(residues_ordered):
#     """Converts residues with alternates into consecutive numbers, adjusting offsets."""
#     groups = group_residues(residues_ordered)
#     cumulative_offset = 0
#     converted_numbers = []
    
#     for group in groups:
#         base = parse_residue(group[0])
#         adjusted_base = base + cumulative_offset
#         group_size = len(group)
#         group_converted = [adjusted_base + i for i in range(group_size)]
#         converted_numbers.extend(group_converted)
#         cumulative_offset += (group_size - 1)  # Update offset for future residues
    
#     return converted_numbers

# # *********************************************


# def get_seqres2atmseq_mask(seqres, atmseq, pdbid):
#     try:
#         aln = run_align_clustalomega(
#             clustal_omega_executable=CLUSTAL_OMEGA_EXECUTABLE,
#             seq1=seqres,
#             seq2=atmseq,
#         )

#         # Check if seqres contains dash
#         if "-" in str(aln[0].seq):
#             raise ValueError("Error: seqres contains dash")

#         aln1 = str(aln[1].seq)  # atmseq in aln may contain "-"
#         seqres2atmseq = [
#             1 if i != "-" else 0 for i in aln1
#         ]  # 1 => in atmseq; 0 => not in atmseq

#         # Ensure the lengths match
#         if len(seqres2atmseq) != len(seqres):
#             raise ValueError("Error: Length mismatch between seqres2atmseq and seqres")

#         return seqres2atmseq
    
#     except Exception as e:
#         # Log the error with the PDB ID
#         logging.error(f"PDB ID {pdbid}: {e}")
#         return None  # Return None or an empty list to indicate failure

    

# def get_pdb_metadata(pdb_path, selected_chains):
#     """Extracts relevant metadata (HEADER, REMARK, SEQRES) for the selected chains."""
#     with open(pdb_path, "r") as f:
#         lines = f.readlines()
    
#     metadata = []
#     for line in lines:
#         if line.startswith(("HEADER", "TITLE", "REMARK", "MODRES")):
#             chain_id = line.split()[2] if len(line.split()) > 2 else None
#             if chain_id is None or chain_id in selected_chains:
#                 metadata.append(line)
#         elif line.startswith("SEQRES"):
#             chain_id = line.split()[2]
#             if chain_id in selected_chains:
#                 metadata.append(line)
    
#     return "".join(metadata)


# """
# TODO: 
# - re-index atmseq based on seqres2atmseq mask 
#     - get atmseq and seqres from the pdb file
#     - perform pairwise alignment between atmseq and seqres using clustal omega
#     - get the atmseq indices from seqres2atmseq mask
#     - create temporary mapping to outside the range of the old indices
#     - assign the new mapping to the residue number 
# """

# def reset_antibody_residue_indices(pdb_path, structure, antibody_chains, pt_graphs_dir, pdb_id):
#     """
#     Resets residue indices for antigen chains while preserving gaps if necessary.
#     """
#     atomic_df = PandasPdb().read_pdb(pdb_path).get_model(1).df["ATOM"]
#     atomic_df = atomic_df[atomic_df["chain_id"].isin(antibody_chains)]
    
#     mask_data = torch.load(pt_graphs_dir + f'{pdb_id}.pt')

#     # Get SEQRES and ATMSEQ
#     ag_chain = list(antigen_chains)[0]  # Get the first chain in the set
#     seqres = str(np.array(mask_data["seqres"]["ag"][ag_chain]))  # Convert to string
#     atmseq_df = atomic_df[["residue_number", "residue_name"]].drop_duplicates()
#     atmseq = "".join(atmseq_df["residue_name"].map(AA_MAP))

#     # Get old residue indices
#     old_indices = atomic_df["residue_number"].unique()
#     min_value = min(old_indices)
#     max_value = max(old_indices)

#     # Don't change indices if they are already in the correct range
#     """
#     NOTE: 
#     - comment this sanity check because the seqres2atmseq mask records needs to be saved eitherway 
#     """
#     # if np.array_equal(old_min_max, new_min_max):
#     #     print("No need to reset indices for PDB ID:", pdb_id)
#     #     return structure

#     # Get alignment mask between SEQRES and ATMSEQ
#     """
#     TODO: 
#     - keep log of the complexes that can't be aligned
#     - gaps in seqres or length of seqres and atmseq are not equal
#     """
#     seqres2atmseq = get_seqres2atmseq_mask(seqres, atmseq, pdb_id)

#     if seqres2atmseq is not None:
#         atmseq_indices = [index for index, bit in enumerate(seqres2atmseq) if bit == 1]

#         # Create temporary mapping to avoid conflicts
#         temp_mapping = {old: new for old, new in zip(old_indices, old_indices + max_value + 2)}
#         structure = transform_indices(structure, temp_mapping, antigen_chains)

#         # Create final mapping to align with SEQRES
#         temp_indices = np.array(list(temp_mapping.values()))
#         mapping = {old: new for old, new in zip(temp_indices, atmseq_indices)}

#         """
#         TODO: 
#         - save pdb_id, seqres, atmseq, seqres2atmseq, and atmseq_indices as npy and csv
#         """

#         # Apply final mapping to the structure
#         structure = transform_indices(structure, mapping, antigen_chains)
#         return structure, seqres, atmseq, seqres2atmseq, atmseq_indices
    
#     else:
#         print("Alignment failed. Check the log for details.")
#         return None



# def transform_indices(structure, mapping, antigen_chains):
#     """Transforms residue indices in a PDB structure using a mapping."""
#     for model in structure:
#         for chain in model:
#             if chain.get_id() in antigen_chains:
#                 for residue in chain:
#                     res_id = residue.get_id()[1]
#                     if res_id in mapping:
#                         residue.id = (residue.id[0], mapping[res_id], residue.id[2])
#     return structure


# def write_pdb_with_metadata(output_path, metadata, structure, select):
#     """Writes a PDB file with the extracted metadata."""
#     with open(output_path, "w") as f:
#         f.write(metadata)  # Write metadata (HEADER, REMARK, SEQRES)
#         io = PDBIO()
#         io.set_structure(structure)
#         io.save(f, select=select)  # Write ATOM records for selected chains



# def split_complex(pdb_path, pt_graphs_dir, pdb_id, ab_output, ag_output):
#     """Splits an antigen-antibody complex PDB file into separate antigen and antibody files while preserving relevant metadata."""
#     structure = PDBParser(QUIET=True).get_structure("complex", pdb_path)
    
#     # all_chains = {chain.get_id() for chain in structure.get_chains()}
#     ab_chains = {"H", "L"}
#     # ag_chains = all_chains - ab_chains

#     """
#     BUG:
#     - PMLs such as esm-if can't generate embeddings when pdb file has heteratoms
#     TODO: 
#     - need to remove heteratoms from the both antigen and antibody chains
#     - this changes the antigen and antibody pdbs, 
#     """
    
#     ab_metadata = get_pdb_metadata(pdb_path, ab_chains)
#     # ag_metadata = get_pdb_metadata(pdb_path, ag_chains)
    
#     # write_pdb_with_metadata(ab_output + f"{pdb_id}_ab.pdb", ab_metadata, structure, ChainSelect(ab_chains))
    
#     structure, seqres, atmseq, seqres2atmseq, atmseq_indices = reset_antibody_residue_indices(
#         pdb_path, structure, ab_chains, pt_graphs_dir, pdb_id)
#     if structure is not None:
#         write_pdb_with_metadata(ag_output + f"{pdb_id}_ab.pdb", ab_metadata, structure, ChainSelect(ab_chains))
#         return seqres, atmseq, seqres2atmseq, atmseq_indices


# def main():
#     parser = argparse.ArgumentParser(description="Split an antigen-antibody complex into separate PDB files.")
#     parser.add_argument("ag_ab_pdb_files", type=Path, help="Input PDB dir")
#     parser.add_argument("pt_graphs_files", help="Input Pytorch graphs dir")
#     parser.add_argument("ab_output", type=str, help="Output antibody PDB dir")
#     # parser.add_argument("ag_output", type=str, help="Output antigen PDB dir")
#     parser.add_argument("ab_seqres2atmseqmask_output", type=str, help="Output antibody seqres2atmseq mask dir")
#     args = parser.parse_args()

#     # Configure the logging
#     logging.basicConfig(filename='alignment_errors.log', level=logging.ERROR,
#                         format='%(asctime)s - %(levelname)s - %(message)s')
    
#     seqres2atmseq_mask_list = []
#     seqres2atmseq_mask_csv = []
    
#     all_structures = os.listdir(args.ag_ab_pdb_files)
#     for file in range(len(all_structures)):
#         pdb_id = all_structures[file].split(".")[0]
#         if not os.path.exists(args.ag_output + f"{pdb_id}_ab.pdb"): # and os.path.isfile(args.ag_output + f"{pdb_id}_ag.pdb"):
#             # print(pdb_id)
#             # seqres, atmseq, seqres2surf_mask, seqres2epitope_mask = get_atmseq_seqres(asep_ag_structures_dir, asep_graphs_dir, "3v6o_1P")
#             seqres, atmseq, seqres2atmseq, atmseq_indices = split_complex(os.path.join(args.ag_ab_pdb_files, 
#                     all_structures[file]), args.pt_graphs_files, pdb_id, args.ab_output, args.ag_output)
#             seqres2atmseq_mask_dict = {"pdb_id": pdb_id, "seqres": seqres, "atmseq": atmseq,
#                                        "seqres2atmseq": seqres2atmseq, "atmseq_indices": atmseq_indices}
#             seqres2atmseq_mask_list.append(seqres2atmseq_mask_dict)
#             seqres2atmseq_mask_csv.append(pd.Series(seqres2atmseq_mask_dict))
    
#     np.save(os.path.join(args.ag_seqres2atmseqmask_output, "seqres2atmseq_mask_list.npy"), seqres2atmseq_mask_list)
#     pd.DataFrame(seqres2atmseq_mask_csv).to_csv(os.path.join(args.ag_seqres2atmseqmask_output, "seqres2atmseq_mask_df.csv"))

#     print(f"Antibody saved to {args.ab_output}")
#     print(f"Antigen saved to {args.ag_output}")


# if __name__ == "__main__":
#     main()

# """
# python3 split_ag_ab_complex_pdb.py  \
#      \
#      \
#      \
#     
# """

# """
# python3 reindex_ab_split_complex.py  \
#      \
#      \
#     
# """


# """
# Problematic case:  
# Problematic case:  
# Problematic case:  
# Problematic case:  
# """


# #*******************************************************************************
