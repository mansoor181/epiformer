# **********************************************************************************
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

# Dictionary for mapping three-letter codes to one-letter amino acid codes
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

    


"""
TODO: 
- re-index atmseq based on seqres2atmseq mask 
    - get atmseq and seqres from the pdb file
    - perform pairwise alignment between atmseq and seqres using clustal omega
    - get the atmseq indices from seqres2atmseq mask
    - create temporary mapping to outside the range of the old indices
    - assign the new mapping to the residue number 
"""


def split_complex_reindex_antigen_chains(pdb_path, pt_graphs_dir, pdb_id, ag_out_dir):
    
    ppdb = PandasPdb().read_pdb(pdb_path)
    atomic_df = ppdb.get_model(1).df["ATOM"]

    mask_data = torch.load(f"{pt_graphs_dir}/{pdb_id}.pt")
    output_path = os.path.join(ag_out_dir, f"{pdb_id}_ag.pdb")

    chains = atomic_df["chain_id"].unique()

    ag_chain = chains[2]

    # Process antigen chain
    chain_data = {}

    # Create a copy of the original DataFrame for antigen chains only
    ab_df = ppdb.df["ATOM"][ppdb.df["ATOM"]["chain_id"].isin(list(ag_chain))].copy()

    chain_df = ab_df[ab_df["chain_id"] == ag_chain]

    # Get SEQRES and ATMSEQ for the chain
    seqres = str(np.array(mask_data["seqres"]["ag"][ag_chain]))

    atmseq_df = atomic_df[atomic_df["chain_id"] == ag_chain]  # NEW LINE
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

    chain_data["Ag"] = (seqres, atmseq, mask)


    # Save only antigen chains with new numbering
    ppdb_ab = PandasPdb()
    ppdb_ab.df["ATOM"] = ab_df
    ppdb_ab.to_pdb(path=output_path, 
                  records=["ATOM"],
                  gz=False,
                  append_newline=True)


    return chain_data


def main():
    parser = argparse.ArgumentParser(description="Split an antigen-antibody complex into separate PDB files.")
    parser.add_argument("input_dir", type=Path, help="Input PDB directory")
    parser.add_argument("pt_graphs_dir", type=Path, help="PyTorch graphs directory")
    parser.add_argument("output_dir", type=Path, help="Output directory for processed antigen PDBs")
    parser.add_argument("metadata_dir", type=Path, help="Output directory for alignment metadata")
    args = parser.parse_args()

    # Configure the logging
    logging.basicConfig(filename='alignment_errors.log', level=logging.ERROR,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    

    metadata_list = []
    for pdb_file in args.input_dir.glob("*.pdb"):
        # print(pdb_file)
        pdb_id = pdb_file.stem.split(".")[0]

        chain_data = split_complex_reindex_antigen_chains(str(pdb_file), args.pt_graphs_dir,
                                 pdb_id, args.output_dir)

        if chain_data:
            metadata_entry = {
                "pdb_id": pdb_id,
                "seqres": chain_data.get("Ag", (None, None, None))[0],
                "atmseq": chain_data.get("Ag", (None, None, None))[1],
                "seqres2atmseq_mask": chain_data.get("Ag", (None, None, None))[2]
            }
            metadata_list.append(metadata_entry)

    # Save metadata
    pd.DataFrame(metadata_list).to_csv(args.metadata_dir/"seqres2atmseq_mask_ag.csv", index=False)
    print(f"Processed {len(metadata_list)} antigen structures")



if __name__ == "__main__":
    main()




"""
python3 reindex_ag_split_complex.py  \
     \
     \
    
"""

"""
python3 reindex_ag_split_complex.py  \
     \
     \
    
"""


"""
Problematic case:  
Problematic case:  
Problematic case:  
Problematic case:  

No need to reset indices for PDB ID: 7bpk_0P
No need to reset indices for PDB ID: 7ue9_0P
No need to reset indices for PDB ID: 4uu9_0P
No need to reset indices for PDB ID: 5d8j_0P
No need to reset indices for PDB ID: 5hbv_0P
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
#     atomic_df = PandasPdb().read_pdb(pdb_structures_dir + f"{pdb_id}_ag.pdb").get_model(1).df["ATOM"]
#     mask_data = torch.load(pt_graphs_dir + f'{pdb_id}.pt')

#     ag_chain = atomic_df["chain_id"].unique()[0]
#     seqres = np.array(mask_data["seqres"]["ag"][ag_chain])

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

# def reset_antigen_residue_indices(pdb_path, structure, antigen_chains, pt_graphs_dir, pdb_id):
#     """
#     Resets residue indices for antigen chains while preserving gaps if necessary.
#     """
#     atomic_df = PandasPdb().read_pdb(pdb_path).get_model(1).df["ATOM"]
#     atomic_df = atomic_df[atomic_df["chain_id"].isin(antigen_chains)]
    
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
#     old_min_max = np.array([min_value.item(), max_value.item()])
#     new_min_max = np.array([0, len(seqres) - 1])

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
    
#     all_chains = {chain.get_id() for chain in structure.get_chains()}
#     ab_chains = {"H", "L"}
#     ag_chains = all_chains - ab_chains

#     """
#     BUG:
#     - PLMs such as esm-if can't generate embeddings when pdb file has heteratoms
#     TODO: 
#     - need to remove heteratoms from the both antigen and antibody chains
#     - this changes the antigen and antibody pdbs, 
#     """
    
#     ab_metadata = get_pdb_metadata(pdb_path, ab_chains)
#     ag_metadata = get_pdb_metadata(pdb_path, ag_chains)
    
#     write_pdb_with_metadata(ab_output + f"{pdb_id}_ab.pdb", ab_metadata, structure, ChainSelect(ab_chains))
    
#     structure, seqres, atmseq, seqres2atmseq, atmseq_indices = reset_antigen_residue_indices(pdb_path, structure, ag_chains, pt_graphs_dir, pdb_id)
#     if structure is not None:
#         write_pdb_with_metadata(ag_output + f"{pdb_id}_ag.pdb", ag_metadata, structure, ChainSelect(ag_chains))
#         return seqres, atmseq, seqres2atmseq, atmseq_indices


# def main():
#     parser = argparse.ArgumentParser(description="Split an antigen-antibody complex into separate PDB files.")
#     parser.add_argument("ag_ab_pdb_files", type=Path, help="Input PDB dir")
#     parser.add_argument("pt_graphs_files", help="Input Pytorch graphs dir")
#     parser.add_argument("ab_output", type=str, help="Output antibody PDB dir")
#     parser.add_argument("ag_output", type=str, help="Output antigen PDB dir")
#     parser.add_argument("ag_seqres2atmseqmask_output", type=str, help="Output antigen seqres2atmseq mask dir")
#     args = parser.parse_args()

#     # Configure the logging
#     logging.basicConfig(filename='alignment_errors.log', level=logging.ERROR,
#                         format='%(asctime)s - %(levelname)s - %(message)s')
    
#     seqres2atmseq_mask_list = []
#     seqres2atmseq_mask_csv = []
    
#     all_structures = os.listdir(args.ag_ab_pdb_files)
#     for file in range(len(all_structures)):
#         pdb_id = all_structures[file].split(".")[0]
#         if not os.path.exists(args.ag_output + f"{pdb_id}_ag.pdb"): # and os.path.isfile(args.ag_output + f"{pdb_id}_ag.pdb"):
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





#****************************** some old script *************************************************






# import os
# import argparse
# import pandas as pd
# import numpy as np
# from pathlib import Path
# from biopandas.pdb import PandasPdb
# from Bio.PDB import PDBIO, PDBParser, Select
# import warnings
# warnings.filterwarnings("ignore")


# class ChainSelect(Select):
#     """Select specific chains from a PDB structure."""
#     def __init__(self, chains):
#         self.chains = set(chains)
    
#     def accept_chain(self, chain):
#         return chain.get_id() in self.chains

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

# def write_pdb_with_metadata(output_path, metadata, structure, select):
#     """Writes a PDB file with the extracted metadata."""
#     with open(output_path, "w") as f:
#         f.write(metadata)
#         io = PDBIO()
#         io.set_structure(structure)
#         io.save(f, select=select)

# """
# TODO: 
# - re-index atmseq based on seqres2atmseq mask 
#     - get atmseq and seqres from the pdb file
#     - perform pairwise alignment between atmseq and seqres using clustal omega
#     - get the atmseq indices from seqres2atmseq mask
#     - create temporary mapping to outside the range of the old indices
#     - assign the new mapping to the residue number 
# """

# def reset_antigen_residue_indices(pdb_path, structure, antigen_chains):
#     """Resets residue indices for antigen chains while preserving gaps if necessary."""
#     atomic_df = PandasPdb().read_pdb(pdb_path).get_model(1).df["ATOM"]
#     atomic_df = atomic_df[atomic_df["chain_id"].isin(antigen_chains)]
    
#     indices = atomic_df["residue_number"].unique()
#     min_value = min(indices)
#     max_value = max(indices)
#     old_min_max = np.array([min_value.item(), max_value.item()])
#     new_min_max = np.array([0, len(indices)-1])
#     # new_min_max = np.array([1, len(indices)])

#     # don't change the indices if the min and max indices are the same
#     """
#     TODO: 
#     - add another condition checking if the old indices are in the range (1, n)
#     """
#     if np.array_equal(old_min_max, new_min_max): # or np.array_equal(old_min_max-1, new_min_max+1):
#         return structure
    
#     if np.array_equal(old_min_max+1, new_min_max) or min(old_min_max) <= max(new_min_max):
    
#         # create the temporary mapping from old to new indices to avoid conflicts with existing indices
#         temp_mapping = {old: new for old, new in zip(indices, indices + max_value + 2)}
#         structure = transform_indices(structure, temp_mapping, antigen_chains)

#         """
#         NOTE: 
#         - something looks wrong with the indices transformation
#         - why is the new mapping a range from 1 to len(indices) + 1?
#             - because for 5lxg_0P.pdb, old indices are: -1, 0, 89, 90, .., 389
#             - the new indices should be: 1, 2, 3, 4, ..., 300
#             - for other 3 cases, 4ypg_0P, 5a3i_1P, and 6ye3_1P, there're no gaps in the indices
#         """
#         if min_value+2 == 1: # check problematic cases where the min index is -1
#             indices = np.array(list(temp_mapping.values()))
#             expected_range = np.arange(1, len(indices) + 1)
#             mapping = {old: new for old, new in zip(indices, expected_range)}
#             print("Problematic case: ", pdb_path )
           
#             return transform_indices(structure, mapping, antigen_chains)

#         indices = np.array(list(temp_mapping.values()))
#         """
#         TODO: 
#         - a bug fix:
#             - change the new indices to start from 0 instead of 1
#             - otherwise the seqres2surf and seqres2epitope will not match the indices correctly
#             - we get different surface and epitope residues for the ATOMSEQRES and SEQRES
#         """
#         mapping = {old: new for old, new in zip(indices, indices - min(indices))}
#         # mapping = {old: new for old, new in zip(indices, indices - min(indices) + 1)}

#         return transform_indices(structure, mapping, antigen_chains)

#     # otherwise transform the indices if the max index in original indices is greater than the min index in the expected range
#     mapping = {old: new for old, new in zip(indices, indices - min_value )}
#     # mapping = {old: new for old, new in zip(indices, indices - min(indices) + 1)}
    
#     return transform_indices(structure, mapping, antigen_chains)


# # def reset_antigen_residue_indices(pdb_path, structure, antigen_chains):
# #     """Resets residue indices for antigen chains while preserving gaps if necessary."""
# #     atomic_df = PandasPdb().read_pdb(pdb_path).get_model(1).df["ATOM"]
# #     atomic_df = atomic_df[atomic_df["chain_id"].isin(antigen_chains)]
    
# #     indices = atomic_df["residue_number"].unique()
# #     min_value = min(indices)
# #     max_value = max(indices)
# #     old_min_max = np.array([min_value.item(), max_value.item()])
# #     new_min_max = np.array([0, len(indices)-1])
# #     # new_min_max = np.array([1, len(indices)])

# #     # don't change the indices if the min and max indices are the same
# #     """
# #     TODO: 
# #     - add another condition checking if the old indices are in the range (1, n)
# #     """
# #     if np.array_equal(old_min_max, new_min_max): # or np.array_equal(old_min_max-1, new_min_max+1):
# #         return structure
    
# #     if np.array_equal(old_min_max+1, new_min_max) or min(old_min_max) <= max(new_min_max):
    
# #         # create the temporary mapping from old to new indices to avoid conflicts with existing indices
# #         temp_mapping = {old: new for old, new in zip(indices, indices + max_value + 2)}
# #         structure = transform_indices(structure, temp_mapping, antigen_chains)

# #         """
# #         NOTE: 
# #         - something looks wrong with the indices transformation
# #         - why is the new mapping a range from 1 to len(indices) + 1?
# #             - because for 5lxg_0P.pdb, old indices are: -1, 0, 89, 90, .., 389
# #             - the new indices should be: 1, 2, 3, 4, ..., 300
# #             - for other 3 cases, 4ypg_0P, 5a3i_1P, and 6ye3_1P, there're no gaps in the indices
# #         """
# #         if min_value+2 == 1: # check problematic cases where the min index is -1
# #             indices = np.array(list(temp_mapping.values()))
# #             expected_range = np.arange(1, len(indices) + 1)
# #             mapping = {old: new for old, new in zip(indices, expected_range)}
# #             print("Problematic case: ", pdb_path )
           
# #             return transform_indices(structure, mapping, antigen_chains)

# #         indices = np.array(list(temp_mapping.values()))
# #         """
# #         TODO: 
# #         - a bug fix:
# #             - change the new indices to start from 0 instead of 1
# #             - otherwise the seqres2surf and seqres2epitope will not match the indices correctly
# #             - we get different surface and epitope residues for the ATOMSEQRES and SEQRES
# #         """
# #         mapping = {old: new for old, new in zip(indices, indices - min(indices))}
# #         # mapping = {old: new for old, new in zip(indices, indices - min(indices) + 1)}

# #         return transform_indices(structure, mapping, antigen_chains)

# #     # otherwise transform the indices if the max index in original indices is greater than the min index in the expected range
# #     mapping = {old: new for old, new in zip(indices, indices - min_value )}
# #     # mapping = {old: new for old, new in zip(indices, indices - min(indices) + 1)}
    
# #     return transform_indices(structure, mapping, antigen_chains)



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


# def split_complex(pdb_path, pdb_id, ab_output, ag_output):
#     """Splits an antigen-antibody complex PDB file into separate antigen and antibody files while preserving relevant metadata."""
#     structure = PDBParser(QUIET=True).get_structure("complex", pdb_path)
    
#     all_chains = {chain.get_id() for chain in structure.get_chains()}
#     ab_chains = {"H", "L"}
#     ag_chains = all_chains - ab_chains
    
#     ab_metadata = get_pdb_metadata(pdb_path, ab_chains)
#     ag_metadata = get_pdb_metadata(pdb_path, ag_chains)
    
#     write_pdb_with_metadata(ab_output + f"{pdb_id}_ab.pdb", ab_metadata, structure, ChainSelect(ab_chains))
    
#     structure = reset_antigen_residue_indices(pdb_path, structure, ag_chains)
#     write_pdb_with_metadata(ag_output + f"{pdb_id}_ag.pdb", ag_metadata, structure, ChainSelect(ag_chains))



# def main():
#     parser = argparse.ArgumentParser(description="Split an antigen-antibody complex into separate PDB files.")
#     parser.add_argument("ag_ab_pdb_files", type=Path, help="Input PDB dir")
#     parser.add_argument("ab_output", type=str, help="Output antibody PDB dir")
#     parser.add_argument("ag_output", type=str, help="Output antigen PDB dir")
#     args = parser.parse_args()
    
#     all_structures = os.listdir(args.ag_ab_pdb_files)
#     for file in range(len(all_structures)):
#         pdb_id = all_structures[file].split(".")[0]
#         if not os.path.exists(args.ag_output + f"{pdb_id}_ag.pdb"): # and os.path.isfile(args.ag_output + f"{pdb_id}_ag.pdb"):
#             # print(pdb_id)
#             split_complex(os.path.join(args.ag_ab_pdb_files, all_structures[file]), 
#                         pdb_id, args.ab_output, args.ag_output)


#     print(f"Antibody saved to {args.ab_output}")
#     print(f"Antigen saved to {args.ag_output}")


# if __name__ == "__main__":
#     main()


# """
# python3 reindex_ag_split_complex.py  \
#      \
#     
# """


# """
# Problematic case:  
# Problematic case:  
# Problematic case:  
# Problematic case:  
# """