"""
TODO: 
1. re-index antibody atmseq while spliting ag-ab complex by aligning seqres and atmseq
2. create seqres2atmseq mask
    - how to apply seqres2cdr mask with heavy and light chains?
NOTE: 
    - be careful about heavy and light chains
    - align heavy chain and light chain seqres separately with atmseq
    - because heavy and light chains are different sequences; aligning them together might 
    lead to mismatches since their variable and constant regions aren't homologous. 
    Clustal Omega would try to align residues across the entire length, possibly creating gaps or 
    misalignments where the chains differ.

3. apply seqres2cdr mask on atmseq to get cdr structures and sequences
4. don't have seqres2paratope mask available in asep, so use atmseq2paratope to get labels
- create seqres2paratope using seqres2atmseq mask and cdr2paratope
- only then antibody seqres can be used when we have seqres2paratope
- cdr2paratope is already available in asep dataset `test_pre_cal.pt` as `y_b`
    - load `test_pre_cal.pt`  as `y_b`
    - chain order in cdr2paratope is forced to be H, L in `preprocess.py`
"""

"""
TODO: 
1. re-index antibody atmseq while spliting ag-ab complex by aligning seqres and atmseq
2. create seqres2atmseq mask for heavy and light chains separately
3. apply seqres2cdr mask on atmseq to get cdr structures and sequences
4. create seqres2paratope using seqres2atmseq mask and cdr2paratope
- cdr2paratope is available in asep dataset `dict_pre_cal.pt` as `y_b`

NOTE: 
- handle heavy and light chains separately
- seqres2cdr mask is concatenated [light_chain + heavy_chain]
- align heavy chain and light chain seqres separately with atmseq
- seqres2paratope needs to be created using seqres2atmseq and cdr2paratope
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio.PDB import PDBParser, PDBIO, Select
from biopandas.pdb import PandasPdb

warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'codebase')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))

# Dictionary for mapping three-letter codes to one-letter amino acid codes
AA_MAP = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G",
    "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
    "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V",
    "TRP": "W", "TYR": "Y"
}

class ChainResidueSelect(Select):
    """Select specific residues from a PDB structure based on residue numbers."""
    def __init__(self, residue_numbers, chain_id):
        self.residue_numbers = set(residue_numbers)
        self.chain_id = chain_id

    def accept_residue(self, residue):
        return (residue.get_id()[1] in self.residue_numbers and 
                residue.get_parent().get_id() == self.chain_id)

def load_pdb_and_masks(pdb_id, pdb_path, masks_pt_path, asep_graphs_processed):
    """Load the antibody PDB and corresponding CDR/paratope masks."""
    
    # Load the mask data from .pt file
    mask_data = torch.load(masks_pt_path + f'{pdb_id}.pt')

    # Read antibody PDB for both chains
    ab_pdb_df = PandasPdb().read_pdb(pdb_path + f'{pdb_id}_ab.pdb')
    ab_pdb_df = ab_pdb_df.get_model(1).df["ATOM"]
    
    # Extract masks and sequences
    seqres2cdr_mask = pd.Series(mask_data["mapping"]["ab"]["seqres2cdr"])

    """
    TODO: 
    - seqres2paratope is not available for antibodies, need to use cdr2paratope as y_b
    """
    cdr2paratope_mask = pd.Series(asep_graphs_processed[pdb_id]["y_b"])

    # Get sequences for both chains
    light_chain_seqres = mask_data["seqres"]["ab"]["L"]
    heavy_chain_seqres = mask_data["seqres"]["ab"]["H"]

    """
    TODO: 
    1. create and return seqres and atmseq
    2. create atmseq2cdr mask of size equal to atmseq, but how?? 
    """
    seqres = heavy_chain_seqres + light_chain_seqres

    seqres2cdr_seq = "".join(residue for residue, bit in zip(list(seqres), seqres2cdr_mask) if bit == 1)

    filtered_ab_temp_df = ab_pdb_df[ab_pdb_df["atom_name"] == "CA"]
    atmseq = "".join(filtered_ab_temp_df["residue_name"].map(AA_MAP))

    # heavy_chain_atmseq = "".join(filtered_ab_temp_df["residue_name"].map(AA_MAP))
    # atmseq = heavy_chain_atmseq + light_chain_atmseq

    # Split seqres2cdr mask into light and heavy chain masks
    light_chain_len = len(list(light_chain_seqres))
    heavy_chain_len = len(list(heavy_chain_seqres))

    """
    NOTE: 
    - assuming that seqres2cdr_mask contains masks ordered as heavy and light chains respectively
    - as per asepv1_dataset.py, seqres2cdr_mask is used with seqres ordered as H+L
    """
    
    seqres2cdr_mask_H = pd.Series(seqres2cdr_mask[:heavy_chain_len])
    seqres2cdr_mask_L = pd.Series(list(seqres2cdr_mask[heavy_chain_len:heavy_chain_len + light_chain_len]))

    return (ab_pdb_df, seqres2cdr_mask, seqres2cdr_mask_L, seqres2cdr_mask_H, cdr2paratope_mask,
            light_chain_seqres, heavy_chain_seqres, seqres2cdr_seq)




def seqres2cdr_mapping(pdb_path, ab_pdb_df, seqres2cdr_mask_L, seqres2cdr_mask_H, cdr_pdb_path, pdb_id):
    """Filter and save CDR residues of antibody PDB for both chains combined using BioPandas."""
    
    """
    FIXME: 
    - first filter the corresponding chain and then apply cdr residues per chain mask
    - use biopandas to load ab pdb file and save the filtered cdr pdb
    """
    
    # Create a new PandasPDB object for CDR residues
    cdr_ppdb = PandasPdb()
    
    # Process Light chain
    cdr_pdb_df_L = ab_pdb_df[ab_pdb_df["chain_id"] == 'L'].copy()
    if not cdr_pdb_df_L.empty:
        # Get CDR residues for L chain
        cdr_residues_L = cdr_pdb_df_L[
            cdr_pdb_df_L["residue_number"].map(seqres2cdr_mask_L) == 1
        ]["residue_number"].unique()
        
        # Filter L chain atoms for CDR residues
        cdr_pdb_df_L = cdr_pdb_df_L[
            cdr_pdb_df_L["residue_number"].isin(cdr_residues_L)
        ]
    
    # Process Heavy chain
    cdr_pdb_df_H = ab_pdb_df[ab_pdb_df["chain_id"] == 'H'].copy()
    if not cdr_pdb_df_H.empty:
        # Get CDR residues for H chain
        cdr_residues_H = cdr_pdb_df_H[
            cdr_pdb_df_H["residue_number"].map(seqres2cdr_mask_H) == 1
        ]["residue_number"].unique()
        
        # Filter H chain atoms for CDR residues
        cdr_pdb_df_H = cdr_pdb_df_H[
            cdr_pdb_df_H["residue_number"].isin(cdr_residues_H)
        ]
    
    # Combine both chains with H chain first (to maintain H-L order)
    combined_cdr_df = pd.concat([cdr_pdb_df_H, cdr_pdb_df_L])
    
    # Update the new PandasPDB object
    cdr_ppdb.df["ATOM"] = combined_cdr_df
    
    # Save to PDB file
    output_pdb_file = os.path.join(cdr_pdb_path, f"{pdb_id}_cdr.pdb")
    cdr_ppdb.to_pdb(
        path=output_pdb_file,
        records=["ATOM" ], #, "ANISOU"],  # Include ANISOU records if present
        gz=False,
        append_newline=True
    )
    
    # return cdr_ppdb


# def seqres2cdr_mapping(pdb_path, ab_pdb_df, seqres2cdr_mask_L, seqres2cdr_mask_H, cdr_pdb_path, pdb_id):
#     """Filter and save CDR residues of antibody PDB for both chains combined."""

#     """
#     FIXME: 
#     - first filter the corresponding chain and then apply cdr residues per chain mask
#     - use biopandas to load ab pdb file and save the filtered cdr pdb
#     """
    
#     # Get residue numbers (indices) that are CDR (mask = 1) for both chains
#     cdr_pdb_df_L = ab_pdb_df[ ab_pdb_df["chain_id"] == 'L']
#     cdr_residues_L = cdr_pdb_df_L[ cdr_pdb_df_L["residue_number"].map(seqres2cdr_mask_L) == 1]["residue_number"].unique()

#     cdr_pdb_df_H = ab_pdb_df[ ab_pdb_df["chain_id"] == 'H']
#     # print(cdr_pdb_df_L, cdr_pdb_df_H)
#     cdr_residues_H = cdr_pdb_df_H[ cdr_pdb_df_H["residue_number"].map(seqres2cdr_mask_H) == 1]["residue_number"].unique()
    

#     class CombinedChainSelect(Select):
#         def __init__(self, residues_L, residues_H):
#             self.residues_L = set(residues_L)
#             self.residues_H = set(residues_H)

#         def accept_residue(self, residue):
#             chain_id = residue.get_parent().get_id()
#             return (chain_id == 'L' and residue.get_id()[1] in self.residues_L) or \
#                    (chain_id == 'H' and residue.get_id()[1] in self.residues_H)

#     # Parse full PDB structure and save combined CDR file
#     parser = PDBParser(QUIET=True)
#     structure = parser.get_structure(pdb_id, pdb_path + f'{pdb_id}_ab.pdb')
#     io = PDBIO()
#     io.set_structure(structure)
#     output_pdb_file = os.path.join(cdr_pdb_path, f"{pdb_id}_cdr.pdb")
#     # io.save(output_pdb_file, select=CombinedChainSelect(cdr_residues_L, cdr_residues_H))

#     """
#     NOTE: force chain order of H-L
#     """
#     io.save(output_pdb_file, select=CombinedChainSelect(cdr_residues_H, cdr_residues_L))

#     # Create combined mask H+L
#     """
#     SANITY CHECK: 
#     - seqres2cdr_mask is alreay available for asep dataset, 
#         why create seqres2atmseq2cdr_mask again? basically the same
#     """


def sequence_filtering(ab_pdb_df, seqres2cdr_mask_L, seqres2cdr_mask_H):
    """Filter antibody pdb dataframe and get CDR sequences for both chains."""
    # Filter Light Chain
    filtered_ab_df_L = ab_pdb_df[
        (ab_pdb_df["chain_id"] == 'L') & 
        (ab_pdb_df["residue_number"].map(seqres2cdr_mask_L) == 1)
    ]
    # Filter Heavy Chain
    filtered_ab_df_H = ab_pdb_df[
        (ab_pdb_df["chain_id"] == 'H') & 
        (ab_pdb_df["residue_number"].map(seqres2cdr_mask_H) == 1)
    ]
    
    # Combine filtered dataframes containing cdr residues only
    # enforce H+L chain order for the filtered ab dataframe to do cdr and paratope masking later on
    filtered_ab_df = pd.concat([filtered_ab_df_H, filtered_ab_df_L])
    
    # Get unique residues and convert to sequence
    filtered_residues_L = filtered_ab_df_L[["residue_number", "residue_name"]].drop_duplicates()
    filtered_residues_H = filtered_ab_df_H[["residue_number", "residue_name"]].drop_duplicates()
    
    atmseq_L = "".join(filtered_residues_L["residue_name"].map(AA_MAP))
    atmseq_H = "".join(filtered_residues_H["residue_name"].map(AA_MAP))
    
    return filtered_ab_df, atmseq_L, atmseq_H


def atmseq2paratope_mapping(filtered_ab_df, cdr2paratope_mask, seqres2cdr_mask):
    """
    Generate combined paratope mapping files for both chains.
    Returns:
    - seqres2paratope_labels
    - binary_paratope_labels
    - atmseq2paratope_labels 
    - seqres_binary_paratope_labels 
    - atmseq2cdr_seq
    """
    # Filter for CA atoms
    filtered_ab_cdr_df = filtered_ab_df[filtered_ab_df["atom_name"] == "CA"]

    atmseq2cdr_seq = "".join(filtered_ab_cdr_df["residue_name"].map(AA_MAP))

    """
    NOTE: 
    - assuming `cdr2paratope_mask` has the mask with cdr residues in the order H+L 
    - chain order in cdr2paratope (y_b) is forced to be H, L in `preprocess.py`
    """
    filtered_paratope_df = filtered_ab_cdr_df[filtered_ab_cdr_df["residue_number"].map(cdr2paratope_mask) == 1]
    
    seqres2paratope_labels = []
    binary_paratope_labels = []
    atmseq2paratope_labels = []
    seqres_temp_indices = np.arange(len(seqres2cdr_mask))

    seqres2paratope_labels = [f"{res}_{resname}_{chain_id}" for res, resname, chain_id in zip(filtered_paratope_df["residue_number"], filtered_paratope_df["residue_name"].map(AA_MAP), filtered_paratope_df["chain_id"])]
    
    """
    TODO: 
    - save atmseq2epitope labels with one letter amino acid codes as well
    - create temp seqres index and fill in the mask seqres_binary_paratope_labels
    - save paratope labels as {res_index}_{resname}_{chain_id}
    """
    atmseq2paratope_labels = [f"{res}_{resname}_{chain_id}" for res, resname, chain_id in zip(filtered_paratope_df["residue_number"], filtered_paratope_df["residue_name"].map(AA_MAP), filtered_paratope_df["chain_id"])]
    binary_paratope_labels = np.array([1 if res in filtered_paratope_df["residue_number"].values else 0 for res in filtered_ab_cdr_df["residue_number"]])

    seqres_binary_paratope_labels = np.array([1 if res in filtered_paratope_df["residue_number"].values else 0 for res in seqres_temp_indices])
            
    return seqres2paratope_labels, binary_paratope_labels, atmseq2paratope_labels, seqres_binary_paratope_labels, atmseq2cdr_seq





def main():
    parser = argparse.ArgumentParser(description="Process antibody CDR mapping.")
    parser.add_argument("ab_pdb_dir", type=str, help="Input antibody PDB dir")
    parser.add_argument("masks_graph_pt_dir", type=str, help="Graph masks dir")
    parser.add_argument("processed_graphs_dir", type=str, help="Processed graphs dir")
    parser.add_argument("ab_cdr_pdb_out_dir", type=str, help="Output CDR PDB dir")
    parser.add_argument("ab_sequences_out_dir", type=str, help="Output antibody seqres & CDR dir")
    args = parser.parse_args()

    # Load processed graphs containing paratope info
    asep_graphs_processed = torch.load(os.path.join(args.processed_graphs_dir, 'dict_pre_cal.pt'))
    
    os.makedirs(args.ab_sequences_out_dir, exist_ok=True)
    os.makedirs(args.ab_cdr_pdb_out_dir, exist_ok=True)
    
    # Initialize lists to store combined data
    combined_data = {
        'seqres': [],
        'atmseq': [],
        'atmseq2paratope': [],
        'seqres2paratope': [],
        'cdr2paratope_mask': [], 
        "seqres2paratope_mask": [],
        "atmseq2cdr_seqres2cdr_seq" :  [],
        "atmseq2paratope_seqres2paratope_labels" :  []
    }

    all_mask_files = os.listdir(args.masks_graph_pt_dir)
    all_mask_files.remove("5nj6_0P.pt")  # Remove problematic file
    
    for mask_file in all_mask_files:
        pdb_id = mask_file.split(".")[0]

        """
        TODO:
        - add two test cases to compare:
        (1) seqres2cdr and atmseq2cdr 
        (2) seqres2paratope and atmseq2paratope
        - mask cdr & paratope from seqres and atmseq
        """
        
        # Load PDB and masks
        (ab_pdb_df, seqres2cdr_mask, seqres2cdr_mask_L, seqres2cdr_mask_H, cdr2paratope_mask,
         light_seqres, heavy_seqres, seqres2cdr_seq) = load_pdb_and_masks(
            pdb_id, args.ab_pdb_dir, args.masks_graph_pt_dir, asep_graphs_processed)

        # Generate CDR mapping and save combined PDB
        seqres2cdr_mapping(
            args.ab_pdb_dir, ab_pdb_df, seqres2cdr_mask_L, seqres2cdr_mask_H, 
            args.ab_cdr_pdb_out_dir, pdb_id)
        
        # Filter sequences and get CDR regions
        filtered_ab_df, atmseq_L, atmseq_H = sequence_filtering(
            ab_pdb_df, seqres2cdr_mask_L, seqres2cdr_mask_H)
        
        # Generate paratope mappings
        seqres2paratope_labels, binary_labels, atmseq2paratope_labels, seqres_binary_paratope_labels, atmseq2cdr_seq = atmseq2paratope_mapping(
            filtered_ab_df, cdr2paratope_mask, seqres2cdr_mask)

        # Store combined data
        combined_data['seqres'].append(f">{pdb_id}\n{heavy_seqres}{light_seqres}\n") # seqres ordered as H, L
        combined_data['atmseq'].append(f">{pdb_id}\n{atmseq_H}{atmseq_L}\n")
        combined_data['atmseq2paratope'].append([pdb_id, atmseq2paratope_labels])
        combined_data['seqres2paratope'].append([pdb_id, seqres2paratope_labels, 
                                np.array(seqres_binary_paratope_labels, dtype=object), seqres2cdr_mask])
        combined_data['cdr2paratope_mask'].append([pdb_id, seqres2paratope_labels, binary_labels ])
        combined_data['seqres2paratope_mask'].append([pdb_id, seqres_binary_paratope_labels])

        combined_data['atmseq2cdr_seqres2cdr_seq'].append([pdb_id, seqres2cdr_seq,
                            atmseq2cdr_seq, len(seqres2cdr_seq), len(atmseq2cdr_seq),
        len(seqres2cdr_seq)== len(atmseq2cdr_seq), seqres2cdr_seq== atmseq2cdr_seq])

        combined_data['atmseq2paratope_seqres2paratope_labels'].append([pdb_id, seqres2paratope_labels,
                            atmseq2paratope_labels, len(seqres2paratope_labels), len(atmseq2paratope_labels),
        len(seqres2paratope_labels)== len(atmseq2paratope_labels), seqres2paratope_labels== atmseq2paratope_labels])


    # Save all files
    save_output_files(args, combined_data)

def save_output_files(args, combined_data):
    """Save all processed files with combined chain data."""
    output_dir = Path(args.ab_sequences_out_dir)
    
    # Save FASTA files
    with open(output_dir / "ab_seqres.fasta", "w") as f:
        f.writelines(combined_data['seqres'])
    with open(output_dir / "ab_cdr_atmseq.fasta", "w") as f:
        f.writelines(combined_data['atmseq'])

    # Save paratope data
    np.save(output_dir / "cdr2paratope_mask.npy",
            np.array(combined_data['cdr2paratope_mask'], dtype=object))
    
    np.save(output_dir / "seqres2paratope_mask.npy",
        np.array(combined_data['seqres2paratope_mask'], dtype=object))
    
    pd.DataFrame(combined_data['atmseq2paratope'],
                columns=["pdbid", "paratope"]).to_csv(
                    output_dir / "atmseq2paratope_residues.csv", index=False)
    
    pd.DataFrame(combined_data['seqres2paratope'],
                columns=["pdbid", "paratope", "seqres2paratope_mask", "seqres2cdr_mask"]).to_csv(
                    output_dir / "seqres2paratope_residues.csv", index=False)

    paratope_df = pd.DataFrame(combined_data['atmseq2cdr_seqres2cdr_seq'], 
                        columns=["pdbid", "seqres2cdr_seq", "atmseq2cdr_seq", "len_seqres2cdr_seq", 
                                 "len_atmseq2cdr_seq", "equal_len", "same_seq"])
    paratope_df.to_csv(output_dir / "atmseq2cdr_seqres2cdr_seq.csv", index=False)

    paratope_df = pd.DataFrame(combined_data['atmseq2paratope_seqres2paratope_labels'], 
                        columns=["pdbid", "seqres2paratope", "atmseq2paratope", "len_seqres2paratope", 
                                 "len_atmseq2paratope", "equal_len", "same_seq"])
    paratope_df.to_csv(output_dir / "atmseq2paratope_seqres2paratope_labels.csv", index=False)

    
    np.save(output_dir / "seqres2paratope_residues",
            np.array(combined_data['seqres2paratope'], dtype=object))
        
    print(f"\nProcessed files saved to: {args.ab_sequences_out_dir}")
    print(f"CDR PDB files saved to: {args.ab_cdr_pdb_out_dir}")
    print("\nFiles saved:")
    print("- ab_seqres.fasta")
    print("- ab_cdr_atmseq.fasta") 
    print("- cdr2paratope_mask.npy") # binary labels of length equal to cdr
    print("- seqres2paratope_mask.npy") # binary labels of length equal to seqres
    print("- atmseq2paratope_residues.csv")
    print("- seqres2paratope_residues.csv")
    print("- atmseq2cdr_seqres2cdr_seq.csv")
    print("- atmseq2paratope_seqres2paratope_labels.csv")

if __name__ == "__main__":
    main()



"""
python3 seqres2cdr_mapping.py  \
 \
 \
 \

"""


"""
python3 seqres2cdr_mapping.py  \
 \
 \
 \

"""


