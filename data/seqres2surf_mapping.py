import os
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio.PDB import PDBParser, PDBIO, Select
from biopandas.pdb import PandasPdb

warnings.filterwarnings("ignore")

# Dictionary for mapping three-letter codes to one-letter amino acid codes
AA_MAP = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G",
    "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
    "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V",
    "TRP": "W", "TYR": "Y"
}

class ChainResidueSelect(Select):
    """Select specific residues from a PDB structure based on residue numbers."""
    def __init__(self, residue_numbers):
        self.residue_numbers = set(residue_numbers)

    def accept_residue(self, residue):
        return residue.get_id()[1] in self.residue_numbers

def load_pdb_and_masks(pdb_id, pdb_path, masks_pt_path):
    """Load the antigen PDB and corresponding surface/epitope masks."""
    
    # Load the mask data from .pt file
    mask_data = torch.load(masks_pt_path + f'{pdb_id}.pt')

    # Read antigen PDB
    ag_pdb_df = PandasPdb().read_pdb(pdb_path + f'{pdb_id}_ag.pdb')
    ag_pdb_df = ag_pdb_df.get_model(1).df["ATOM"]
    
    # Extract masks
    seqres2surf_mask = pd.Series(mask_data["mapping"]["ag"]["seqres2surf"])
    seqres2epitope_mask = pd.Series(mask_data["mapping"]["ag"]["seqres2epitope"])
    """
    TODO: 
    - return SEQRES and seqres2epitope residue labels as well
    """
    ag_chain = ag_pdb_df["chain_id"].unique()[0]
    ag_seqres = mask_data["seqres"]["ag"][ag_chain]

    # print(ag_seqres.tolist(), seqres2surf_mask)

    seqres2surf_seq = "".join(residue for residue, bit in zip(list(ag_seqres), seqres2surf_mask) if bit == 1)

    ag_seqres = np.array(mask_data["seqres"]["ag"][ag_chain])

    ag_seqres2epi_indices = [epi_index for epi_index, bit in enumerate(seqres2epitope_mask) if bit == 1]
    ag_seqres2epi_labels = list("".join(residue for residue, bit in zip(list(ag_seqres.item()), seqres2epitope_mask) if bit == 1))
    ag_seqres2epi_labels = [f"{res}_{resname}" for res, resname in zip(ag_seqres2epi_indices, ag_seqres2epi_labels)]

    return ag_pdb_df, seqres2surf_mask, seqres2epitope_mask, ag_seqres, ag_seqres2epi_labels, seqres2surf_seq

def seqres2surf_mapping(pdb_path, ag_pdb_df, seqres2surf_mask, ag_surf_pdb_path, pdb_id):
    """Filter and save only surface residues of antigen PDB."""
    # Get residue numbers (indices) that are surface (mask = 1) 
    surface_residues = ag_pdb_df[ag_pdb_df["residue_number"].map(seqres2surf_mask) == 1]["residue_number"].unique()

    """
    TODO: 
    - create and save seqres2atmseq2surf_mask (equal to length of seqres) based on surface residues
        - assign 1 to surface residues while 0 to others 
    - this mask will be used to map seqres to surface atmseq 
    - essentially construct a graph using atmseq2surf, generate PLM embeddings for seqres
        - then apply seqres2atmseq2surf mask on seqres
    """

    seqres2atmseq2surf_mask = np.zeros(len(seqres2surf_mask), dtype=int)
    for index in surface_residues:
        if 0 <= index < len(seqres2surf_mask):
            seqres2atmseq2surf_mask[index] = 1

    # Parse full PDB structure
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, pdb_path + f'{pdb_id}_ag.pdb')

    # Save filtered PDB with only surface residues
    io = PDBIO()
    io.set_structure(structure)
    output_pdb_file =  os.path.join(ag_surf_pdb_path, f"{pdb_id}_surf.pdb")
    io.save(output_pdb_file, select=ChainResidueSelect(surface_residues))

    return seqres2atmseq2surf_mask


def sequence_filtering(ag_pdb_df, seqres2surf_mask, output_dir, pdb_id):
    """Filter antigen pdb dataframe and save sequence files."""
    filtered_antigen_df  = ag_pdb_df[ag_pdb_df["residue_number"].map(seqres2surf_mask) == 1]

    # atmseq2surf_seq = "".join(filtered_antigen_df["residue_name"].map(AA_MAP))

    filtered_residues = filtered_antigen_df[["residue_number", "residue_name"]].drop_duplicates()
    atmseq = "".join(filtered_residues["residue_name"].map(AA_MAP)) # surface atmseq
    return filtered_antigen_df, atmseq          #, atmseq2surf_seq = atmseq

def atmseq2epitope_mapping(filtered_antigen_df, seqres2epitope_mask, output_dir, pdb_id):
    """Generate and save epitope mapping files."""
    filtered_epitope_df = filtered_antigen_df[filtered_antigen_df["residue_number"].map(seqres2epitope_mask) == 1]
    # filtered_epitope_residues = filtered_antigen_df[filtered_antigen_df["residue_number"].map(seqres2epitope_mask) == 1]["residue_number"].unique()
    filtered_antigen_df = filtered_antigen_df[filtered_antigen_df.loc[:,"atom_name"]=="CA"]
    
    """
    TODO: 
    - issue: the binary epitope labels from antigen df are not filtered for the carbon atoms
    - solution: filter the epitope df to only carbon atoms
    - create binary atmseq2epitope mapping as binary_epitope_labels of size equal to atmseq
    """
    filtered_epitope_df = filtered_epitope_df[filtered_epitope_df.loc[:,"atom_name"]=="CA"]
    epitope_list = [f"{res}_{resname}" for res, resname in zip(filtered_epitope_df["residue_number"], filtered_epitope_df["residue_name"])]
    """
    TODO: 
    - save atmseq2epitope labels with one letter amino acid codes as well
    """
    atmseq2epitope_labels = [f"{res}_{resname}" for res, resname in zip(filtered_epitope_df["residue_number"], filtered_epitope_df["residue_name"].map(AA_MAP))]
    binary_epitope_labels = np.array([1 if res in filtered_epitope_df["residue_number"].values else 0 for res in filtered_antigen_df["residue_number"]])
    return epitope_list, binary_epitope_labels, atmseq2epitope_labels


def main():
    parser = argparse.ArgumentParser(description="Split an antigen-antibody complex into separate PDB files.")
    parser.add_argument("ag_pdb_dir", type=str, help="Input antigen PDB dir")
    parser.add_argument("masks_graph_pt_dir", type=str, help="Graph masks dir")
    parser.add_argument("ag_surf_pdb_out_dir", type=str, help="Output surface antigen PDB dir")
    parser.add_argument("ag_sequences_out_dir", type=str, help="Output antigen seqres & epitope dir")
    args = parser.parse_args()
    
    os.makedirs(args.ag_sequences_out_dir, exist_ok=True)
    combined_ag_seqres = []
    combined_ag_atmseq = []
    combined_atmseq2epitope_residues = []
    combined_seqres2epitope_residues = []
    combined_seqres2epi_atmseq2epi_residues = []
    combined_binary_labels = []
    combined_seqres2epitope_seq =  []
    combined_atmres2epitope_seq =  []
    combined_atmres2epitope_seqres2epitope_seq =  []
    combined_atmseq2surf_seqres2surf_seq =  []


    # split_complex(args.ag_ab_pdb_files, args.ab_output, args.ag_output)

    all_mask_files = os.listdir(args.masks_graph_pt_dir)
    """
    FIXME: 
    - remove 5nj6_0P from the graphs list because of alignment error
    - remaining files are 1722
    NOTE: 5nj6_0P.pdb moved from structures directory to main asep directory
    """
    all_mask_files.remove("5nj6_0P.pt")
    for file in range(len(all_mask_files)):
        # print(file, all_mask_files)

        pdb_id = os.path.basename(all_mask_files[file]).split(".")[0]
        # Load PDB and masks
        ag_pdb_df, seqres2surf_mask, seqres2epitope_mask, seqres, ag_seqres2epi_labels, seqres2surf_seq = load_pdb_and_masks(pdb_id,
                args.ag_pdb_dir, args.masks_graph_pt_dir)

        # Perform surface mapping and save PDB
        seqres2atmseq2surf_mask  = seqres2surf_mapping(args.ag_pdb_dir, ag_pdb_df, 
                            seqres2surf_mask, args.ag_surf_pdb_out_dir, pdb_id)

        filtered_antigen_df, atmseq = sequence_filtering(ag_pdb_df, 
                                            seqres2surf_mask, args.ag_sequences_out_dir, pdb_id)
        
        epitope_list, binary_epitope_labels, atmseq2epitope_labels = atmseq2epitope_mapping(filtered_antigen_df, 
                                            seqres2epitope_mask, args.ag_sequences_out_dir, pdb_id)

        combined_ag_seqres.append(f">{pdb_id}\n{seqres.item()}\n")
        combined_ag_atmseq.append(f">{pdb_id}\n{atmseq}\n")

        combined_atmseq2epitope_residues.append([pdb_id, epitope_list])
        combined_seqres2epitope_residues.append([pdb_id, ag_seqres2epi_labels])
        
        combined_seqres2epi_atmseq2epi_residues.append([pdb_id, ag_seqres2epi_labels, 
            atmseq2epitope_labels, len(ag_seqres2epi_labels), len(atmseq2epitope_labels),
            len(ag_seqres2epi_labels)== len(atmseq2epitope_labels), ag_seqres2epi_labels== atmseq2epitope_labels])
        
        combined_atmseq2surf_seqres2surf_seq.append([pdb_id, seqres2surf_seq, 
            atmseq, len(seqres2surf_seq), len(atmseq),
            len(seqres2surf_seq)== len(atmseq), seqres2surf_seq== atmseq])
        
        combined_binary_labels.append([pdb_id, binary_epitope_labels])
        # Create a NumPy dictionary with seqres and epitope binary labels
        seqres_epitope_dict = {
            "pdb_id": pdb_id, 
            "seqres": np.array(seqres).item(),  # Filtered sequence in 1-letter codes
            "epitope": np.array(seqres2epitope_mask),  # Binary labels (1 for epitope, 0 otherwise)
            "seqres2atmseq2surf_mask": seqres2atmseq2surf_mask
        }
        atmseq_epitope_dict = {
            "pdb_id": pdb_id, 
            "atmseq": np.array(atmseq).item(),  # Filtered sequence in 1-letter codes
            "epitope": binary_epitope_labels , # Binary labels (1 for epitope, 0 otherwise)
            "seqres2atmseq2surf_mask": seqres2atmseq2surf_mask
        }
        atmseq_seqres_epitope_dict = {
            "pdb_id": pdb_id, 
            "seqres": np.array(seqres).item(),  # Filtered sequence in 1-letter codes
            "epitope": np.array(seqres2epitope_mask),  # Binary labels (1 for epitope, 0 otherwise)
            "seqres2atmseq2surf_mask": seqres2atmseq2surf_mask,
            "atmseq": np.array(atmseq).item(),  # Filtered sequence in 1-letter codes
            "epitope": binary_epitope_labels , # Binary labels (1 for epitope, 0 otherwise)
            "seqres2atmseq2surf_mask": seqres2atmseq2surf_mask
        }
        combined_seqres2epitope_seq.append(seqres_epitope_dict)
        combined_atmres2epitope_seq.append(atmseq_epitope_dict)
        combined_atmres2epitope_seqres2epitope_seq.append(atmseq_seqres_epitope_dict)
    
    """
    NOTE: 
    - save both seqres and surface atmseq 
    - applying seqres2atmseq2surf_mask on seqres should give us surface atmseq
    """

    with open(Path(args.ag_sequences_out_dir) / "ag_seqres.fasta", "w") as f:
        f.writelines(combined_ag_seqres)

    with open(Path(args.ag_sequences_out_dir) / "ag_surf_atmseq.fasta", "w") as f:
        f.writelines(combined_ag_atmseq)

    """
    TODO: 
    - save two files with pdb_id, {seqres. atmseq}, and epitope labels:
        - seqres2epitope.npy (epitope labels filtered from the SEQRES) to be used for PLM feature extraction
        - atmseq2epitope.npy (epitope labels filtered from the ATMSEQ2SURF) to be used for GNN feature extraction
        - both files should contain the same epitope residues but the sequences are different
    """

    np.save(Path(args.ag_sequences_out_dir) / "ag_seqres2epitope_labels.npy", np.array(combined_seqres2epitope_seq, dtype=object))
    np.save(Path(args.ag_sequences_out_dir) / "ag_atmseq2epitope_labels.npy", np.array(combined_atmres2epitope_seq, dtype=object))
    np.save(Path(args.ag_sequences_out_dir) / "ag_seqres2epitope_atmseq2epitope_labels.npy", np.array(combined_atmres2epitope_seqres2epitope_seq, dtype=object))

    epitope_df = pd.DataFrame(combined_atmseq2epitope_residues, columns=["pdbid", "epitope"])
    epitope_df.to_csv(Path(args.ag_sequences_out_dir) / "atmseq2epitope_residues.csv", index=False)

    epitope_df = pd.DataFrame(combined_seqres2epitope_residues, columns=["pdbid", "epitope"])
    epitope_df.to_csv(Path(args.ag_sequences_out_dir) / "seqres2epitope_residues.csv", index=False)

    """
    NOTE: 
    - this joint table of seqres2epitope and atmseq2epitope labels is to compare if they are equal
        - turns out 58 antigens have different epitope sequences with some missing residues &
        - 16 antigens have different sequence lengths
    """

    epitope_df = pd.DataFrame(combined_seqres2epi_atmseq2epi_residues, columns=["pdbid", "seqres2epitope",
                                    "atmseq2epitope", "len_seqres2epitope", "len_atmseq2epitope", "equal_len", "same_seq"])
    epitope_df.to_csv(Path(args.ag_sequences_out_dir) / "seqres2epi_atmseq2epi_residues.csv", index=False)

    surf_df = pd.DataFrame(combined_atmseq2surf_seqres2surf_seq, columns=["pdbid", "seqres2surf",
                                    "atmseq2surf", "len_seqres2surf", "len_atmseq2surf", "equal_len", "same_seq"])
    surf_df.to_csv(Path(args.ag_sequences_out_dir) / "atmseq2surf_seqres2surf_seq.csv", index=False)

    np.save(Path(args.ag_sequences_out_dir) / "ag_binary_epitope_labels.npy", np.array(combined_binary_labels, dtype=object))
    print(f"Processed files saved to: {args.ag_sequences_out_dir}")

    print(f"Filtered surface PDB files saved to: {args.ag_surf_pdb_out_dir}")
    print(f"Files saved:\n- ag_seqres.fasta \n- ag_surf_atmseq.fasta \n- ag_seqres2epitope_labels.npy\n- ag_atmseq2epitope_labels.npy")
    print( f"- atmseq2epitope_residues.csv\n- seqres2epitope_residues.csv \n- seqres2epi_atmseq2epi_residues.csv")
    print(f"- ag_atmseq2epitope_labels.npy \n- ag_binary_epitope_labels.npy")




if __name__ == "__main__":
    main()


"""
python3 seqres2surf_mapping.py  \
 \
 \

"""

"""
python3 seqres2surf_mapping.py  \
 \
 \

"""