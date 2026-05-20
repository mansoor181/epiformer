"""
TODO:
1. create and preprocess/refine antibody structures using seqres2cdr_mapping.py
    - atmseq2cdr and atmseq2paratope mapping

    dict(['complex_code', 'coord_AG', 'label_AG', 'coord_AB', 'label_AB', 
    'edge_AGAB', 'edge_AB', 'edge_AG', 'vertex_AB', 'vertex_AG', 'AbLang_AB', 'ESM1b_AG'])

    1.1 construct 62-dimensional vectors for AB and AG (vertex_AG and vertex_AB)
        - A one-hot encoding representing residue types, with a dimension of 20.
        - A PSSM obtained through PSI-BLAST computation, with a dimension of 20.
        - The absolute and relative SASA computed by STRIDE, with a dimension of 2.
        - A local amino acid profile reveals the frequency of each amino acid type within an 8A ̊ radius of the residue, with a dimension of 20.
    1.2 create label_AG and label_AG of size ATMSEQ (seqres2atmseq masking)
    1.3 generate AG sequence embeddings using ESM2-1b (1280) and AB-Lang (768)
        - mask the SEQRES after embedding generation to downstream surf or cdr mask
        - how to encode AB sequence??? not AntiBERTy surely (512)
            - use AbLang model for residue embedding of size 768 per residue
            - it's a RoBERTa inspired language model
    1.4 construct AB and AG individual and joint edges 
        - how to construct these edges? distance threshold of 10A

2. load cvdata.pkl and testdata.pkl and analyze at the data
3. add comments to the code
4. reproduce the reported results in the paper
5. add evaluation metrics of F1 score, precision, recall, BAcc
6. refactor the hyperparameters tuning code (learning rate, batch size, optimizer)


NOTE: 
working of the script:
- takes in cvdata.pkl
- performs k-fold cross-validation
- generates graph using cvdata for each batch in a fold `CreateGearnetGraph()`
- training, validation, and testing in `main.py`
- compute the evaluation metrics for AB and AG
- during testing, calculates metrics for each test sample and takes the average


NOTE:
- torchdrug can't run on local mac due to some c++ error at the backend 

"""


# add other directories to the path to import modules
import sys, os
sys.path.append( os.path.abspath(os.path.join(os.getcwd(), 'm3epi')))  # cd m3epi/code
sys.path.append( os.path.abspath(os.path.join(os.getcwd(), '../../../m3epi')))
sys.path.append( os.path.abspath(os.path.join(os.getcwd(), '../../walle')))


import numpy as np
import pandas as pd
import torch, re
from scipy.stats import norm
from Bio import SeqIO
from Bio.PDB import PDBParser, Polypeptide, PDBIO
from biopandas.pdb import PandasPdb
# from prody import parsePDBHeader
from torch.utils.data import DataLoader, Dataset
from Bio.PDB import *
from collections import Counter
import ablang


import warnings
warnings.filterwarnings('ignore')


proj_dir = os.path.join(os.getcwd(), '../../../')
dataset_dir = os.path.join(proj_dir, "data/")
figures_dir = os.path.join(proj_dir, "figures/")
results_dir = os.path.join(proj_dir, "results/hgraphepi/baselines/mipe/")

asep_data_dir = os.path.join(dataset_dir, "asep/")
asep_structures_dir = os.path.join(asep_data_dir, "structures2/")
asep_graphs_dir = os.path.join(asep_data_dir, "asepv1_interim_graphs/")
asep_sequences_dir = os.path.join(asep_data_dir, "sequences/")
asep_processed_data_path = os.path.join(asep_data_dir, "processed")
asep_test_dir = os.path.join(asep_data_dir, "test/")
asep_trans_baselines_dir = os.path.join(asep_data_dir, "trans_baselines")
orig_baselines_dataset_dir = os.path.join(dataset_dir, "orig_baselines")

asep_ag_atmseq2surf_dir = os.path.join(asep_data_dir, "antigen/atmseq2surf")
asep_ab_ag_sequences_fasta_path = os.path.join(asep_sequences_dir, "asep_ab_ag_seqres_1722.fasta")

asep_ab_atmseq2cdr_dir = os.path.join(asep_data_dir, "antibody/atmseq2cdr/")

asep_dict_pre_cal_path = os.path.join(asep_data_dir, "processed", 'dict_pre_cal.pt')
asep_dict_pre_cal_esm2_esm2_path = os.path.join(asep_processed_data_path, 'dict_pre_cal_esm2_esm2.pt')


# MIPE data
mipe_orig_data_dir = os.path.join(os.getcwd(), "data")
mipe_asep_transform_dir = os.path.join(asep_trans_baselines_dir, "mipe")
mipe_cvdata_pkl_path = os.path.join(mipe_asep_transform_dir, "mipe_cvdata_cpu.pkl")
mipe_testdata_pkl_path = os.path.join(mipe_asep_transform_dir, "testdata.pkl")
mipe_test_results = pd.read_csv(os.path.join(results_dir, "test_results.csv"))

# M3Epi data
m3epi_asep_transform_dir = os.path.join(asep_data_dir, "m3epi")



AA_MAP = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G",
    "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
    "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V",
    "TRP": "W", "TYR": "Y"
}


""" 
TODO: 
- load antibody cdr pdb files and reorder as H+L chains
- load antigen surf pdb files
- filter out the 3d coordinates of the CA atom from these pdb files
1.1 construct 62-dimensional vectors for AB and AG (vertex_AG and vertex_AB)
    - A one-hot encoding representing residue types, with a dimension of 20.
    - A PSSM obtained through PSI-BLAST computation, with a dimension of 20.
    - The absolute and relative SASA computed by STRIDE, with a dimension of 2.
    - A local amino acid profile reveals the frequency of each amino acid type 
    within an 8A ̊ radius of the residue, with a dimension of 20.

"""



def get_local_aa_profile(pdb_file, radius=8.0):
    """
    Compute 20D amino acid frequency profiles for each residue within 8Å radius
    
    Args:
        pdb_file: Path to PDB file
        radius: Radius in Angstroms (default: 8Å)
    
    Returns:
        numpy array of shape (n_residues, 20) 
    """
    # Parse PDB structure
    parser = PDBParser()
    structure = parser.get_structure("protein", pdb_file)
    
    # Get all residues
    residues = [res for res in structure.get_residues() if is_aa(res)]
    aa_types = [res.get_resname() for res in residues]
    
    # Get Cα coordinates (or CB for non-glycine)
    ca_coords = []
    for res in residues:
        if 'CA' in res:
            ca_coords.append(res['CA'].get_coord())
        elif 'CB' in res:
            ca_coords.append(res['CB'].get_coord())
    
    # Compute pairwise distances
    dist_matrix = np.zeros((len(residues), len(residues)))
    for i in range(len(residues)):
        for j in range(len(residues)):
            dist_matrix[i,j] = np.linalg.norm(ca_coords[i] - ca_coords[j])
    
    # Generate profiles
    aa_order = AA_MAP.keys()  # Fixed AA order
    
    profiles = []
    for i in range(len(residues)):
        # Find residues within radius
        neighbors = np.where(dist_matrix[i] < radius)[0]
        
        # Count AA types in neighborhood
        neighbor_aas = [aa_types[j] for j in neighbors]
        counts = Counter(neighbor_aas)
        
        # Create 20D vector
        profile = [counts.get(aa, 0) for aa in aa_order]
        profiles.append(profile)
    
    return np.array(profiles, dtype=np.float32)




# Define the set of 20 standard amino acids
amino_acids = AA_MAP.values() 

def create_one_hot_encoding(seq):
    # Create a DataFrame for one-hot encoding
    one_hot_df = pd.DataFrame(0, index=np.arange(len(seq)), columns=list(amino_acids))

    # Fill the DataFrame with one-hot encoding
    for i, aa in enumerate(seq):
        if aa in amino_acids:
            one_hot_df.at[i, aa] = 1
    
    return one_hot_df



asep_graphs_processed = torch.load(asep_dict_pre_cal_esm2_esm2_path)

asep_mipe_transformed = []

"""
NOTE: 
MIPE data needs to follow the following structure:
    - dict(['complex_code', 'coord_AG', 'label_AG', 'coord_AB', 'label_AB', 
    'edge_AGAB', 'edge_AB', 'edge_AG', 'vertex_AB', 'vertex_AG', 'AbLang_AB', 'ESM1b_AG'])
"""

heavy_ablang = ablang.pretrained("heavy") # Use "light" if you are working with light chains
heavy_ablang.freeze()

light_ablang = ablang.pretrained("light")
light_ablang.freeze()

fasta_sequences = SeqIO.parse(open(asep_ab_ag_sequences_fasta_path),'fasta')


######## create and transform graphs pickle dataset #######

asep_m3epi_transformed = []

# dict(['complex_code', 'coord_AG', 'label_AG', 'coord_AB', 'label_AB', 
# 'edge_AGAB', 'edge_AB', 'edge_AG', 'vertex_AB', 'vertex_AG', 'AbLang_AB', 'ESM1b_AG'])

heavy_ablang = ablang.pretrained("heavy") # Use "light" if you are working with light chains
heavy_ablang.freeze()

light_ablang = ablang.pretrained("light")
light_ablang.freeze()

fasta_sequences = SeqIO.parse(open(asep_ab_ag_sequences_fasta_path),'fasta')

i = 0
for fasta in fasta_sequences:
    asep_m3epi_transformed_dict = {}

    name, sequence = fasta.id, str(fasta.seq)

    pdb_id = name.split("|")[0]
    asep_m3epi_transformed_dict["complex_code"] = pdb_id

    H_chain = sequence.split(":")[0]
    L_chain = sequence.split(":")[1]
    Ag_chain = sequence.split(":")[2]
    """ 
    FIXME: 
    - the size of ab cdr coordinates don't match AbLang embeddings (seqres2cdr \neq atmseq2cdr)
    - probably an issue with the cdr atmseq being saved 
    - atmseq2cdr_seq is correct in seqres2cdr_mapping with size equal to seqres2cdr_seq 
    """

    if len(H_chain) and len(L_chain) <= 157 and pdb_id != "5ies_0P":

        asep_graphs_file = torch.load(os.path.join(asep_graphs_dir, f"{pdb_id}.pt"))
        seqres2cdr_mask = torch.tensor(asep_graphs_file["mapping"]["ab"]["seqres2cdr"]).bool()
        seqres2surf_mask = torch.tensor(asep_graphs_file["mapping"]["ag"]["seqres2surf"]).bool()
        
        heavy_rescodings = torch.tensor(heavy_ablang(H_chain, mode='rescoding'))
        light_rescodings = torch.tensor(light_ablang(L_chain, mode='rescoding'))

        ab_rescodings = torch.cat((heavy_rescodings, light_rescodings), dim=1).squeeze()
        # asep_m3epi_transformed_dict["AbLang_AB"] = ab_rescodings[seqres2cdr_mask].numpy()
        # asep_m3epi_transformed_dict["ESM1b_AG"] = asep_graphs_processed[pdb_id]["x_g"].numpy()

        ab_plm_embeddings = ab_rescodings[seqres2cdr_mask].numpy()
        ag_plm_embeddings = asep_graphs_processed[pdb_id]["x_g"].numpy()

        asep_m3epi_transformed_dict["edge_AG"] = asep_graphs_processed[pdb_id]["edge_index_g"].tolist()
        asep_m3epi_transformed_dict["edge_AB"] = asep_graphs_processed[pdb_id]["edge_index_b"].tolist()
        """ 
        FIXME: 
        - swap `edge_index_bg` to `edge_index_gb` as is needed for `edge_AGAB`
        - originally first row contains ab nodes while 2nd row contains ag nodes
        """
        edge_index_gb = torch.tensor([asep_graphs_processed[pdb_id]["edge_index_bg"][1].tolist(),
                                            asep_graphs_processed[pdb_id]["edge_index_bg"][0].tolist()])
        asep_m3epi_transformed_dict["edge_AGAB"] = edge_index_gb.tolist()
        
        asep_m3epi_transformed_dict["label_AG"] = asep_graphs_processed[pdb_id]["y_g"].tolist()
        asep_m3epi_transformed_dict["label_AB"] = asep_graphs_processed[pdb_id]["y_b"].tolist()

        ag_pdb_file_path = os.path.join(asep_ag_atmseq2surf_dir, f'{pdb_id}_surf.pdb')
        ag_pdb_df = PandasPdb().read_pdb(ag_pdb_file_path)
        filtered_ag_df = ag_pdb_df.df["ATOM"][ag_pdb_df.df["ATOM"].loc[:,"atom_name"]=="CA"]
        ag_pdb_coordinates = filtered_ag_df[["x_coord", "y_coord", "z_coord"]]

        # Read antibody PDB for both chains
        ab_pdb_file_path = os.path.join(asep_ab_atmseq2cdr_dir, f'{pdb_id}_cdr.pdb')
        ab_pdb_df = PandasPdb().read_pdb(ab_pdb_file_path)
        ab_pdb_df = ab_pdb_df.get_model(1).df["ATOM"]

        # Get residue numbers (indices) that are CDR (mask = 1) for both chains
        cdr_pdb_df_L = ab_pdb_df[ ab_pdb_df["chain_id"] == 'L']
        cdr_pdb_df_H = ab_pdb_df[ ab_pdb_df["chain_id"] == 'H']

        # enforce H+L chain order for the filtered ab dataframe to do cdr and paratope masking later on
        filtered_ab_df = pd.concat([cdr_pdb_df_H, cdr_pdb_df_L])
        filtered_ab_df = filtered_ab_df[filtered_ab_df.loc[:,"atom_name"]=="CA"]
        ab_pdb_coordinates = filtered_ab_df[["x_coord", "y_coord", "z_coord"]]

        asep_m3epi_transformed_dict["coord_AG"] = ag_pdb_coordinates.values.tolist()
        asep_m3epi_transformed_dict["coord_AB"] = ab_pdb_coordinates.to_numpy()

        """
        TODO: 
        - concatenate PLM embeddings to ag and ab vertex features
        - ESM2 for Ag and AbLang for Ab
        """

        ab_atmseq = "".join(filtered_ab_df["residue_name"].map(AA_MAP))
        ab_one_hot_df = create_one_hot_encoding(ab_atmseq)
        ag_atmseq = "".join(filtered_ag_df["residue_name"].map(AA_MAP))
        ag_one_hot_df = create_one_hot_encoding(ag_atmseq)

        ag_local_profiles = get_local_aa_profile(ag_pdb_file_path)
        ab_local_profiles = get_local_aa_profile(ab_pdb_file_path)

        # ag_vertex_features = np.concatenate([ag_local_profiles, np.array(ag_one_hot_df)], axis=1)
        # ab_vertex_features = np.concatenate([ab_local_profiles, np.array(ab_one_hot_df)], axis=1)

        # new vertex dimensions = 40 + 480 = 520
        ag_vertex_features = np.concatenate([ag_local_profiles, np.array(ag_one_hot_df), ag_plm_embeddings], axis=1)
        # new vertex dimensions = 40 + 768 = 808
        ab_vertex_features = np.concatenate([ab_local_profiles, np.array(ab_one_hot_df), ab_plm_embeddings], axis=1)

        asep_m3epi_transformed_dict["vertex_AG"] = ag_vertex_features
        asep_m3epi_transformed_dict["vertex_AB"] = ab_vertex_features

        asep_m3epi_transformed.append(asep_m3epi_transformed_dict)

        i = i +1
        if i % 100 == 0:
            print(f"Processed {i} files..")
            # break

    else:
        print("Can't generate embeddings..")
        print(f"Sequence length of antibody {pdb_id} chains is more than 157..")
        

    # print(pdb_id, H_chain, L_chain, Ag_chain)
torch.save(asep_m3epi_transformed, os.path.join(m3epi_asep_transform_dir, "asep_m3epi_transformed.pkl" ) )

"""
NOTE:
we skip these files from analysis:
1. skip 5ies_0P.pdb because its seqres2cdr_seq and atmseq2cdr have different lengths
2. skip 4hg4_8P.pdb because its heavy chain has length more than 157 
    which is why AbLang embeddings can't be generated
"""

"""
Usage: cd m3epi/code
python data/preprocess.py
"""