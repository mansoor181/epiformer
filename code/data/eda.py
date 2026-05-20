# %%
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
    1.3 generate AG sequence embeddings using ESM2-1b (1280) and AB (768)
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

# %%
# add other directories to the path to import modules
import sys, os
sys.path.append( os.path.abspath(os.path.join(os.getcwd(), 'm3epi')))  # cd m3epi/code
sys.path.append( os.path.abspath(os.path.join(os.getcwd(), '../../../m3epi')))
sys.path.append( os.path.abspath(os.path.join(os.getcwd(), '../../walle')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))

# %%
a = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
torch.sigmoid(a.sum(dim=1))

# %%
import pickle
import _pickle as cPickle
import datetime
from Bio import SeqIO

import numpy as np
import pandas as pd
import torch, re
import seaborn as sns
from scipy.stats import norm
import matplotlib.pyplot as plt
import pickle, h5py
import _pickle as cPickle
import datetime
from Bio import SeqIO
from Bio.PDB import PDBParser, Polypeptide, PDBIO
from biopandas.pdb import PandasPdb
# from prody import parsePDBHeader
from typing import Optional
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

import warnings
warnings.filterwarnings('ignore')



# %%
proj_dir = os.path.join(os.getcwd(), '../../../') # cd m3epi/code
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

asep_ag_structures_dir = os.path.join(asep_data_dir, "antigen/structures")
asep_ag_sequences_dir = os.path.join(asep_data_dir, "antigen/sequences")
asep_ag_atmseq2surf_dir = os.path.join(asep_data_dir, "antigen/atmseq2surf")
asep_ab_ag_sequences_fasta_path = os.path.join(asep_sequences_dir, "asep_ab_ag_seqres_1722.fasta")

asep_ab_structures_dir = os.path.join(asep_data_dir, "antibody/structures/")
asep_ab_sequences_dir = os.path.join(asep_data_dir, "antibody/sequences/")
asep_ab_atmseq2cdr_dir = os.path.join(asep_data_dir, "antibody/atmseq2cdr/")
asep_ab_test_atmseq2cdr_dir = os.path.join(asep_data_dir, "antibody/test/atmseq2cdr/")

asep_dict_pre_cal_path = os.path.join(asep_data_dir, "processed", 'dict_pre_cal.pt')
asep_dict_pre_cal_esm2_esm2_path = os.path.join(asep_processed_data_path, 'dict_pre_cal_esm2_esm2.pt')

# antigen sequences and epitope labels
ag_atmseq2epitope_labels = np.load(os.path.join(asep_ag_sequences_dir, "ag_atmseq2epitope_labels.npy"), allow_pickle=True)
ag_seqres2epitope_labels = np.load(os.path.join(asep_ag_sequences_dir, "ag_seqres2epitope_labels.npy"), allow_pickle=True)
ag_binary_epitope_labels = np.load(os.path.join(asep_ag_sequences_dir, "ag_binary_epitope_labels.npy"), allow_pickle=True)
ag_atmseq2epitope_residues = pd.read_csv(os.path.join(asep_ag_sequences_dir, "atmseq2epitope_residues.csv"))
ag_seqres2epitope_residues = pd.read_csv(os.path.join(asep_ag_sequences_dir, "seqres2epitope_residues.csv"))
# ag_atmseq2epitope_residues and ag_seqres2epitope_residues have same epitope residues

# antibody sequences and paratope labels
ab_cdr2paratope_labels = np.load(os.path.join(asep_ab_sequences_dir, "cdr2paratope_mask.npy"), allow_pickle=True)
ab_seqres2paratope_labels = np.load(os.path.join(asep_ab_sequences_dir, "seqres2paratope_mask.npy"), allow_pickle=True)
# ab_paratope_labels_chainwise = pd.read_csv(os.path.join(asep_ab_sequences_dir, "seqres2atmseq_mask_ab_HL_chain.csv"))
ab_atmseq2paratope_residues = pd.read_csv(os.path.join(asep_ab_sequences_dir, "atmseq2paratope_residues.csv"))
ab_seqres2paratope_residues = pd.DataFrame(np.load(os.path.join(asep_ab_sequences_dir, "seqres2paratope_residues.npy"), allow_pickle=True),
                                           columns=["pdbid", "paratope", "seqres2paratope_mask", "seqres2cdr_mask"])

# MIPE data
mipe_orig_data_dir = os.path.join(os.getcwd(), "data")
mipe_asep_transform_dir = os.path.join(asep_trans_baselines_dir, "mipe")
asep_m3epi_transformed_path = os.path.join(mipe_asep_transform_dir, "mipe_cvdata_cpu.pkl")
mipe_testdata_pkl_path = os.path.join(mipe_asep_transform_dir, "testdata.pkl")
mipe_test_results = pd.read_csv(os.path.join(results_dir, "test_results.csv"))
mipe_test_results = pd.read_csv(os.path.join(results_dir, "test_results_asep_data.csv"))

# M3Epi data
m3epi_asep_transform_dir = os.path.join(asep_data_dir, "m3epi")
m3epi_pkl_path = os.path.join(m3epi_asep_transform_dir, "asep_mipe_transformed_100_examples.pkl")


# %%
AA_MAP = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G",
    "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
    "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V",
    "TRP": "W", "TYR": "Y"
}

# %%
##### original asep data #####
asep_graphs_processed = torch.load(asep_dict_pre_cal_esm2_esm2_path)
print(asep_graphs_processed["7f3q_0P"])
print(asep_graphs_processed["7f3q_0P"]["edge_index_bg"]) # first row represents Ab, second row Ag
print(len(asep_graphs_processed["7f3q_0P"]["x_g"]), len(asep_graphs_processed["7f3q_0P"]["x_b"]))

# %%
###### m3epi transformed data @ #######
asep_m3epi_transformed = torch.load(m3epi_pkl_path)
asep_m3epi_transformed[0]

# %%
print(len(asep_m3epi_transformed[0]["vertex_AG"]))
print(len(asep_m3epi_transformed[0]["vertex_AB"]))
print(asep_m3epi_transformed[0]["edge_AGAB"])

# %%
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

pdb_id = "3v6o_1P"
ag_pdb_df = PandasPdb().read_pdb(os.path.join(asep_ag_atmseq2surf_dir, f'{pdb_id}_surf.pdb'))
filtered_ag_df = ag_pdb_df.df["ATOM"][ag_pdb_df.df["ATOM"].loc[:,"atom_name"]=="CA"]
ag_pdb_coordinates = filtered_ag_df[["x_coord", "y_coord", "z_coord"]]

# Read antibody PDB for both chains
ab_pdb_df = PandasPdb().read_pdb(os.path.join(asep_ab_atmseq2cdr_dir, f'{pdb_id}_cdr.pdb'))
ab_pdb_df = ab_pdb_df.get_model(1).df["ATOM"]

# Get residue numbers (indices) that are CDR (mask = 1) for both chains
cdr_pdb_df_L = ab_pdb_df[ ab_pdb_df["chain_id"] == 'L']
cdr_pdb_df_H = ab_pdb_df[ ab_pdb_df["chain_id"] == 'H']

# enforce H+L chain order for the filtered ab dataframe to do cdr and paratope masking later on
filtered_ab_df = pd.concat([cdr_pdb_df_H, cdr_pdb_df_L])
filtered_ab_df = filtered_ab_df[filtered_ab_df.loc[:,"atom_name"]=="CA"]
ab_pdb_coordinates = filtered_ab_df[["x_coord", "y_coord", "z_coord"]]
# print(ab_pdb_coordinates) 
print(ag_pdb_coordinates)

# %%
######## create and transform graphs pickle dataset #######
import ablang

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
    - the size of ab surface coordinates don't match AbLang embeddings (seqres2cdr \neq atmseq2cdr)
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
        asep_m3epi_transformed_dict["AbLang_AB"] = ab_rescodings[seqres2cdr_mask].numpy()
        asep_m3epi_transformed_dict["ESM1b_AG"] = asep_graphs_processed[pdb_id]["x_g"].numpy()

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

        ab_atmseq = "".join(filtered_ab_df["residue_name"].map(AA_MAP))
        ab_one_hot_df = create_one_hot_encoding(ab_atmseq)
        ag_atmseq = "".join(filtered_ag_df["residue_name"].map(AA_MAP))
        ag_one_hot_df = create_one_hot_encoding(ag_atmseq)

        ag_local_profiles = get_local_aa_profile(ag_pdb_file_path)
        ab_local_profiles = get_local_aa_profile(ab_pdb_file_path)

        ag_vertex_features = np.concatenate([ag_local_profiles, np.array(ag_one_hot_df)], axis=1)
        ab_vertex_features = np.concatenate([ab_local_profiles, np.array(ab_one_hot_df)], axis=1)

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
torch.save(asep_m3epi_transformed, os.path.join(mipe_asep_transform_dir, "asep_m3epi_transformed.pkl" ) )

"""
NOTE:
we skip these files from analysis:
1. skip 5ies_0P.pdb because its seqres2cdr_seq and atmseq2cdr have different lengths
2. skip 4hg4_8P.pdb because its heavy chain has length more than 157 
    which is why AbLang embeddings can't be generated
"""

# %% [markdown]
# ## random tests for data preprocessing and analysis

# %%
from Bio.PDB import PDBParser, NeighborSearch
import numpy as np
from Bio.PDB import *
from collections import Counter

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

ag_pdb_file = os.path.join(asep_ag_atmseq2surf_dir, f'{pdb_id}_surf.pdb')
ab_pdb_file = os.path.join(asep_ab_atmseq2cdr_dir, f'{pdb_id}_cdr.pdb')

ag_local_profiles = get_local_aa_profile(ag_pdb_file)
ab_local_profiles = get_local_aa_profile(ab_pdb_file)

# Output shape: (num_residues, 20)
ag_local_profiles.shape, ab_local_profiles.shape

# %%
#### create one hot encoding #####
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

# ag_one_hot_df = create_one_hot_encoding(ag_atmseq)
# ab_one_hot_df = create_one_hot_encoding(ab_atmseq)
# ag_one_hot_df.shape, ab_one_hot_df.shape

# %%
ab_atmseq = "".join(filtered_ab_df["residue_name"].map(AA_MAP))
ab_one_hot_df = create_one_hot_encoding(ab_atmseq)
ag_atmseq = "".join(filtered_ag_df["residue_name"].map(AA_MAP))
ag_one_hot_df = create_one_hot_encoding(ag_atmseq)

# %%
ag_vertex_features = np.concatenate([ag_local_profiles, np.array(ag_one_hot_df)], axis=1)
ab_vertex_features = np.concatenate([ab_local_profiles, np.array(ab_one_hot_df)], axis=1)
ag_vertex_features.shape, ab_vertex_features.shape

# %%
seqres2paratope_mask = np.array(ab_seqres2paratope_residues[ab_atmseq2paratope_residues["pdbid"] == "7f3q_0P"]["seqres2paratope_mask"][0])
seqres2paratope_mask

# %%
seqres2cdr_mask = np.array(ab_seqres2paratope_residues[ab_atmseq2paratope_residues["pdbid"] == "7f3q_0P"]["seqres2cdr_mask"][0])
seqres2cdr_mask

# %%
# create dictionary for faster access
ab_cdr2paratope_labels_dict = {item[0]: (item[1], item[2]) for item in ab_cdr2paratope_labels}
cdr2paratope_mask = ab_cdr2paratope_labels_dict["1a14_0P"]
cdr2paratope_mask

# %%
pdb_id = "3v6o_1P"
print(asep_graphs_processed[pdb_id]["edge_index_bg"])
# print(asep_graphs_processed[pdb_id]["edge_index_g"])
# print(asep_graphs_processed[pdb_id]["edge_index_b"])
print(asep_graphs_processed[pdb_id]["edge_index_bg"][0])
print(asep_graphs_processed[pdb_id]["edge_index_bg"][1])
edge_index_bg = torch.tensor([asep_graphs_processed[pdb_id]["edge_index_bg"][1].tolist(),
                                                  asep_graphs_processed[pdb_id]["edge_index_bg"][0].tolist()])
edge_index_bg 


# %%
pdb_ids = []
for i in range(len(asep_m3epi_transformed)):
    pdb_ids.append(asep_m3epi_transformed[i]["complex_code"])
pdb_ids

# %%
asep_m3epi_transformed = torch.load(m3epi_pkl_path)
asep_m3epi_transformed

# %%
print(asep_m3epi_transformed[90]["complex_code"])
print(len(asep_m3epi_transformed[90]["AbLang_AB"]))
print(len(asep_m3epi_transformed[90]["coord_AB"]))
print(len(asep_m3epi_transformed[90]["vertex_AB"]))

print(len(asep_m3epi_transformed[22]["ESM1b_AG"]))
print(len(asep_m3epi_transformed[22]["coord_AG"]))
print(len(asep_m3epi_transformed[22]["vertex_AG"]))

# %%
pdb_id = "7u8m_3P"

pdb_path = f"{asep_ab_atmseq2cdr_dir}/{pdb_id}_cdr.pdb"
ppdb = PandasPdb().read_pdb(pdb_path)
atomic_df = ppdb.get_model(1).df["ATOM"]
atomic_df = atomic_df[atomic_df.loc[:,"atom_name"]=="CA"]

cdr_pdb_df_L = atomic_df[ atomic_df["chain_id"] == 'L']
cdr_pdb_df_H = atomic_df[ atomic_df["chain_id"] == 'H']

# enforce H+L chain order for the filtered ab dataframe to do cdr and paratope masking later on
filtered_ab_df = pd.concat([cdr_pdb_df_H, cdr_pdb_df_L])
filtered_ab_df

# %%
##### @@ running AbLang to generate AB embeddings ######
""" 
TODO: 
- load ab heavy and light chain sequences
- generate ablang embeddings per chain and concat them together
- store embeddings in a dictionary with pdb_id as key
- apply seqres2cdr mask to keep only cdr residue embeddings
"""

import ablang

heavy_ablang = ablang.pretrained("heavy") # Use "light" if you are working with light chains
heavy_ablang.freeze()

light_ablang = ablang.pretrained("light")
light_ablang.freeze()

""" 
FIXME: 
- pretrained AbLang model encodes heavy or light chains sequences
 with lengths less than or equal to 157
- 4hg4_8P has light chain sequence length of 163, hence AbLang can't generate embeddings
"""

h_chain = 'SPSSLSASVGDRVTITCQASQDIRKYLNWYQQKPGKAPNLLIYDASNVKTGVPSRFRGSGSGTDFTFTISSLQPEDIATYYCQQYDNLPITFGQGTRLEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESV'

l_chain = 'SPSSLSASVGDRVTITCQASQDIRKYLNWYQQKPGKAPNLLIYDASNVKTGVPSRFRGSGSGTDFTFTISSLQPEDIATYYCQQYDNLPITFGQGTRLEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESV'

heavy_rescodings = torch.tensor(heavy_ablang(h_chain, mode='rescoding'))
light_rescodings = torch.tensor(light_ablang(l_chain, mode='rescoding'))


print("-"*100)
print("The output shape of a first sequence:", heavy_rescodings[0].shape)
print("The output shape of a second sequence:", light_rescodings[0].shape)
print("This shape is different for each sequence, depending on their length.")
print("-"*100)


# %%
print(type(asep_m3epi_transformed[0]["coord_AG"]))
print(type(asep_m3epi_transformed[0]["label_AG"]))
print(type(asep_m3epi_transformed[0]["coord_AB"]))
print(type(asep_m3epi_transformed[0]["label_AB"]))
print(type(asep_m3epi_transformed[0]["edge_AGAB"]))
print(type(asep_m3epi_transformed[0]["edge_AB"]))
print(type(asep_m3epi_transformed[0]["edge_AG"]))
print(type(asep_m3epi_transformed[0]["vertex_AB"]))
print(type(asep_m3epi_transformed[0]["vertex_AG"]))
print(type(asep_m3epi_transformed[0]["AbLang_AB"]))
print(type(asep_m3epi_transformed[0]["ESM1b_AG"]))

print(type(asep_m3epi_transformed[1]["coord_AG"]))
print(type(asep_m3epi_transformed[1]["label_AG"]))
print(type(asep_m3epi_transformed[1]["coord_AB"]))
print(type(asep_m3epi_transformed[1]["label_AB"]))
print(type(asep_m3epi_transformed[1]["edge_AGAB"]))
print(type(asep_m3epi_transformed[1]["edge_AB"]))
print(type(asep_m3epi_transformed[1]["edge_AG"]))
print(type(asep_m3epi_transformed[1]["vertex_AB"]))
print(type(asep_m3epi_transformed[1]["vertex_AG"]))
print(type(asep_m3epi_transformed[1]["AbLang_AB"]))
print(type(asep_m3epi_transformed[1]["ESM1b_AG"]))



# %% [markdown]
# ## test issues with cdr extraction and alignment

# %%
import os
import argparse, torch
import pandas as pd
import numpy as np
from pathlib import Path
from biopandas.pdb import PandasPdb
from Bio.PDB import PDBIO, PDBParser, Select
import shutil
import tempfile
from typing import List
from Bio import AlignIO, SeqIO
from Bio.Align.Applications import ClustalOmegaCommandline
from Bio.Seq import Seq
from Bio.SeqIO import SeqRecord
import logging
import warnings

warnings.filterwarnings("ignore")
CLUSTAL_OMEGA_EXECUTABLE = shutil.which("clustalo")

AA_MAP = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G",
    "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
    "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V",
    "TRP": "W", "TYR": "Y"
}

class ChainSelect(Select):
    def __init__(self, chains):
        self.chains = set(chains)
    
    def accept_chain(self, chain):
        return chain.get_id() in self.chains
    
    def accept_residue(self, residue):
        return residue.id[0] == " " and residue.id[2] == " "


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

# %%
pdb_id = "7m30_2P"

atomic_df = PandasPdb().read_pdb(f"{asep_structures_dir}/{pdb_id}.pdb").get_model(1).df["ATOM"]
# atomic_df = PandasPdb().read_pdb(f"{asep_ab_structures_dir}/{pdb_id}_ab.pdb").get_model(1).df["ATOM"]
mask_data = torch.load(f"{asep_graphs_dir}/{pdb_id}.pt")

antibody_chains = ["H", "L"]
# Process heavy and light chains separately
chain_data = {}
for chain_type in ["H", "L"]:
    if chain_type not in antibody_chains:
        continue

    chain_df = atomic_df[atomic_df["chain_id"] == chain_type]
    if chain_df.empty:
        continue

    # Get SEQRES and ATMSEQ for the chain
    seqres = str(np.array(mask_data["seqres"]["ab"][chain_type]))

    atmseq_df = atomic_df[atomic_df["chain_id"] == chain_type]  # NEW LINE
    atmseq_df = atmseq_df[["residue_number", "residue_name"]].drop_duplicates()
    atmseq = "".join(atmseq_df["residue_name"].map(AA_MAP))

    # Generate alignment mask seqres2atmseq
    mask = get_seqres2atmseq_mask(seqres, atmseq, pdb_id)
    print(seqres)
    print(atmseq)
    print(np.array(mask))

# %%
atmseq_df.columns

# %%
atomic_df

# %%
# ************** antibody residue offsetting 
def parse_residue(res):
    """Extracts the base residue number from a residue identifier (e.g., '29A' → 29)."""
    base_str = ''.join([c for c in str(res) if c.isdigit()])
    return int(base_str) if base_str else None

def group_residues(residues):
    """Groups consecutive residues with the same base number."""
    groups = []
    current_group = []
    prev_base = None
    
    for res in residues:
        base = parse_residue(res)
        if base != prev_base:
            if current_group:
                groups.append(current_group)
                current_group = []
            prev_base = base
        current_group.append(res)
    
    if current_group:
        groups.append(current_group)
    
    return groups

def convert_residues(residues_ordered):
    """Converts residues with alternates into consecutive numbers, adjusting offsets."""
    groups = group_residues(residues_ordered)
    cumulative_offset = 0
    converted_numbers = []
    
    for group in groups:
        base = parse_residue(group[0])
        adjusted_base = base + cumulative_offset
        group_size = len(group)
        group_converted = [adjusted_base + i for i in range(group_size)]
        converted_numbers.extend(group_converted)
        cumulative_offset += (group_size - 1)  # Update offset for future residues
    
    return converted_numbers

# Usage in your script:
residues_ordered = ['0', '1', '2', '29', '29A', '29B', '30', '40', '40A', '40B', '40C', '45', '49', '49A', '70']

converted_numbers = convert_residues(residues_ordered)
print(converted_numbers)

# Test
# print(convert_residues(residues_ordered))

# %%
# ********** antibody seqres-atmseq chain-wise (H, L) comparison ********** #
all_mask_files = os.listdir(asep_structures_dir)
all_mask_files.remove(".DS_Store")  # Remove problematic file

ab_seqres_atmseq_comparison = []
chain_type = "H"

for mask_file in all_mask_files:
    pdb_id = mask_file.split(".")[0]

    atomic_df = PandasPdb().read_pdb(f"{asep_structures_dir}/{pdb_id}.pdb").get_model(1).df["ATOM"]

    # atomic_df = PandasPdb().read_pdb(f"{asep_ab_structures_dir}/../test/structures/{pdb_id}_ab.pdb").get_model(1).df["ATOM"]
    mask_data = torch.load(f"{asep_graphs_dir}/{pdb_id}.pt")

    antibody_chains = ["H", "L"]

    # for chain_type in ["H"] : # , "L"]:
    if chain_type not in antibody_chains:
        continue

    chain_df = atomic_df[atomic_df["chain_id"] == chain_type]
    if chain_df.empty:
        continue

    # Get SEQRES
    seqres = str(np.array(mask_data["seqres"]["ab"][chain_type]))

    # Process ATMSEQ with alternates preserved
    # First get ALL residues in original order (including alternates)
    atmseq_full = chain_df.assign(
        full_residue=chain_df["residue_number"].astype(str) + chain_df["insertion"].fillna('')
    )
    
    # Get ordered unique residues (with alternates)
    residues_ordered = atmseq_full["full_residue"].unique()
    
    converted_numbers = convert_residues(residues_ordered)
    
    # Now get ATMSEQ string with original residues (including alternates)
    atmseq_df = atmseq_full.drop_duplicates("full_residue")
    atmseq = "".join(atmseq_df["residue_name"].map(AA_MAP))

    # Generate alignment mask
    mask = get_seqres2atmseq_mask(seqres, atmseq, pdb_id)
    
    atmseq_seqres_comparison_dict = {
        "pdb_id": str(pdb_id), 
        "heavy_seqres": seqres,
        "heavy_atmseq": atmseq,
        "len_heavy_seqres": len(seqres),
        "len_heavy_atmseq": len(atmseq),
        "len_heavy_seqres_atmseq_is_equal": len(seqres) == len(atmseq),
        "heavy_seqres_atmseq_is_equal": len(seqres) == len(atmseq),
        "len_heavy_orig_residues": len(residues_ordered),
        "len_heavy_modified_indices": len(converted_numbers),
        "heavy_seqres2atmseq_mask": mask
    }

    ab_seqres_atmseq_comparison.append(atmseq_seqres_comparison_dict)

ab_seqres_atmseq_comparison_df = pd.DataFrame(ab_seqres_atmseq_comparison, columns=["pdbid", "heavy_seqres",
                                "heavy_atmseq", "len_heavy_seqres", "len_heavy_atmseq", "len_heavy_seqres_atmseq_is_equal", 
                                "heavy_seqres_atmseq_is_equal", "len_heavy_orig_residues", 
                                "len_heavy_modified_indices", "heavy_seqres2atmseq_mask"])

pdb_ids = [ab_seqres_atmseq_comparison[i]["pdb_id"] for i in range(len(ab_seqres_atmseq_comparison))]
ab_seqres_atmseq_comparison_df["pdbid"] = pdb_ids

ab_seqres_atmseq_comparison_df.to_csv(Path(asep_ab_sequences_dir) / "ab_seqres_atmseq_comparison.csv", index=False)


# %%
# ********** antigen seqres-atmseq comparison ********** #
all_mask_files = os.listdir(asep_structures_dir)
all_mask_files.remove(".DS_Store")  # Remove problematic file

ag_seqres_atmseq_comparison = []

for mask_file in all_mask_files:
    pdb_id = mask_file.split(".")[0]

    atomic_df = PandasPdb().read_pdb(f"{asep_structures_dir}/{pdb_id}.pdb").get_model(1).df["ATOM"]
    chains = atomic_df["chain_id"].unique()

    ag_chain = chains[2]

    # atomic_df = PandasPdb().read_pdb(f"{asep_ab_structures_dir}/../test/structures/{pdb_id}_ab.pdb").get_model(1).df["ATOM"]
    mask_data = torch.load(f"{asep_graphs_dir}/{pdb_id}.pt")


    chain_df = atomic_df[atomic_df["chain_id"] == ag_chain]
    if chain_df.empty:
        continue

    # Get SEQRES
    seqres = str(np.array(mask_data["seqres"]["ag"][ag_chain]))

    # Process ATMSEQ with alternates preserved
    # First get ALL residues in original order (including alternates)
    atmseq_full = chain_df.assign(
        full_residue=chain_df["residue_number"].astype(str) + chain_df["insertion"].fillna('')
    )
    
    # Get ordered unique residues (with alternates)
    residues_ordered = atmseq_full["full_residue"].unique()
    
    converted_numbers = convert_residues(residues_ordered)
    
    # Now get ATMSEQ string with original residues (including alternates)
    atmseq_df = atmseq_full.drop_duplicates("full_residue")
    atmseq = "".join(atmseq_df["residue_name"].map(AA_MAP))

    # Generate alignment mask
    mask = get_seqres2atmseq_mask(seqres, atmseq, pdb_id)
    
    atmseq_seqres_comparison_dict = {
        "pdb_id": str(pdb_id), 
        "seqres": seqres,
        "atmseq": atmseq,
        "len_seqres": len(seqres),
        "len_atmseq": len(atmseq),
        "len_seqres_atmseq_is_equal": len(seqres) == len(atmseq),
        "seqres_atmseq_is_equal": len(seqres) == len(atmseq),
        "len_orig_residues": len(residues_ordered),
        "len_modified_indices": len(converted_numbers),
        "seqres2atmseq_mask": mask
    }

    ag_seqres_atmseq_comparison.append(atmseq_seqres_comparison_dict)

ag_seqres_atmseq_comparison_df = pd.DataFrame(ag_seqres_atmseq_comparison, columns=["pdbid", "seqres",
                                "atmseq", "len_seqres", "len_atmseq", "len_seqres_atmseq_is_equal", 
                                "seqres_atmseq_is_equal", "len_orig_residues", 
                                "len_modified_indices", "seqres2atmseq_mask"])

pdb_ids = [ag_seqres_atmseq_comparison[i]["pdb_id"] for i in range(len(ag_seqres_atmseq_comparison))]
ag_seqres_atmseq_comparison_df["pdbid"] = pdb_ids

ag_seqres_atmseq_comparison_df.to_csv(Path(asep_ag_sequences_dir) / "ag_seqres_atmseq_comparison.csv", index=False)


# %%
seqres = "IRIGVSNRDFVEGMSGGTWVDVVLEHGGCVTVMAQDKPTVDIELVTTTVSNMAEVRSYYEASISDMASDSRCPTQGEAYLDKQSDTQYVCKRTLVDRGWGNGCGLFGKSLTKFACSKKMTGKSIQPENLEYRMSVHGSQHSGMIVNDTGHETDENRAKVETPNSPRAEATLGGFGSLGLDCEPRTGLDFSDLYYLTMNNKHWLVHKEWFHDIPLWHAGTPHWNNKEALVEFKDAHAKRQTVVVLGSQEGAVHTALAGALEAEMDGAKRLSSGHLKCRLKMDKLRLKGVSYSLTAAFTFTKIPAETLHGTTVEVQYAGTDGPCKVPAQMVDMQTLTPVGRLITANPVITESTENSKMMLELDPPFGDSYIVIGVGEKKITHHWHRS"
atmseq = "IRIGVSNRDFVEGMSGGTWVDVVLEHGGCVTVMAQDKPTVDIELVTTTVSNMAEVRSYYEASISDMASDSRCPTQGEAYLDKQSDTQYVCKRTLVDRGWGNGCGLFGKSLTKFACSKKMTGKSIQPENLEYRMSVHGSQHSGMIVNDTGHETDENRAKVETPNSPRAEATLGGFGSLGLDCEPRTGLDFSDLYYLTMNNKHWLVHKEWFHDIPLWHAGAPHWNNKEALVEFKDAHAKRQTVVVLGSQEGAVHTALAGALEAEMDGAKRLSSGHLKCRLKMDKLRLKGVSYSLTAAFTFTKIPAETLHGTTVEVQYAGTDGPCKVPAQMVDMQTLTPVGRLITANPVITESTENSKMMLELDPPFGDSYIVIGVGEKKITHHWHRS"
print(np.array(get_seqres2atmseq_mask(seqres, atmseq, pdb_id)))
print(seqres == atmseq)

# %%
pdb_id = "6w4m_0P"

pdb_path = f"{asep_structures_dir}/{pdb_id}.pdb"
ppdb = PandasPdb().read_pdb(pdb_path)
atomic_df = ppdb.get_model(1).df["ATOM"]
# atomic_df = PandasPdb().read_pdb().get_model(1).df["ATOM"]

# atomic_df = PandasPdb().read_pdb(f"{asep_ab_structures_dir}/../test/structures/{pdb_id}_ab.pdb").get_model(1).df["ATOM"]
mask_data = torch.load(f"{asep_graphs_dir}/{pdb_id}.pt")

antibody_chains = ["H", "L"]
chain_type = "H"

chain_df = atomic_df[atomic_df["chain_id"] == chain_type]


# Get SEQRES
seqres = str(np.array(mask_data["seqres"]["ab"][chain_type]))

# Process ATMSEQ with alternates preserved
# First get ALL residues in original order (including alternates)
atmseq_full = chain_df.assign(
    full_residue=chain_df["residue_number"].astype(str) + chain_df["insertion"].fillna('')
)

# Get ordered unique residues (with alternates)
residues_ordered = atmseq_full["full_residue"].unique()

converted_numbers = convert_residues(residues_ordered)

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
ppdb.df["ATOM"].loc[chain_df.index, "residue_number"] = chain_df["new_residue_number"]

print(len(seqres), len(atmseq), len(new_indices))
print(seqres)
print(atmseq)

# Save modified PDB
output_path = os.path.join(asep_ab_structures_dir, f"{pdb_id}_abtest.pdb")
ppdb.to_pdb(path=output_path, 
            records=["ATOM", "ANISOU"],  # Preserve important records
            gz=False,
            append_newline=True)


# %%
pdb_id = "6w4m_0P"

pdb_path = f"{asep_structures_dir}/{pdb_id}.pdb"
ppdb = PandasPdb().read_pdb(pdb_path)
atomic_df = ppdb.get_model(1).df["ATOM"]
# atomic_df = PandasPdb().read_pdb().get_model(1).df["ATOM"]

# atomic_df = PandasPdb().read_pdb(f"{asep_ab_structures_dir}/../test/structures/{pdb_id}_ab.pdb").get_model(1).df["ATOM"]
mask_data = torch.load(f"{asep_graphs_dir}/{pdb_id}.pt")

chain_data = {}
for chain_type in ["H", "L"]:
    if chain_type not in antibody_chains:
        continue

    chain_df = atomic_df[atomic_df["chain_id"] == chain_type]
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
    ppdb.df["ATOM"].loc[chain_df.index, "residue_number"] = chain_df["new_residue_number"]
    ppdb.df["ATOM"].loc[chain_df.index, "insertion"] = ""  # Clear insertion codes

    # print(len(seqres), len(atmseq), len(new_indices))
    # print(seqres)
    # print(atmseq)


# Save modified PDB
output_path = os.path.join(asep_ag_structures_dir, f"{pdb_id}_abtessst.pdb")
ppdb.to_pdb(path=output_path, 
            records=["ATOM", "ANISOU"],  # Preserve important records
            gz=False,
            append_newline=True)

# %%
new_pdb_df = ppdb.get_model(1).df["ATOM"] #["residue_number"].unique()
new_pdb_df[new_pdb_df["chain_id"] == chain_type]["residue_number"].unique()

# %%
old_indices = chain_df["residue_number"].unique()
# old_indices = residues_ordered
old_indices = np.arange(len(seqres))
offset = 1000 if chain_type == "H" else 2000  # Prevent index overlap

# Two-step remapping
temp_mapping = {old: old + offset for old in old_indices}
final_mapping = {temp: idx  for temp, idx in zip(temp_mapping.values(), 
                        [i for i, bit in enumerate(mask) if bit == 1])}

print(temp_mapping)
print(final_mapping)
new_indices = [i for i, bit in enumerate(mask) if bit == 1]
print(new_indices)
print(len(temp_mapping), len(mask), len(final_mapping), len(new_indices))

# %%
ab_seqres_atmseq_comparison_df

# %%
""" 
FIXME: 
- atmseq new indices don't have the alternate residue conformations
- that's why atmseq2cdr mapping doesn't contain those missing residues
"""
pdb_id = "7m30_2P"

# atomic_df = PandasPdb().read_pdb(f"{asep_structures_dir}/{pdb_id}.pdb").get_model(1).df["ATOM"]
atomic_df = PandasPdb().read_pdb(f"{asep_ab_structures_dir}/../test/structures/{pdb_id}_ab.pdb").get_model(1).df["ATOM"]
mask_data = torch.load(f"{asep_graphs_dir}/{pdb_id}.pt")

antibody_chains = ["H", "L"]
# Process heavy and light chains separately
chain_data = {}
for chain_type in ["H", "L"]:
    if chain_type not in antibody_chains:
        continue

    chain_df = atomic_df[atomic_df["chain_id"] == chain_type]
    if chain_df.empty:
        continue

    # Get SEQRES and ATMSEQ for the chain
    seqres = str(np.array(mask_data["seqres"]["ab"][chain_type]))

    # In process_antibody_chains():
    atmseq_df = atomic_df[atomic_df["chain_id"] == chain_type]
    atmseq_df = atmseq_df.assign(
        full_residue=atmseq_df["residue_number"].astype(str) + atmseq_df["insertion"]
    ).drop_duplicates("full_residue")
    # atmseq_df = atmseq_df.sort_values("atom_serial")
    residues_ordered = atmseq_df["full_residue"].unique()
    residue_mapping = {res: idx+1 for idx, res in enumerate(residues_ordered)}


    atmseq_df = atomic_df[atomic_df["chain_id"] == chain_type]  # NEW LINE
    atmseq_df = atmseq_df[["residue_number", "residue_name"]].drop_duplicates()
    atmseq = "".join(atmseq_df["residue_name"].map(AA_MAP))

    # Get all residue numbers in order they appear
    residue_numbers = atmseq_df["residue_number"]

    # Get the desired output
    converted_numbers = convert_residue_numbers(residue_numbers)

    print(residues_ordered)
    print(list(residue_numbers))
    print(converted_numbers)  # Output: [72, 73, 74, 76] for your example
    print(len(converted_numbers), len(residue_numbers), len(residues_ordered))

    # Generate alignment mask seqres2atmseq
    mask = get_seqres2atmseq_mask(seqres, atmseq, pdb_id)

    # print(np.array(residue_mapping))

    # print(np.array(atmseq_df["residue_number"].drop_duplicates()))
    print(seqres)
    print(atmseq)
    print(list(mask))

# %%
atmseq_df

# %%
# Function to process residue numbers
def convert_residue_numbers(residue_numbers):
    seen = set()
    unique_residues_ordered = []
    
    # First pass: Get unique residues in order, keeping alternates
    for num in residue_numbers:
        if num not in seen:
            seen.add(num)
            unique_residues_ordered.append(num)
    
    # Second pass: Convert alternates to consecutive numbers
    output_numbers = []
    last_base_num = None
    current_num = None
    
    for res in unique_residues_ordered:
        # Check if residue is an alternate (e.g., "72A")
        if isinstance(res, str) and res[-1].isalpha():
            base_num = int(res[:-1])  # Extract base number (72 from "72A")
            alt_id = res[-1]          # Extract alternate ID ("A")
        else:
            base_num = int(res)       # Normal residue (e.g., 74)
            alt_id = None
        
        # If new base number, reset current_num
        if base_num != last_base_num:
            current_num = base_num
            last_base_num = base_num
        # If same base but alternate (e.g., 72A after 72), increment
        elif alt_id is not None:
            current_num += 1
        
        output_numbers.append(current_num)
    
    return np.array(output_numbers)


# %%
# In process_antibody_chains():
atmseq_df = atomic_df[atomic_df["chain_id"] == chain_type]
atmseq_df = atmseq_df.assign(
    full_residue=atmseq_df["residue_number"].astype(str) + atmseq_df["insertion_code"]
).drop_duplicates("full_residue")
atmseq_df = atmseq_df.sort_values("atom_serial")
residues_ordered = atmseq_df["full_residue"].unique()
residue_mapping = {res: idx+1 for idx, res in enumerate(residues_ordered)}


# %%
asep_mipe_transformed = torch.load(os.path.join(mipe_asep_transform_dir, "asep_mipe_transformed.pkl" ))
asep_mipe_transformed_cdr_unindexed = torch.load(os.path.join(mipe_asep_transform_dir, "asep_mipe_transformed_cdr_unindexed.pkl" ))


# %%
index = 10
print_shapes(asep_mipe_transformed_cdr_unindexed, index)
print("*"*50)
print_shapes(asep_mipe_transformed, index)

# %%
# binary labels, but wait, where are the sequences? are AB sequences ordered as H+L or L+H??
# order doesn't matter when embeddings are generated separately for each chain in the same order as pdb data

def print_shapes(data, index):
    ##### sequence data AB, AG
    print(data[index]["complex_code"])
    print(data[index]["AbLang_AB"].shape)
    print(data[index]["ESM1b_AG"].shape)

    ##### graph data AB, AG
    print(data[index]["vertex_AB"].shape)
    print(np.array(data[index]["edge_AB"]).shape)
    print(len(data[index]["label_AB"]))
    print(data[index]["coord_AB"].shape)

    print(data[index]["vertex_AG"].shape)
    print(np.array(data[index]["edge_AG"]).shape)
    print(len(data[index]["label_AG"]))
    print(np.array(data[index]["coord_AG"]).shape)
    print(np.array(data[index]["edge_AGAB"]).shape)


# %%
print(asep_m3epi_transformed[0]["coord_AG"])
print(asep_m3epi_transformed[0]["label_AG"])
print(asep_m3epi_transformed[0]["coord_AB"])
print(asep_m3epi_transformed[0]["label_AB"])
print(asep_m3epi_transformed[0]["edge_AGAB"])
print(asep_m3epi_transformed[0]["edge_AB"])
print(asep_m3epi_transformed[0]["edge_AG"])
print(asep_m3epi_transformed[0]["vertex_AB"])
print(asep_m3epi_transformed[0]["vertex_AG"])
print(asep_m3epi_transformed[0]["AbLang_AB"])
print(asep_m3epi_transformed[0]["ESM1b_AG"])


# %%
print(asep_m3epi_transformed[0]["coord_AG"])
print(asep_m3epi_transformed[0]["label_AG"])
print(asep_m3epi_transformed[0]["coord_AB"])
print(asep_m3epi_transformed[0]["label_AB"])
print(asep_m3epi_transformed[0]["edge_AGAB"])
print(asep_m3epi_transformed[0]["edge_AB"])
print(asep_m3epi_transformed[0]["edge_AG"])
print(asep_m3epi_transformed[0]["vertex_AB"])
print(asep_m3epi_transformed[0]["vertex_AG"])
print(asep_m3epi_transformed[0]["AbLang_AB"])
print(asep_m3epi_transformed[0]["ESM1b_AG"])

# %%
print(asep_m3epi_transformed[0])

# %%

asep_m3epi_transformed = torch.load(asep_m3epi_transformed_path, map_location='cpu')


# %%
# create total.csv labels file for asep dataset
episcan_db1_fasta = SeqIO.parse(open(asep_ab_ag_sequences_fasta_path),'fasta')
ag_chains = {}
for fasta in episcan_db1_fasta:
    name = fasta.id
    pdb_id = name.split("|")[0]
    ag_chain = name.split(":")[2]
    ag_chains[pdb_id] = ag_chain
# remove 5nj6_0P.pdb from analysis
del ag_chains["5nj6_0P"]

graphbepi_asep_epi_labels = {}
for i in range(len(graphbepi_asep_atmseq2epi)):
    pdb_id = graphbepi_asep_atmseq2epi.loc[i,"pdbid"]
    # new_pdb_id = pdb_id.split("_")[0] + "_" + ag_chains[pdb_id]
    new_pdb_id = pdb_id + "_" + ag_chains[pdb_id]
    epi = str(graphbepi_asep_atmseq2epi.loc[i,"epitope"]).replace('[', '').replace(']', '').replace("'", '')
    graphbepi_asep_epi_labels[new_pdb_id] = epi

graphbepi_asep_epi_labels_df = pd.DataFrame(list(graphbepi_asep_epi_labels.items()), columns=["PDB chain", "Epitopes (resi_resn)"])
graphbepi_asep_epi_labels_df.to_csv(graphbepi_asep_transform_dir + "total.csv")

graphbepi_asep_epi_labels_df

# %%
graphbepi_asep_epi_labels_df.drop_duplicates(subset=["PDB chain"])

# %%
# rename antigen pdb files by appending ag chain
# Iterate over the ag chains dictionary and rename the files
for pdb_id, chain_id in ag_chains.items():
    # Construct the old and new file paths
    old_file_path = os.path.join(graphbepi_asep_trans_purepdb_dir, f"{pdb_id}_ag.pdb")
    new_pdb_id = pdb_id.split("_")[0]  # Extract the base PDB ID
    
    new_file_path = os.path.join(graphbepi_asep_trans_purepdb_dir, f"{pdb_id}_{chain_id}.pdb")

    # Rename the file
    os.rename(old_file_path, new_file_path)

# %%
"""
sequence (list): List of amino acid symbols in the chain
amino (list): List of amino acid IDs corresponding to the sequence
coord (list): List of 3D coordinates of alpha carbon atom for each amino acid
site (dict): Mapping of residue positions to sequence indices
date (str): Date information from the PDB file
length (int): Length of the amino acid sequence
adj (torch.Tensor): Adjacency matrix for the chain's graph representation
edge (torch.Tensor): Edge features for the chain's graph representation
feat (torch.Tensor): Feature tensor extracted from the sequence
dssp (torch.Tensor): DSSP features containing secondary structure information
name (str): Unique identifier for the chain (protein_chain format)
chain_name (str): Chain identifier within the protein
protein_name (str): Name of the protein
rsa (torch.Tensor): Relative solvent accessibility information
label (torch.Tensor): Binary labels for epitope residues
"""

# %%
def load_pdb(pdb_file_path):
    atomic_df = PandasPdb().read_pdb(pdb_file_path)
    atomic_df = atomic_df.get_model(1)
    # pd.concat([atomic_df.df["ATOM"], atomic_df.df["HETATM"]]) #, header
    # select the alpha carbon atoms only
    atomic_df = atomic_df.df["ATOM"][atomic_df.df["ATOM"].loc[:,"atom_name"]=="CA"]
    chain_ids = [chain.get_id() for chain in pdb_model]
    atomic_df = atomic_df[atomic_df.loc[:,"chain_id"]==chain_ids[2]]
    return atomic_df
# map with the antigen sequence with surface residues and select those residues only
# transform the structure using mmseq from asep data processing code
# load_pdb(pdb_file_path)

# %%
antigen_pdb = Path(episcan_asep_transform_dir + f"{pdb_file_path.stem}_antigen.pdb")
antigen_pdb_dssp = Path(episcan_asep_transform_dir + f"{pdb_file_path.stem}_antigen_dssp.pdb")
antibody_pdb = Path(episcan_asep_transform_dir + f"{pdb_file_path.stem}_antibody.pdb")

load_pdb(antigen_pdb)

# %% [markdown]
# # random tests

# %%
sequence_examples = "PRTEINO"
sequence_examples = [sequence_examples]
# replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

sequence_examples[0]

# %% [markdown]
# ## results analysis

# %%
mipe_test_results

# %%
# Calculate mean and std for each metric
metrics = ["ag_auprc", "ag_auroc", "ag_precision", "ag_recall", "ag_f1", "ag_bacc", "ag_mcc"]
results = {}

for metric in metrics:
    mean_value = round(np.mean(mipe_test_results.loc[11:13, metric]), 4)
    std_value = round(np.std(mipe_test_results.loc[11:13, metric]), 4)
    results[metric] = f"{mean_value} (±{std_value})"

results_df = pd.DataFrame([results])
results_df


