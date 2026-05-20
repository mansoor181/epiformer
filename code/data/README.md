# Data Preprocessing

This directory contains scripts for preprocessing the AsEP dataset for EpiFormer training.

## AsEP Dataset

The AsEP (Antibody-Specific Epitope Prediction) dataset contains 1,723 antibody-antigen complexes. We use 1,721 complexes after excluding two problematic entries (5nj6_0P and 5ies_0P).

**Dataset Source:**
- Zenodo: https://zenodo.org/records/11495514
- GitHub: https://github.com/biochunan/AsEP-dataset

## Quick Start (Using Pre-processed Data)

If you have access to pre-processed data, you only need these two files:

1. `data/asep/m3epi/res_graph_tensor_esm2_650m.pkl` - Graph tensors (1,721 HeteroData objects)
2. `data/asep/split/split_dict_corrected.pt` - Train/val/test splits

Place these files in the appropriate directories and skip to training.

## Full Preprocessing Pipeline

The preprocessing consists of 6 steps:

```
Step 1: Download AsEP dataset
    ↓
Step 2: Reindex PDB files (align SEQRES to ATMSEQ)
    ↓
Step 3: Create surface/CDR mappings and filtered PDBs
    ↓
Step 4: Create corrected splits (exclude problematic complexes)
    ↓
Step 5: Generate PLM embeddings (ESM-2, AntiBERTy)
    ↓
Step 6: Construct graph tensors (HeteroData objects)
```

### Prerequisites

```bash
pip install biopython biopandas torch torch-geometric torch-scatter esm antiberty
```

Additional requirements:
- ClustalOmega (for sequence alignment in Steps 2-3)
- GPU with 24GB+ VRAM (for ESM-2 3B embeddings, optional)

### Working Directory

All scripts should be run from `code/data/` directory:
```bash
cd /path/to/icml_26/code/data
```

The scripts expect data at `../../data/asep/` (i.e., `icml_26/data/asep/`).
If your data is in a different location, adjust the `proj_dir` variable in each script.

---

### Step 1: Download and Preprocess AsEP Dataset

**1a. Download from Zenodo:**

```bash
# Download asep-dataset.zip from https://zenodo.org/records/11495514
# Extract to data/asep/
```

The Zenodo download (`asep-dataset.zip`) includes:
- `structures2/` - Full PDB complex structures (1,723 files)
- `sequences/asep_ab_ag_seqres_1723.fasta` - FASTA sequences (format: H_chain:L_chain:AG_chain)
- `split/split_dict.pt` - Original train/val/test splits
- `asepv1-AbDb-IDs.txt` - List of 1,723 PDB IDs
- `asepv1_interim_graphs/` - Per-complex graph files with pre-computed embeddings (1,723 .pt files)

**1b. Generate dict_pre_cal.pt (WALLE format):**

The `processed/dict_pre_cal.pt` file converts the individual interim graph files into a single dictionary format. This file is required by `construct_res_graphs_tensor.py` and baseline preprocessing scripts.

**Option A: Generate from interim graphs (Recommended)**

The Zenodo download includes `asepv1_interim_graphs/` directory with pre-computed embeddings (IgFold + ESM-2) for each complex. Use the provided script to convert them to dictionary format:

```bash
cd code/data

# Generate dict_pre_cal.pt with default embeddings (IgFold for AB, ESM-2 for AG)
python create_dict_pre_cal.py \
    --interim_dir ../../data/asep/asepv1_interim_graphs \
    --output_dir ../../data/asep/processed

# Or with ESM-2 for both antibody and antigen (if available in interim graphs)
python create_dict_pre_cal.py \
    --interim_dir ../../data/asep/asepv1_interim_graphs \
    --output_dir ../../data/asep/processed \
    --ab_embedding esm2 \
    --ag_embedding esm2 \
    --output_suffix "_esm2_esm2"
```

**Option B: Use AsEP repository**

Alternatively, use the official AsEP repository preprocessing:

```bash
git clone https://github.com/biochunan/AsEP-dataset.git
cd AsEP-dataset
pip install -e .
# Follow AsEP documentation for preprocessing
```

**Output format:**

The generated file is a PyTorch dictionary mapping PDB IDs to preprocessed data:
```python
{
    'pdb_id': {
        'x_g': [n_ag, emb_dim],   # Antigen embeddings (surface residues only)
        'x_b': [n_ab, emb_dim],   # Antibody embeddings (CDR residues only)
        'edge_index_g': [2, E],   # Antigen intra-chain edges
        'edge_index_b': [2, E],   # Antibody intra-chain edges
        'edge_index_bg': [2, E],  # Cross-chain bipartite edges
        'y_g': [n_ag],            # Epitope labels (0/1)
        'y_b': [n_ab],            # Paratope labels (0/1)
    },
    ...
}
```

**Embedding dimensions:**
- IgFold (antibody): 512D
- ESM-2 35M (antigen): 480D
- ESM-2 650M: 1280D

**Note:** The `walle/` directory in this repository contains reference code from the AsEP codebase needed for data loading.

---

### Step 2: Reindex PDB Files

Split the complex PDB files and reindex residues to align SEQRES with ATMSEQ sequences.

**Antigen chains:**
```bash
cd code/data
python reindex_ag_split_complex.py \
    ../../data/asep/structures2 \
    ../../data/asep/asepv1_interim_graphs \
    ../../data/asep/antigen/structures \
    ../../data/asep/antigen
```

Arguments:
- `input_dir`: Directory with raw PDB complexes (`structures2/`)
- `pt_graphs_dir`: Directory with interim graph files (`asepv1_interim_graphs/`)
- `output_dir`: Output directory for reindexed antigen PDBs
- `metadata_dir`: Output directory for alignment metadata CSV

**Output:**
- `data/asep/antigen/structures/` - Reindexed antigen PDB files (`{pdb_id}_ag.pdb`)
- `data/asep/antigen/seqres2atmseq_mask_ag.csv` - Alignment metadata

**Antibody chains:**
```bash
python reindex_ab_split_complex.py \
    ../../data/asep/structures2 \
    ../../data/asep/asepv1_interim_graphs \
    ../../data/asep/antibody/structures \
    ../../data/asep/antibody
```

**Output:**
- `data/asep/antibody/structures/` - Reindexed antibody PDB files (`{pdb_id}_ab.pdb`)
- `data/asep/antibody/seqres2atmseq_mask_ab_HL_chain.csv` - Alignment metadata

---

### Step 3: Create Surface/CDR Mappings

Create filtered PDB files containing only surface (antigen) or CDR (antibody) residues.

**Prerequisites:** Step 1b (dict_pre_cal.pt) must be completed first for CDR mapping.

**Antigen surface mapping:**
```bash
python seqres2surf_mapping.py \
    ../../data/asep/antigen/structures \
    ../../data/asep/asepv1_interim_graphs \
    ../../data/asep/antigen/atmseq2surf \
    ../../data/asep/antigen/sequences
```

Arguments:
- `ag_pdb_dir`: Reindexed antigen PDB directory (from Step 2)
- `masks_graph_pt_dir`: Interim graph directory with mappings
- `ag_surf_pdb_out_dir`: Output directory for surface-filtered PDBs
- `ag_sequences_out_dir`: Output directory for sequence/epitope data

**Output:**
- `data/asep/antigen/atmseq2surf/` - Surface-filtered PDB files (`{pdb_id}_surf.pdb`)

**Antibody CDR mapping:**
```bash
python seqres2cdr_mapping.py \
    ../../data/asep/antibody/structures \
    ../../data/asep/asepv1_interim_graphs \
    ../../data/asep/processed \
    ../../data/asep/antibody/atmseq2cdr \
    ../../data/asep/antibody/sequences
```

Arguments:
- `ab_pdb_dir`: Reindexed antibody PDB directory (from Step 2)
- `masks_graph_pt_dir`: Interim graph directory with mappings
- `processed_graphs_dir`: Directory containing `dict_pre_cal.pt` (from Step 1b)
- `ab_cdr_pdb_out_dir`: Output directory for CDR-filtered PDBs
- `ab_sequences_out_dir`: Output directory for sequence/paratope data

**Output:**
- `data/asep/antibody/atmseq2cdr/` - CDR-filtered PDB files (`{pdb_id}_cdr.pdb`)

**Note:** Two complexes are excluded at this step:
- `5nj6_0P` - Alignment error between SEQRES and ATMSEQ
- `5ies_0P` - CDR length mismatch

---

### Step 4: Create Corrected Splits

Create split files that exclude the two problematic complexes:

```bash
python create_corrected_splits.py
```

**Input:** `data/asep/split/split_dict.pt`
**Output:** `data/asep/split/split_dict_corrected.pt`

This creates corrected indices for both split methods:
- `epitope_ratio` - Split based on epitope ratio distribution
- `epitope_group` - Split based on epitope clustering

---

### Step 5: Generate PLM Embeddings

**Antigen embeddings (ESM-2 650M and 3B):**

```bash
python embed_esm2.py
```

**Input:**
- `data/asep/sequences/asep_ab_ag_seqres_1723.fasta`
- `data/asep/asepv1_interim_graphs/*.pt` (for surface mappings)

**Output:** `data/asep/antigen/plm_embeddings/ag_esm2_embeddings_asep.pt`

The output contains a dictionary with keys `esm2_650m` and `esm2_3b`, each mapping PDB IDs to numpy arrays of shape `[n_surface_residues, embedding_dim]`.

**Antibody embeddings (AntiBERTy):**

```bash
python embed_antiberty.py
```

**Input:**
- `data/asep/sequences/asep_ab_ag_seqres_1723.fasta`
- `data/asep/asepv1_interim_graphs/*.pt` (for CDR mappings)

**Output:** `data/asep/antibody/antiberty_embeddings/asep_antiberty_embeddings.pt`

---

### Step 6: Construct Graph Tensors

Build the final HeteroData graph tensors:

```bash
python construct_res_graphs_tensor.py
```

**Input:**
- `data/asep/processed/dict_pre_cal.pt` - WALLE format data (from Step 1b)
- `data/asep/antigen/atmseq2surf/*.pdb` - Antigen surface PDB files (from Step 3)
- `data/asep/antibody/atmseq2cdr/*.pdb` - Antibody CDR PDB files (from Step 3)
- `data/asep/antigen/plm_embeddings/ag_esm2_embeddings_asep.pt` - ESM-2 embeddings (from Step 5)
- `data/asep/antibody/antiberty_embeddings/asep_antiberty_embeddings.pt` - AntiBERTy embeddings (from Step 5)

**Note:** This script requires the `walle/` directory to be in the Python path for loading AsEP data structures.

**Output:** `data/asep/m3epi/res_graph_tensor_esm2_650m.pkl`

The output contains 1,721 HeteroData objects with the following structure:

```python
HeteroData(
    complex_id='1s78_0P',

    # Antigen residue-level data
    ag_res={
        x=[n_ag, 105],      # RAAD node features
        plm=[n_ag, 1280],   # ESM-2 embeddings
        pos=[n_ag, 3],      # CA coordinates
        y=[n_ag],           # Epitope labels (0/1)
    },

    # Antibody residue-level data
    ab_res={
        x=[n_ab, 105],      # RAAD node features
        plm=[n_ab, 512],    # AntiBERTy embeddings
        pos=[n_ab, 3],      # CA coordinates
        y=[n_ab],           # Paratope labels (0/1)
    },

    # Multi-relational edges (4 relation types per chain)
    (ag_res, r0, ag_res)={edge_index, edge_attr},  # Sequential +-1 edges
    (ag_res, r1, ag_res)={edge_index, edge_attr},  # Sequential +-2 edges
    (ag_res, r2, ag_res)={edge_index, edge_attr},  # k-NN edges (k=10)
    (ag_res, r3, ag_res)={edge_index, edge_attr},  # Spatial edges (<8A)
    # ... same for ab_res

    # Cross-chain interactions
    (ag_res, interacts, ab_res)={edge_index},
)
```

---

## Directory Structure

After preprocessing, the data directory should look like (relative to `icml_26/`):

```
icml_26/
├── code/
│   └── data/                           # Preprocessing scripts (run from here)
└── data/
    └── asep/
        ├── structures2/                        # Raw PDB complexes (from Zenodo)
        │   ├── 1s78_0P.pdb
        │   └── ...
        ├── sequences/
        │   └── asep_ab_ag_seqres_1723.fasta    # FASTA sequences (from Zenodo)
        ├── processed/
        │   └── dict_pre_cal.pt                 # WALLE format data (generated Step 1b)
        ├── asepv1_interim_graphs/              # Per-complex graph files (from Zenodo)
        │   ├── 1s78_0P.pt
        │   └── ...
        ├── antigen/
        │   ├── structures/                     # Reindexed antigen PDBs (Step 2)
        │   ├── atmseq2surf/                    # Surface-filtered PDBs (Step 3)
        │   │   ├── 1s78_0P_surf.pdb
        │   │   └── ...
        │   └── plm_embeddings/
        │       └── ag_esm2_embeddings_asep.pt  # ESM-2 embeddings (Step 5)
        ├── antibody/
        │   ├── structures/                     # Reindexed antibody PDBs (Step 2)
        │   ├── atmseq2cdr/                     # CDR-filtered PDBs (Step 3)
        │   │   ├── 1s78_0P_cdr.pdb
        │   │   └── ...
        │   └── antiberty_embeddings/
        │       └── asep_antiberty_embeddings.pt  # AntiBERTy embeddings (Step 5)
        ├── split/
        │   ├── split_dict.pt                   # Original splits (from Zenodo)
        │   └── split_dict_corrected.pt         # Corrected splits (Step 4)
        ├── m3epi/
        │   └── res_graph_tensor_esm2_650m.pkl  # Final graph tensors (Step 6)
        └── asepv1-AbDb-IDs.txt                 # PDB ID list (from Zenodo)
```

**Legend:**
- **(from Zenodo)** - Included in `asep-dataset.zip` download
- **(generated Step N)** - Generated by running the corresponding preprocessing script

---

## Feature Descriptions

### Node Features (105D RAAD features)

| Feature | Dimensions | Description |
|---------|------------|-------------|
| Residue type | 20 | One-hot encoding of amino acid type |
| Positional encoding | 16 | Sinusoidal positional encoding |
| Bond/dihedral angles | 12 | 6 angles (phi, psi, omega, alpha, beta, gamma) x sin/cos |
| RBF distances | 48 | Distances to C, N, O atoms (3 x 16 RBF) |
| Local coordinate frame | 9 | 3x3 rotation matrix flattened |

### Edge Features (100D)

| Feature | Dimensions | Description |
|---------|------------|-------------|
| Edge type | 4 | One-hot encoding of relation type |
| Relative position | 16 | Sinusoidal positional encoding |
| RBF distances | 64 | Distances to 4 backbone atoms (4 x 16 RBF) |
| Direction vectors | 12 | Local frame directions (4 x 3D) |
| Quaternion | 4 | Relative rotation between residues |

### PLM Embedding Dimensions

| Model | Dimensions | Chain |
|-------|------------|-------|
| ESM-2 650M | 1280 | Antigen |
| ESM-2 3B | 2560 | Antigen |
| AntiBERTy | 512 | Antibody |

---

## Script Reference

| Script | Step | Purpose |
|--------|------|---------|
| `create_dict_pre_cal.py` | 1b | Convert interim graphs to dict_pre_cal.pt format |
| `reindex_ag_split_complex.py` | 2 | Reindex antigen residues (SEQRES-ATMSEQ alignment) |
| `reindex_ab_split_complex.py` | 2 | Reindex antibody residues for H and L chains |
| `seqres2surf_mapping.py` | 3 | Create antigen surface mappings and filtered PDBs |
| `seqres2cdr_mapping.py` | 3 | Create antibody CDR mappings and filtered PDBs |
| `create_corrected_splits.py` | 4 | Create split files excluding problematic complexes |
| `embed_esm2.py` | 5 | Generate ESM-2 embeddings (650M, 3B) for antigens |
| `embed_antiberty.py` | 5 | Generate AntiBERTy embeddings for antibodies |
| `construct_res_graphs_tensor.py` | 6 | Build final HeteroData graph tensors |
| `data_splits.py` | - | Dataset splitting utilities (used during training) |

---

## Excluded Complexes

Two complexes are excluded during preprocessing:

1. **5nj6_0P**: Alignment error between SEQRES and ATMSEQ
2. **5ies_0P**: CDR length mismatch between seqres2cdr_seq and atmseq2cdr

Final dataset: 1,721 complexes (1,723 - 2 = 1,721)

---
