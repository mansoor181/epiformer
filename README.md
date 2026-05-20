# EpiFormer: Antibody-Aware Epitope Prediction

This repository contains the implementation of EpiFormer, a GNN-based model for epitope and paratope prediction on antibody-antigen complexes.

## Requirements

- Python 3.10
- PyTorch 2.2+ with CUDA 12.1 (for GPU training)
- PyTorch Geometric 2.5+

## Quick Start

### Step 1: Environment Setup

Create a conda environment and install dependencies:

```bash
conda create -n epiformer python=3.10 -y
conda activate epiformer

# Install PyTorch (GPU)
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.2.1+cu121.html
pip install torch-geometric==2.5.1

# Install other dependencies
pip install -r requirements.txt
```

For CPU-only installation:

```bash
pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cpu
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.2.1+cpu.html
pip install torch-geometric==2.5.1
```

### Step 2: Data Preprocessing

**Option A: Use pre-processed data**

Place the pre-processed files in:
```
data/asep/m3epi/res_graph_tensor_esm2_650m.pkl
data/asep/split/split_dict_corrected.pt
```

**Option B: Preprocess from scratch**

1. Download the AsEP dataset from https://zenodo.org/records/11495514

2. Run preprocessing scripts (6 steps):
```bash
cd code/data

# Step 2: Reindex PDB files
python reindex_ag_split_complex.py
python reindex_ab_split_complex.py

# Step 3: Create surface/CDR mappings
python seqres2surf_mapping.py
python seqres2cdr_mapping.py

# Step 4: Create corrected splits
python create_corrected_splits.py

# Step 5: Generate PLM embeddings
python embed_esm2.py
python embed_antiberty.py

# Step 6: Build graph tensors
python construct_res_graphs_tensor.py
```

See [code/data/README.md](code/data/README.md) for detailed preprocessing documentation.

### Step 3: Training

Train the model using the best hyperparameters.

**For epitope-group split:**
```bash
cd code
./scripts/best_glamorous_sweep_epi_group.sh \
    --gpu_id 0 \
    --batch_size 8 \
    --epochs 130 \
    --server local
```

**For epitope-ratio split:**
```bash
./scripts/best_playful_sweep_epi_ratio.sh \
    --gpu_id 0 \
    --batch_size 8 \
    --epochs 130 \
    --server local
```

**With Weights & Biases logging:**
```bash
./scripts/best_glamorous_sweep_epi_group.sh \
    --gpu_id 0 --batch_size 8 --epochs 130 --server local --wandb
```

### Step 4: Inference

#### Option A: Evaluate on AsEP Dataset

Evaluate a trained model checkpoint on both dataset splits.

**First, place checkpoint files:**
```
checkpoints/
└── best-glamorous-sweep-37/
    └── epiformer_best.pt
```

**Run evaluation:**
```bash
cd code
python evaluate.py \
    --checkpoint ../checkpoints/best-glamorous-sweep-37/epiformer_best.pt \
    --data_dir ../data/asep \
    --gpu_id 0
```

This evaluates on both `epitope_ratio` and `epitope_group` splits and reports:
- AUROC, AUPRC, F1, MCC, Precision, Recall

**Custom threshold:**
```bash
python evaluate.py --checkpoint path/to/checkpoint.pt --threshold 0.5
```

#### Option B: Predict on New PDB Files

Run end-to-end epitope prediction on new antigen-antibody PDB files. This requires only the checkpoint file - all preprocessing is performed on-the-fly.

**Important:** The model was trained on:
- **Antigen**: Surface residues only (detected via SASA)
- **Antibody**: CDR residues only (H1/H2/H3/L1/L2/L3 loops)

**Additional requirements for inference:**
```bash
pip install antiberty    # Required: Antibody embeddings
pip install anarci       # Optional: Better CDR detection (falls back to Chothia numbering)
```

**Basic usage (automatic filtering):**
```bash
cd code
python inference.py \
    --antigen_pdb path/to/antigen.pdb \
    --antibody_pdb path/to/antibody.pdb \
    --checkpoint ../checkpoints/best-glamorous-sweep-37/epiformer_best.pt
```

This will:
1. Filter antigen to surface residues (SASA > 5.0 A^2)
2. Identify antibody CDR regions via ANARCI/Chothia numbering
3. Generate ESM-2 embeddings for filtered antigen sequence
4. Generate AntiBERTy embeddings for filtered antibody sequence
5. Build residue graphs and run prediction

**With pre-filtered PDBs:**
```bash
# If your PDBs already contain only surface/CDR residues
python inference.py \
    --antigen_pdb antigen_surface.pdb \
    --antibody_pdb antibody_cdr.pdb \
    --checkpoint checkpoint.pt \
    --skip_filtering
```

**Additional options:**
```bash
python inference.py \
    --antigen_pdb ag.pdb \
    --antibody_pdb ab.pdb \
    --checkpoint checkpoint.pt \
    --threshold 0.5 \           # Classification threshold (default: 0.3)
    --sasa_threshold 10.0 \     # SASA cutoff for surface detection
    --output predictions.json \ # Save results to JSON
    --output_pdb ag_epitope.pdb # Save labeled PDB for visualization
```

**Output:**
- Console: Summary of predicted epitope residues with probabilities and PyMOL selection command
- JSON file (optional): Full prediction results including per-residue details
- Labeled PDB (optional): Original antigen with epitope probabilities in B-factor column

**Visualization in PyMOL:**
```
# Load the labeled PDB
load ag_epitope.pdb

# Color by epitope probability (blue=low, red=high)
spectrum b, blue_white_red, minimum=0, maximum=100

# Or use the selection command from console output
select epitope, (chain N and resi 249) or (chain N and resi 250) ...
color red, epitope
```

## Directory Structure

```
epiformer/
├── README.md
├── requirements.txt
├── code/
│   ├── trainer.py           # Main training script
│   ├── evaluate.py          # Evaluation on AsEP dataset
│   ├── inference.py         # End-to-end prediction on new PDBs
│   ├── utils.py
│   ├── conf/                # Hydra configuration
│   ├── model/               # Model implementation
│   │   ├── epiformer.py
│   │   ├── encoder.py
│   │   ├── decoder.py
│   │   └── ...
│   ├── data/                # Preprocessing scripts
│   │   ├── README.md        # Preprocessing documentation
│   │   ├── construct_res_graphs_tensor.py
│   │   └── ...
│   └── scripts/
│       ├── best_glamorous_sweep_epi_group.sh
│       └── best_playful_sweep_epi_ratio.sh
├── walle/                   # AsEP data loading utilities
├── checkpoints/             # Model checkpoints
│   └── best-glamorous-sweep-37/
│       └── epiformer_best.pt
└── data/                    # Data directory
    └── asep/
        ├── m3epi/res_graph_tensor_esm2_650m.pkl
        └── split/split_dict_corrected.pt
```

## Configuration

Override config parameters from command line:
```bash
python trainer.py \
    model.epiformer.residue_layers=6 \
    hparams.train.learning_rate=1e-4 \
    dataset.split.method="epitope_ratio"
```

## Training Modes

- `val`: Single train/val/test split (default)
- `train`: K-fold cross-validation
- `test`: Train on train+val, evaluate on test

## Metrics

- AUROC 
- AUPRC
- F1 Score
- MCC
- Precision
- Recall

## Citation

```bibtex
@inproceedings{epiformer2026,
  title={EpiFormer: Antibody-Aware Epitope Prediction with Interleaved Cross-Attention},
  author={Mansoor Ahmed, Huirong Chai, Haoxin Wang, Hemanth Venkateswara, Murray Patterson},
  booktitle={arXiv},
  year={2026}
}
```

## License

MIT License
