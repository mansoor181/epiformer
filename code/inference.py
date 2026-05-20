#!/usr/bin/env python3
"""
EpiFormer Inference Script

Performs end-to-end epitope prediction on new antibody-antigen PDB files.
Requires only a trained checkpoint - all preprocessing is done on-the-fly.

The model expects:
- Antigen: Surface residues only (computed via SASA)
- Antibody: CDR residues only (identified via ANARCI numbering)

Usage:
    python inference.py \
        --antigen_pdb path/to/antigen.pdb \
        --antibody_pdb path/to/antibody.pdb \
        --checkpoint ../checkpoints/best-glamorous-sweep-37/epiformer_best.pt

    # Skip filtering if PDBs are already pre-filtered
    python inference.py \
        --antigen_pdb antigen_surface.pdb \
        --antibody_pdb antibody_cdr.pdb \
        --checkpoint checkpoint.pt \
        --skip_filtering
"""

import os
import sys
import json
import math
import shutil
import argparse
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData, Batch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.SASA import ShrakeRupley

import esm
from antiberty import AntiBERTyRunner

# Optional: ANARCI for antibody numbering
try:
    from anarci import anarci
    ANARCI_AVAILABLE = True
except ImportError:
    ANARCI_AVAILABLE = False

warnings.filterwarnings("ignore")

# Add code directory to path
CODE_DIR = Path(__file__).parent
sys.path.insert(0, str(CODE_DIR))

from model.epiformer import EpiformerModel


# ==================== AMINO ACID MAPPING ====================

AA_MAP = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G",
    "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
    "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V",
    "TRP": "W", "TYR": "Y"
}

AA_TYPES = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
            'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
            'THR', 'TRP', 'TYR', 'VAL']

# CDR definitions (IMGT numbering)
CDR_IMGT = {
    'H': {
        'CDR1': (27, 38),
        'CDR2': (56, 65),
        'CDR3': (105, 117),
    },
    'L': {
        'CDR1': (27, 38),
        'CDR2': (56, 65),
        'CDR3': (105, 117),
    }
}

# CDR definitions (Chothia numbering) - more commonly used
CDR_CHOTHIA = {
    'H': {
        'CDR1': (26, 32),
        'CDR2': (52, 56),
        'CDR3': (95, 102),
    },
    'L': {
        'CDR1': (24, 34),
        'CDR2': (50, 56),
        'CDR3': (89, 97),
    }
}


# ==================== SURFACE RESIDUE DETECTION ====================

def compute_surface_residues(pdb_path: str, sasa_threshold: float = 5.0) -> List[int]:
    """
    Compute surface residues based on Solvent Accessible Surface Area (SASA).

    Args:
        pdb_path: Path to PDB file
        sasa_threshold: Minimum SASA (in Angstrom^2) to be considered surface

    Returns:
        List of residue IDs that are on the surface
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)

    # Compute SASA using Shrake-Rupley algorithm
    sr = ShrakeRupley()
    sr.compute(structure, level="R")  # Residue level

    surface_residues = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != ' ':  # Skip heteroatoms
                    continue
                if hasattr(residue, 'sasa') and residue.sasa > sasa_threshold:
                    surface_residues.append(residue.id[1])

    return surface_residues


def filter_pdb_by_residues(input_pdb: str, output_pdb: str, residue_ids: List[int]):
    """Filter PDB to keep only specified residue IDs."""

    class ResidueSelect(Select):
        def __init__(self, residue_ids):
            self.residue_ids = set(residue_ids)

        def accept_residue(self, residue):
            return residue.id[1] in self.residue_ids

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", input_pdb)

    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb, ResidueSelect(residue_ids))


# ==================== CDR IDENTIFICATION ====================

def identify_cdr_residues_anarci(pdb_path: str) -> Tuple[List[int], str]:
    """
    Identify CDR residues using ANARCI antibody numbering.

    Returns:
        Tuple of (list of CDR residue IDs, full sequence)
    """
    if not ANARCI_AVAILABLE:
        print("Warning: ANARCI not installed. Install with: pip install anarci")
        print("Falling back to Chothia numbering based on residue IDs.")
        return identify_cdr_residues_chothia(pdb_path)

    # Extract sequence from PDB
    sequence = extract_sequence_from_pdb(pdb_path)
    residues = extract_residues_from_pdb(pdb_path)

    # Get chain info
    chains = {}
    for res in residues:
        chain_id = res['chain_id']
        if chain_id not in chains:
            chains[chain_id] = []
        chains[chain_id].append(res)

    cdr_residue_ids = []

    for chain_id, chain_residues in chains.items():
        chain_seq = "".join(AA_MAP.get(r['resname'], 'X') for r in chain_residues)

        # Run ANARCI numbering
        results = anarci([("chain", chain_seq)], scheme="imgt")

        if results[0] is None:
            print(f"Warning: ANARCI failed for chain {chain_id}, using Chothia fallback")
            continue

        numbering = results[0][0][0]

        # Map ANARCI positions to original residue IDs
        for i, (pos, aa) in enumerate(numbering):
            if aa == '-':
                continue

            imgt_num = pos[0]
            chain_type = 'H' if chain_id == 'H' else 'L'

            # Check if in CDR region
            for cdr_name, (start, end) in CDR_IMGT[chain_type].items():
                if start <= imgt_num <= end:
                    if i < len(chain_residues):
                        cdr_residue_ids.append(chain_residues[i]['res_id'])
                    break

    return cdr_residue_ids, sequence


def identify_cdr_residues_chothia(pdb_path: str) -> Tuple[List[int], str]:
    """
    Identify CDR residues using Chothia numbering scheme.
    Falls back to this if ANARCI is not available.

    Note: This assumes standard Chothia numbering in the PDB file.
    """
    residues = extract_residues_from_pdb(pdb_path)
    sequence = "".join(AA_MAP.get(r['resname'], 'X') for r in residues)

    cdr_residue_ids = []

    for res in residues:
        chain_id = res['chain_id']
        res_id = res['res_id']

        # Determine chain type
        chain_type = 'H' if chain_id in ['H', 'A'] else 'L'

        # Check if residue is in any CDR region
        for cdr_name, (start, end) in CDR_CHOTHIA[chain_type].items():
            if start <= res_id <= end:
                cdr_residue_ids.append(res_id)
                break

    if not cdr_residue_ids:
        print("Warning: No CDR residues identified with Chothia numbering.")
        print("Your antibody PDB may use non-standard numbering.")
        print("Consider providing pre-filtered CDR PDB with --skip_filtering")

    return cdr_residue_ids, sequence


# ==================== PLM EMBEDDING GENERATION ====================

def load_esm2_model(device):
    """Load ESM-2 650M model for antigen embeddings."""
    print("Loading ESM-2 650M model...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    model.to(device)
    return model, batch_converter


def load_antiberty_model():
    """Load AntiBERTy model for antibody embeddings."""
    print("Loading AntiBERTy model...")
    antiberty = AntiBERTyRunner()
    return antiberty


def embed_esm2(model, batch_converter, sequence: str, device) -> torch.Tensor:
    """
    Generate ESM-2 650M embeddings for a protein sequence.
    Returns tensor of shape (L, 1280).
    """
    data = [("protein", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33])
        embeddings = results["representations"][33]
        embeddings = embeddings[0, 1:-1, :].cpu()
    return embeddings


def embed_antiberty(antiberty, sequence: str) -> torch.Tensor:
    """
    Generate AntiBERTy embeddings for an antibody sequence.
    Returns tensor of shape (L, 512).
    """
    sequences = [sequence]
    embeddings = antiberty.embed(sequences)
    residue_embeddings = embeddings[0][1:-1, :]
    return torch.tensor(residue_embeddings, dtype=torch.float)


# ==================== PDB PROCESSING ====================

def extract_residues_from_pdb(pdb_path: str) -> List[Dict]:
    """Extract residue information from a PDB file."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pdb", pdb_path)
    residues = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != ' ':
                    continue

                if 'CA' not in residue:
                    continue

                ca_coord = residue['CA'].get_coord()

                residues.append({
                    'resname': residue.resname,
                    'chain_id': chain.id,
                    'res_id': residue.id[1],
                    'ca_coord': ca_coord,
                    'residue_obj': residue
                })

    return residues


def extract_sequence_from_pdb(pdb_path: str) -> str:
    """Extract amino acid sequence from PDB file."""
    residues = extract_residues_from_pdb(pdb_path)
    sequence = ""
    for res in residues:
        aa = AA_MAP.get(res['resname'], 'X')
        sequence += aa
    return sequence


# ==================== RESIDUE GRAPH CONSTRUCTION ====================

class ResidueGraphBuilder:
    """Builds multi-relational residue graphs from PDB structures with RAAD features."""

    def __init__(self, k_nn=10, spatial_cutoff=8.0):
        self.k_nn = k_nn
        self.spatial_cutoff = spatial_cutoff
        self.parser = PDBParser(QUIET=True)
        self.rbf_centers = torch.linspace(0, 20, 16)
        self.rbf_width = 1.0

    def compute_rbf(self, distances):
        """Compute radial basis function encoding."""
        if isinstance(distances, (int, float)):
            distances = torch.tensor([distances], dtype=torch.float)
        elif not isinstance(distances, torch.Tensor):
            distances = torch.tensor(distances, dtype=torch.float)

        if distances.dim() == 0:
            distances = distances.unsqueeze(0)

        rbf = torch.exp(-0.5 * ((distances.unsqueeze(-1) - self.rbf_centers) / self.rbf_width) ** 2)
        return rbf.squeeze(0) if rbf.shape[0] == 1 else rbf

    def compute_positional_encoding(self, position, max_len=1000):
        """Compute sinusoidal positional encoding."""
        pe = torch.zeros(16)
        position = torch.tensor([position], dtype=torch.float)

        div_term = torch.exp(torch.arange(0, 16, 2).float() *
                           -(math.log(10000.0) / 16))

        pe[0::2] = torch.sin(position * div_term)
        pe[1::2] = torch.cos(position * div_term)
        return pe

    def compute_local_coordinate_frame(self, ca_coord, c_coord, n_coord):
        """Compute local coordinate frame Q_i from CA, C, N atoms."""
        ca = torch.tensor(ca_coord, dtype=torch.float)
        c = torch.tensor(c_coord, dtype=torch.float)
        n = torch.tensor(n_coord, dtype=torch.float)

        v1 = c - ca
        v2 = n - ca

        u1 = F.normalize(v1, dim=0)
        u2_temp = v2 - torch.dot(v2, u1) * u1
        u2 = F.normalize(u2_temp, dim=0)
        u3 = torch.cross(u1, u2)

        Q = torch.stack([u1, u2, u3], dim=1)
        return Q

    def compute_dihedral_angles(self, coord_dict, prev_coord=None, next_coord=None):
        """Compute 6 backbone angles."""
        angles = torch.zeros(6)

        if 'N' not in coord_dict or 'CA' not in coord_dict or 'C' not in coord_dict:
            return angles

        n_curr = torch.tensor(coord_dict['N'], dtype=torch.float)
        ca_curr = torch.tensor(coord_dict['CA'], dtype=torch.float)
        c_curr = torch.tensor(coord_dict['C'], dtype=torch.float)

        if prev_coord is not None and 'C' in prev_coord:
            c_prev = torch.tensor(prev_coord['C'], dtype=torch.float)
            angles[0] = self._compute_dihedral(c_prev, n_curr, ca_curr, c_curr)

        if next_coord is not None and 'N' in next_coord:
            n_next = torch.tensor(next_coord['N'], dtype=torch.float)
            angles[1] = self._compute_dihedral(n_curr, ca_curr, c_curr, n_next)

        if prev_coord is not None and 'CA' in prev_coord and 'C' in prev_coord:
            ca_prev = torch.tensor(prev_coord['CA'], dtype=torch.float)
            c_prev = torch.tensor(prev_coord['C'], dtype=torch.float)
            angles[2] = self._compute_dihedral(ca_prev, c_prev, n_curr, ca_curr)

        if 'O' in coord_dict:
            o_curr = torch.tensor(coord_dict['O'], dtype=torch.float)
            angles[3] = self._compute_bond_angle(n_curr, ca_curr, c_curr)
            angles[4] = self._compute_bond_angle(ca_curr, c_curr, o_curr)
            angles[5] = self._compute_bond_angle(c_curr, ca_curr, n_curr)

        return angles

    def _compute_dihedral(self, p1, p2, p3, p4):
        """Compute dihedral angle between 4 points."""
        v1 = p2 - p1
        v2 = p3 - p2
        v3 = p4 - p3

        n1 = torch.cross(v1, v2)
        n2 = torch.cross(v2, v3)

        n1 = F.normalize(n1, dim=0)
        n2 = F.normalize(n2, dim=0)

        cos_angle = torch.clamp(torch.dot(n1, n2), -1, 1)
        angle = torch.acos(cos_angle)

        if torch.dot(torch.cross(n1, n2), F.normalize(v2, dim=0)) < 0:
            angle = -angle

        return angle

    def _compute_bond_angle(self, p1, p2, p3):
        """Compute bond angle at p2."""
        v1 = p1 - p2
        v2 = p3 - p2

        cos_angle = torch.dot(F.normalize(v1, dim=0), F.normalize(v2, dim=0))
        cos_angle = torch.clamp(cos_angle, -1, 1)
        return torch.acos(cos_angle)

    def compute_residue_features(self, residues):
        """Compute 105-dimensional node features for each residue."""
        features = []

        for idx, res in enumerate(residues):
            feature_parts = []

            # 1. Residue type embedding (20-D)
            res_type = res['resname'] if res['resname'] in AA_TYPES else 'ALA'
            res_idx = AA_TYPES.index(res_type)
            res_onehot = torch.zeros(20)
            res_onehot[res_idx] = 1
            feature_parts.append(res_onehot)

            # 2. Positional encoding (16-D)
            pos_enc = self.compute_positional_encoding(idx)
            feature_parts.append(pos_enc)

            # Get atomic coordinates
            residue_atoms = res['residue_obj']
            coord_dict = {}
            for atom in residue_atoms:
                if atom.element != 'H':
                    coord_dict[atom.name] = atom.coord

            # 3. Bond/dihedral angles (12-D)
            prev_res = residues[idx-1] if idx > 0 else None
            next_res = residues[idx+1] if idx < len(residues)-1 else None

            prev_coord = {}
            next_coord = {}
            if prev_res is not None:
                for atom in prev_res['residue_obj']:
                    if atom.element != 'H':
                        prev_coord[atom.name] = atom.coord
            if next_res is not None:
                for atom in next_res['residue_obj']:
                    if atom.element != 'H':
                        next_coord[atom.name] = atom.coord

            angles = self.compute_dihedral_angles(coord_dict, prev_coord, next_coord)
            angle_features = torch.stack([torch.sin(angles), torch.cos(angles)], dim=1).flatten()
            feature_parts.append(angle_features)

            # 4. RBF distances (48-D)
            if 'CA' in coord_dict:
                ca_pos = torch.tensor(coord_dict['CA'], dtype=torch.float)
                distance_features = []

                for atom_name in ['C', 'N', 'O']:
                    if atom_name in coord_dict:
                        atom_pos = torch.tensor(coord_dict[atom_name], dtype=torch.float)
                        dist = torch.norm(ca_pos - atom_pos)
                        rbf_dist = self.compute_rbf(dist)
                        distance_features.append(rbf_dist)
                    else:
                        default_dist = 1.5
                        rbf_dist = self.compute_rbf(default_dist)
                        distance_features.append(rbf_dist)

                feature_parts.append(torch.cat(distance_features))
            else:
                feature_parts.append(torch.zeros(48))

            # 5. Local coordinate frame (9-D)
            if all(atom in coord_dict for atom in ['CA', 'C', 'N']):
                Q = self.compute_local_coordinate_frame(
                    coord_dict['CA'], coord_dict['C'], coord_dict['N']
                )
                frame_features = Q.flatten()
                feature_parts.append(frame_features)
            else:
                feature_parts.append(torch.eye(3).flatten())

            full_feature = torch.cat(feature_parts)
            features.append(full_feature)

        return torch.stack(features)

    def compute_edge_features(self, i, j, residues, ca_coords, rel_type):
        """Compute 100-dimensional edge features."""
        feature_parts = []

        res_i = residues[i]
        res_j = residues[j]

        coord_i = {}
        coord_j = {}
        for atom in res_i['residue_obj']:
            if atom.element != 'H':
                coord_i[atom.name] = torch.tensor(atom.coord, dtype=torch.float)
        for atom in res_j['residue_obj']:
            if atom.element != 'H':
                coord_j[atom.name] = torch.tensor(atom.coord, dtype=torch.float)

        # 1. Edge type (4-D)
        edge_type_onehot = torch.zeros(4)
        edge_type_onehot[rel_type] = 1
        feature_parts.append(edge_type_onehot)

        # 2. Relative positional encoding (16-D)
        rel_pos = j - i if res_i['chain_id'] == res_j['chain_id'] else 0
        rel_pos_enc = self.compute_positional_encoding(rel_pos)
        feature_parts.append(rel_pos_enc)

        # 3. RBF distances (64-D)
        ca_i = ca_coords[i]
        distance_features = []

        for atom_name in ['CA', 'C', 'N', 'O']:
            if atom_name in coord_j:
                atom_j = coord_j[atom_name]
                dist = torch.norm(ca_i - atom_j)
                rbf_dist = self.compute_rbf(dist)
                distance_features.append(rbf_dist)
            else:
                dist = torch.norm(ca_i - ca_coords[j])
                rbf_dist = self.compute_rbf(dist)
                distance_features.append(rbf_dist)

        feature_parts.append(torch.cat(distance_features))

        # 4. Direction vectors (12-D)
        if all(atom in coord_i for atom in ['CA', 'C', 'N']):
            Q_i = self.compute_local_coordinate_frame(
                coord_i['CA'].numpy(), coord_i['C'].numpy(), coord_i['N'].numpy()
            )

            direction_features = []
            for atom_name in ['CA', 'C', 'N', 'O']:
                if atom_name in coord_j:
                    atom_j = coord_j[atom_name]
                    direction = atom_j - ca_i
                    direction_norm = torch.norm(direction)
                    if direction_norm > 1e-6:
                        direction = direction / direction_norm
                    else:
                        direction = torch.zeros(3)

                    local_direction = Q_i.T @ direction
                    direction_features.append(local_direction)
                else:
                    direction_features.append(torch.zeros(3))

            feature_parts.append(torch.cat(direction_features))
        else:
            feature_parts.append(torch.zeros(12))

        # 5. Quaternion (4-D)
        if (all(atom in coord_i for atom in ['CA', 'C', 'N']) and
            all(atom in coord_j for atom in ['CA', 'C', 'N'])):

            Q_i = self.compute_local_coordinate_frame(
                coord_i['CA'].numpy(), coord_i['C'].numpy(), coord_i['N'].numpy()
            )
            Q_j = self.compute_local_coordinate_frame(
                coord_j['CA'].numpy(), coord_j['C'].numpy(), coord_j['N'].numpy()
            )

            rel_rotation = Q_i.T @ Q_j
            quaternion = self._rotation_matrix_to_quaternion(rel_rotation)
            feature_parts.append(quaternion)
        else:
            feature_parts.append(torch.tensor([1.0, 0.0, 0.0, 0.0]))

        return torch.cat(feature_parts)

    def _rotation_matrix_to_quaternion(self, R):
        """Convert rotation matrix to quaternion."""
        R_np = R.detach().numpy()
        try:
            rot = Rotation.from_matrix(R_np)
            quat = rot.as_quat()
            quat = np.array([quat[3], quat[0], quat[1], quat[2]])
        except Exception:
            quat = np.array([1.0, 0.0, 0.0, 0.0])

        return torch.tensor(quat, dtype=torch.float)

    def build_multi_relational_graph(self, residues):
        """Build multi-relational residue graph."""
        n_residues = len(residues)
        ca_coords = torch.tensor([r['ca_coord'] for r in residues], dtype=torch.float)

        edge_lists = {r: [] for r in range(4)}

        # Sequential edges
        for i in range(n_residues):
            for j in range(n_residues):
                if i == j:
                    continue

                if residues[i]['chain_id'] == residues[j]['chain_id']:
                    seq_dist = abs(residues[i]['res_id'] - residues[j]['res_id'])

                    if seq_dist == 1:
                        edge_lists[0].append([i, j])
                    elif seq_dist == 2:
                        edge_lists[1].append([i, j])

        # Spatial edges
        dist_matrix = torch.cdist(ca_coords, ca_coords)

        # k-NN edges
        for i in range(n_residues):
            distances = dist_matrix[i]
            _, indices = torch.topk(-distances, k=min(self.k_nn + 1, n_residues))
            for j in indices[1:self.k_nn + 1]:
                if j != i:
                    edge_lists[2].append([i, j.item()])

        # Spatial proximity edges
        spatial_edges = torch.nonzero((dist_matrix < self.spatial_cutoff) & (dist_matrix > 0)).tolist()
        for i, j in spatial_edges:
            already_connected = any([i, j] in edge_lists[r] for r in range(3))
            if not already_connected:
                edge_lists[3].append([i, j])

        # Compute edge features
        edge_features = {}
        for rel_type, edges in edge_lists.items():
            for src, dst in edges:
                edge_key = (src, dst, rel_type)
                edge_feat = self.compute_edge_features(src, dst, residues, ca_coords, rel_type)
                edge_features[edge_key] = edge_feat

        return edge_lists, ca_coords, edge_features

    def build_residue_graph(self, pdb_path: str):
        """Build complete residue graph from PDB file."""
        residues = extract_residues_from_pdb(pdb_path)

        if len(residues) == 0:
            raise ValueError(f"No valid residues found in {pdb_path}")

        node_features = self.compute_residue_features(residues)
        edge_lists, ca_coords, edge_features = self.build_multi_relational_graph(residues)

        return Data(
            x=node_features,
            pos=ca_coords,
            edge_lists=edge_lists,
            edge_features=edge_features,
            n_residues=len(residues)
        )


# ==================== HETERODATA CONSTRUCTION ====================

def build_hetero_data_for_inference(ag_graph, ab_graph, ag_plm, ab_plm):
    """Build HeteroData object for inference."""
    hetero_data = HeteroData()

    # Add antigen residue data
    hetero_data['ag_res'].x = ag_graph.x
    hetero_data['ag_res'].plm = ag_plm
    hetero_data['ag_res'].y = torch.zeros(ag_graph.x.shape[0])
    hetero_data['ag_res'].pos = ag_graph.pos

    # Add antibody residue data
    hetero_data['ab_res'].x = ab_graph.x
    hetero_data['ab_res'].plm = ab_plm
    hetero_data['ab_res'].y = torch.zeros(ab_graph.x.shape[0])
    hetero_data['ab_res'].pos = ab_graph.pos

    # Add edge lists
    hetero_data['ag_res'].edge_lists = ag_graph.edge_lists
    hetero_data['ab_res'].edge_lists = ab_graph.edge_lists

    # Convert edge features to tensor format
    for prefix, graph in [('ag_res', ag_graph), ('ab_res', ab_graph)]:
        edge_features_list = []
        edge_keys_list = []
        for key, feat in graph.edge_features.items():
            edge_keys_list.append(list(key))
            edge_features_list.append(feat)

        if edge_features_list:
            hetero_data[prefix].edge_features_tensor = torch.stack(edge_features_list)
            hetero_data[prefix].edge_keys_tensor = torch.tensor(edge_keys_list, dtype=torch.long)

    # Add residue relation edges
    for prefix, graph in [('ag_res', ag_graph), ('ab_res', ab_graph)]:
        for rel_type in [0, 1, 2, 3]:
            rel_name = f"r{rel_type}"
            edges = graph.edge_lists.get(rel_type, [])

            if edges:
                edge_index = torch.tensor(edges).t().contiguous()
                hetero_data[prefix, rel_name, prefix].edge_index = edge_index

                edge_attrs = []
                for edge in edges:
                    edge_key = (edge[0], edge[1], rel_type)
                    if edge_key in graph.edge_features:
                        edge_attrs.append(graph.edge_features[edge_key])

                if edge_attrs:
                    hetero_data[prefix, rel_name, prefix].edge_attr = torch.stack(edge_attrs)
            else:
                hetero_data[prefix, rel_name, prefix].edge_index = torch.zeros((2, 0), dtype=torch.long)
                hetero_data[prefix, rel_name, prefix].edge_attr = torch.zeros((0, 100))

    # Placeholder interaction edges
    hetero_data['ag_res', 'interacts', 'ab_res'].edge_index = torch.zeros((2, 0), dtype=torch.long)

    hetero_data.complex_id = "inference"

    return hetero_data


# ==================== MODEL LOADING ====================

def load_checkpoint(checkpoint_path: str, device):
    """Load model checkpoint and reconstruct config."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    cfg = OmegaConf.create(checkpoint['config'])

    if 'evoformer' in cfg.model and 'epiformer' not in cfg.model:
        cfg.model.epiformer = cfg.model.evoformer

    if 'epiformer' in cfg.model:
        if 'ag_resmp_type' not in cfg.model.epiformer:
            cfg.model.epiformer.ag_resmp_type = 'egnn'
        if 'ab_resmp_type' not in cfg.model.epiformer:
            cfg.model.epiformer.ab_resmp_type = 'egnn'
        if 'geo_dim' not in cfg.model.epiformer:
            cfg.model.epiformer.geo_dim = cfg.model.get('geo_dim', 105)

    if 'graph_num_relations' not in cfg.dataset:
        cfg.dataset.graph_num_relations = 4
    if 'plm_type' not in cfg.dataset:
        cfg.dataset.plm_type = 'esm2_650m'

    model = EpiformerModel(cfg).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, cfg


# ==================== MAIN INFERENCE ====================

def run_inference(
    antigen_pdb: str,
    antibody_pdb: str,
    checkpoint_path: str,
    threshold: float = 0.3,
    device: str = None,
    skip_filtering: bool = False,
    sasa_threshold: float = 5.0
):
    """
    Run end-to-end epitope prediction on new PDB files.

    Args:
        antigen_pdb: Path to antigen PDB file
        antibody_pdb: Path to antibody PDB file
        checkpoint_path: Path to model checkpoint
        threshold: Classification threshold (default: 0.3)
        device: Device to use (default: auto-detect)
        skip_filtering: If True, assume PDBs are already filtered
        sasa_threshold: SASA threshold for surface residue detection

    Returns:
        Dictionary with predictions and residue information
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    print(f"Using device: {device}")

    # Load model
    model, cfg = load_checkpoint(checkpoint_path, device)

    # Prepare filtered PDB paths
    if skip_filtering:
        ag_filtered_pdb = antigen_pdb
        ab_filtered_pdb = antibody_pdb
        ag_full_sequence = extract_sequence_from_pdb(antigen_pdb)
        ab_full_sequence = extract_sequence_from_pdb(antibody_pdb)
        print("\nUsing input PDBs directly (--skip_filtering enabled)")
    else:
        # Create temp directory for filtered PDBs
        temp_dir = tempfile.mkdtemp()
        ag_filtered_pdb = os.path.join(temp_dir, "antigen_surface.pdb")
        ab_filtered_pdb = os.path.join(temp_dir, "antibody_cdr.pdb")

        # Filter antigen to surface residues
        print("\nFiltering antigen to surface residues...")
        ag_full_sequence = extract_sequence_from_pdb(antigen_pdb)
        surface_residues = compute_surface_residues(antigen_pdb, sasa_threshold)
        print(f"  Found {len(surface_residues)} surface residues (SASA > {sasa_threshold})")

        if len(surface_residues) == 0:
            raise ValueError("No surface residues found. Try lowering --sasa_threshold")

        filter_pdb_by_residues(antigen_pdb, ag_filtered_pdb, surface_residues)

        # Filter antibody to CDR residues
        print("\nIdentifying CDR residues...")
        cdr_residues, ab_full_sequence = identify_cdr_residues_anarci(antibody_pdb)
        print(f"  Found {len(cdr_residues)} CDR residues")

        if len(cdr_residues) == 0:
            print("Warning: No CDR residues identified. Using full antibody structure.")
            ab_filtered_pdb = antibody_pdb
        else:
            filter_pdb_by_residues(antibody_pdb, ab_filtered_pdb, cdr_residues)

    # Build residue graphs from filtered PDBs
    print("\nBuilding residue graphs...")
    graph_builder = ResidueGraphBuilder()
    ag_graph = graph_builder.build_residue_graph(ag_filtered_pdb)
    ab_graph = graph_builder.build_residue_graph(ab_filtered_pdb)

    print(f"  Antigen: {ag_graph.n_residues} residues")
    print(f"  Antibody: {ab_graph.n_residues} residues")

    # Extract sequences from filtered PDBs (for embeddings)
    ag_filtered_sequence = extract_sequence_from_pdb(ag_filtered_pdb)
    ab_filtered_sequence = extract_sequence_from_pdb(ab_filtered_pdb)

    # Generate PLM embeddings for filtered sequences
    print("\nGenerating PLM embeddings...")

    # ESM-2 for antigen
    esm_model, esm_batch_converter = load_esm2_model(device)
    ag_plm = embed_esm2(esm_model, esm_batch_converter, ag_filtered_sequence, device)
    print(f"  Antigen ESM-2 embeddings: {ag_plm.shape}")

    # AntiBERTy for antibody
    antiberty = load_antiberty_model()
    ab_plm = embed_antiberty(antiberty, ab_filtered_sequence)
    print(f"  Antibody AntiBERTy embeddings: {ab_plm.shape}")

    # Verify embedding dimensions match graph
    if ag_plm.shape[0] != ag_graph.n_residues:
        raise ValueError(f"Antigen embedding size ({ag_plm.shape[0]}) != graph size ({ag_graph.n_residues})")
    if ab_plm.shape[0] != ab_graph.n_residues:
        raise ValueError(f"Antibody embedding size ({ab_plm.shape[0]}) != graph size ({ab_graph.n_residues})")

    # Build HeteroData
    print("\nBuilding graph tensor...")
    hetero_data = build_hetero_data_for_inference(ag_graph, ab_graph, ag_plm, ab_plm)

    # Run prediction
    batch = Batch.from_data_list([hetero_data]).to(device)

    print("\nRunning prediction...")
    with torch.no_grad():
        outputs = model(batch)

    epitope_probs = outputs['epitope_prob'].cpu().numpy()
    epitope_pred = (epitope_probs > threshold).astype(int)

    # Get residue information from filtered PDB
    ag_residues = extract_residues_from_pdb(ag_filtered_pdb)

    # Compile results
    results = {
        'antigen_pdb': antigen_pdb,
        'antibody_pdb': antibody_pdb,
        'threshold': threshold,
        'n_antigen_residues': len(ag_residues),
        'n_antibody_residues': ab_graph.n_residues,
        'epitope_probabilities': epitope_probs,
        'epitope_predictions': epitope_pred,
        'predicted_epitope_count': int(epitope_pred.sum()),
        'residue_details': []
    }

    for i, res in enumerate(ag_residues):
        results['residue_details'].append({
            'index': i,
            'residue_name': res['resname'],
            'residue_id': res['res_id'],
            'chain_id': res['chain_id'],
            'probability': float(epitope_probs[i]),
            'is_epitope': bool(epitope_pred[i])
        })

    # Cleanup temp files
    if not skip_filtering:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return results


def print_results(results: Dict):
    """Print prediction results in a formatted way."""
    print(f"\n{'='*60}")
    print("EPITOPE PREDICTION RESULTS")
    print(f"{'='*60}")
    print(f"Antigen PDB: {results['antigen_pdb']}")
    print(f"Antibody PDB: {results['antibody_pdb']}")
    print(f"Threshold: {results['threshold']}")
    print(f"{'-'*60}")
    print(f"Total antigen residues (surface): {results['n_antigen_residues']}")
    print(f"Total antibody residues (CDR): {results['n_antibody_residues']}")
    print(f"Predicted epitope residues: {results['predicted_epitope_count']}")
    print(f"Epitope ratio: {results['predicted_epitope_count']/results['n_antigen_residues']:.2%}")
    print(f"{'-'*60}")

    print("\nPredicted Epitope Residues:")
    print(f"{'Chain':<6} {'ResID':<8} {'ResName':<8} {'Probability':<12}")
    print(f"{'-'*40}")

    for res in results['residue_details']:
        if res['is_epitope']:
            print(f"{res['chain_id']:<6} {res['residue_id']:<8} {res['residue_name']:<8} {res['probability']:.4f}")

    print(f"\n{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="EpiFormer Inference: Predict epitope residues from PDB files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (with automatic surface/CDR filtering)
  python inference.py --antigen_pdb ag.pdb --antibody_pdb ab.pdb --checkpoint model.pt

  # Skip filtering if PDBs are already pre-processed
  python inference.py --antigen_pdb ag_surface.pdb --antibody_pdb ab_cdr.pdb \\
      --checkpoint model.pt --skip_filtering

  # Save results to JSON
  python inference.py --antigen_pdb ag.pdb --antibody_pdb ab.pdb \\
      --checkpoint model.pt --output predictions.json

Note:
  - The model expects antigen SURFACE residues and antibody CDR residues
  - By default, surface residues are detected via SASA calculation
  - CDR regions are identified using ANARCI (if installed) or Chothia numbering
  - Use --skip_filtering if your PDBs are already filtered appropriately
        """
    )
    parser.add_argument(
        "--antigen_pdb",
        type=str,
        required=True,
        help="Path to antigen PDB file"
    )
    parser.add_argument(
        "--antibody_pdb",
        type=str,
        required=True,
        help="Path to antibody PDB file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Classification threshold (default: 0.3)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: auto-detect)"
    )
    parser.add_argument(
        "--skip_filtering",
        action="store_true",
        help="Skip surface/CDR filtering (use if PDBs are already filtered)"
    )
    parser.add_argument(
        "--sasa_threshold",
        type=float,
        default=5.0,
        help="SASA threshold for surface residue detection (default: 5.0 A^2)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for results (JSON format)"
    )
    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.antigen_pdb):
        print(f"Error: Antigen PDB not found: {args.antigen_pdb}")
        sys.exit(1)
    if not os.path.exists(args.antibody_pdb):
        print(f"Error: Antibody PDB not found: {args.antibody_pdb}")
        sys.exit(1)
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Run inference
    results = run_inference(
        antigen_pdb=args.antigen_pdb,
        antibody_pdb=args.antibody_pdb,
        checkpoint_path=args.checkpoint,
        threshold=args.threshold,
        device=args.device,
        skip_filtering=args.skip_filtering,
        sasa_threshold=args.sasa_threshold
    )

    # Print results
    print_results(results)

    # Save to file if requested
    if args.output:
        output_results = results.copy()
        output_results['epitope_probabilities'] = results['epitope_probabilities'].tolist()
        output_results['epitope_predictions'] = results['epitope_predictions'].tolist()

        with open(args.output, 'w') as f:
            json.dump(output_results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()


"""
(asep_baselines) user@myspace:~/epiformer/icml_26/code$ python inference.py     --antigen_pdb /media/volume/data/asep/antigen/structures/1a14_0P_ag.pdb     --antibody_pdb /media/volume/data/asep/antibody/structures/1a14_0P_ab.pdb     --checkpoint ../../checkpoints/best-glamorous-sweep-37/epiformer_best.pt

The cache for model files in Transformers v4.22.0 has been updated. Migrating your old cache. This is a one-time only operation. You can interrupt this and resume the migration later on by calling `transformers.utils.move_cache()`.
0it [00:00, ?it/s]
Using device: cuda
Loading checkpoint from: ../../checkpoints/best-glamorous-sweep-37/epiformer_best.pt
Model loaded successfully
  Parameters: 5,820,178

Filtering antigen to surface residues...
  Found 263 surface residues (SASA > 5.0)

Identifying CDR residues...
Warning: ANARCI not installed. Install with: pip install anarci
Falling back to Chothia numbering based on residue IDs.
  Found 47 CDR residues

Building residue graphs...
  Antigen: 263 residues
  Antibody: 64 residues

Generating PLM embeddings...
Loading ESM-2 650M model...
  Antigen ESM-2 embeddings: torch.Size([263, 1280])
Loading AntiBERTy model...
  Antibody AntiBERTy embeddings: torch.Size([64, 512])

Building graph tensor...

Running prediction...

============================================================
EPITOPE PREDICTION RESULTS
============================================================
Antigen PDB: /media/volume/data/asep/antigen/structures/1a14_0P_ag.pdb
Antibody PDB: /media/volume/data/asep/antibody/structures/1a14_0P_ab.pdb
Threshold: 0.3
------------------------------------------------------------
Total antigen residues (surface): 263
Total antibody residues (CDR): 64
Predicted epitope residues: 23
Epitope ratio: 8.75%
------------------------------------------------------------

Predicted Epitope Residues:
Chain  ResID    ResName  Probability 
----------------------------------------
N      236      ILE      0.4631
N      240      VAL      0.3692
N      246      ARG      0.5946
N      247      PRO      0.6711
N      248      ASN      0.6081
N      249      ASP      0.7393
N      250      PRO      0.5534
N      251      THR      0.4678
N      252      VAL      0.7005
N      253      GLY      0.9078
N      254      LYS      0.7063
N      255      CYS      0.7636
N      256      ASN      0.3820
N      259      TYR      0.4509
N      261      GLY      0.4153
N      275      GLY      0.3312
N      299      ASN      0.4417
N      301      LEU      0.7119
N      302      THR      0.8297
N      303      ASP      0.8690
N      304      ASP      0.7180
N      305      LYS      0.8029
N      306      SER      0.7815

============================================================

PyMOL selection command:
select epitope, (chain N and resi 236) or (chain N and resi 240) or (chain N and resi 246) or (chain N and resi 247) or (chain N and resi 248) or (chain N and resi 249) or (chain N and resi 250) or (chain N and resi 251) or (chain N and resi 252) or (chain N and resi 253) or (chain N and resi 254) or (chain N and resi 255) or (chain N and resi 256) or (chain N and resi 259) or (chain N and resi 261) or (chain N and resi 275) or (chain N and resi 299) or (chain N and resi 301) or (chain N and resi 302) or (chain N and resi 303) or (chain N and resi 304) or (chain N and resi 305) or (chain N and resi 306)

"""