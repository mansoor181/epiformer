# """
# Create dataset for hierarchical encoder combining AsEP graphs with PDB structure files.

# This script processes the AsEP dataset and PDB structure files to create comprehensive
# data tensors containing atom-level, residue-level, and edge-level graphs for the hierarchical encoder.

# Graph Hierarchy:
# 1. AtomMP: processes atom_graph → atom embeddings
# 2. EdgeMP: processes edge_graph (line graph of residue edges) → edge embeddings  
# 3. ResMP: processes residue_graph + aggregated atom + injected edge embeddings → final residue embeddings
# 4. Decoder: processes final antigen & antibody residue embeddings → interaction prediction
# """

# """
# TODO:
# - add the plm embeddings to the residue graph node features
# - remove the bipartite matrix, except edge_index_ag_ab
# """


"""

HeteroData(
  complex_id='complex_identifier',  # e.g., '1s78_0P'
  
  # ==================== NODE STORES ====================
  
  # Antigen Atom Level
  ag_atom={
    x=[num_ag_atoms, 28],     # Atom features (7 elements + 21 residue types)
    pos=[num_ag_atoms, 3],    # 3D atomic coordinates
  },
  
  # Antigen Residue Level  
  ag_res={
    x=[num_ag_residues, 105],                    # Residue features (RAAD: 20+16+12+48+9)
    plm=[num_ag_residues, plm_dim],              # PLM embeddings (ESM-2: 480/1280 dim)
    y=[num_ag_residues],                         # Epitope labels (0/1)
    pos=[num_ag_residues, 3],                    # CA coordinates
    edge_lists={                                 # Edge lists by relation type
      0=[num_r0_edges],                          # r0 edges (sequential ±1)
      1=[num_r1_edges],                          # r1 edges (sequential ±2)  
      2=[num_r2_edges],                          # r2 edges (k-NN)
      3=[num_r3_edges],                          # r3 edges (spatial)
    },
    edge_features_tensor=[total_edge_features, 100],  # All edge features stacked
    edge_keys_tensor=[total_edge_features, 3],        # Edge keys (src, dst, rel)
  },
  
  # Antibody Atom Level
  ab_atom={
    x=[num_ab_atoms, 28],     # Atom features (7 elements + 21 residue types)
    pos=[num_ab_atoms, 3],    # 3D atomic coordinates
  },
  
  # Antibody Residue Level
  ab_res={
    x=[num_ab_residues, 105],                    # Residue features (RAAD: 20+16+12+48+9)
    plm=[num_ab_residues, plm_dim],              # PLM embeddings (AbLang/AntiBERTy: 512 dim)
    y=[num_ab_residues],                         # Paratope labels (0/1)
    pos=[num_ab_residues, 3],                    # CA coordinates
    edge_lists={                                 # Edge lists by relation type
      0=[num_r0_edges],                          # r0 edges (sequential ±1)
      1=[num_r1_edges],                          # r1 edges (sequential ±2)  
      2=[num_r2_edges],                          # r2 edges (k-NN)
      3=[num_r3_edges],                          # r3 edges (spatial)
    },
    edge_features_tensor=[total_edge_features, 100],  # All edge features stacked
    edge_keys_tensor=[total_edge_features, 3],        # Edge keys (src, dst, rel)
  },
  
  # Edge Graph Nodes (Line Graph for EdgeMP)
  ag_edge={ 
    x=[num_ag_edge_nodes, 100],           # Edge node features (from residue graph edges)
    edge_mapping=[num_ag_edge_nodes, 3],  # Mapping to residue edges (src, dst, rel)
  },
  ab_edge={ 
    x=[num_ab_edge_nodes, 100],           # Edge node features (from residue graph edges)
    edge_mapping=[num_ab_edge_nodes, 3],  # Mapping to residue edges (src, dst, rel)
  },
  
  # ==================== EDGE STORES ====================
  
  # Atom-Level Bonds
  (ag_atom, atom_bond, ag_atom)={
    edge_index=[2, num_ag_atom_bonds],    # Spatial bonds within cutoff (4.5Å)
    edge_attr=[num_ag_atom_bonds, 17],    # Distance + RBF encoding (1+16)
  },
  (ab_atom, atom_bond, ab_atom)={
    edge_index=[2, num_ab_atom_bonds],    # Spatial bonds within cutoff (4.5Å)
    edge_attr=[num_ab_atom_bonds, 17],    # Distance + RBF encoding (1+16)
  },
  
  # Residue-Level Multi-Relational Edges
  
  # r0: Sequential ±1 neighbors
  (ag_res, r0, ag_res)={
    edge_index=[2, num_ag_r0_edges],      # Sequential distance = 1
    edge_attr=[num_ag_r0_edges, 100],     # RAAD edge features
  },
  (ab_res, r0, ab_res)={
    edge_index=[2, num_ab_r0_edges],      # Sequential distance = 1
    edge_attr=[num_ab_r0_edges, 100],     # RAAD edge features
  },
  
  # r1: Sequential ±2 neighbors
  (ag_res, r1, ag_res)={
    edge_index=[2, num_ag_r1_edges],      # Sequential distance = 2
    edge_attr=[num_ag_r1_edges, 100],     # RAAD edge features
  },
  (ab_res, r1, ab_res)={
    edge_index=[2, num_ab_r1_edges],      # Sequential distance = 2
    edge_attr=[num_ab_r1_edges, 100],     # RAAD edge features
  },
  
  # r2: k-NN spatial neighbors (k=10)
  (ag_res, r2, ag_res)={
    edge_index=[2, num_ag_r2_edges],      # k-nearest neighbors
    edge_attr=[num_ag_r2_edges, 100],     # RAAD edge features
  },
  (ab_res, r2, ab_res)={
    edge_index=[2, num_ab_r2_edges],      # k-nearest neighbors
    edge_attr=[num_ab_r2_edges, 100],     # RAAD edge features
  },
  
  # r3: Spatial proximity (within 8.0Å, not in r0-r2)
  (ag_res, r3, ag_res)={
    edge_index=[2, num_ag_r3_edges],      # Spatial cutoff edges
    edge_attr=[num_ag_r3_edges, 100],     # RAAD edge features
  },
  (ab_res, r3, ab_res)={
    edge_index=[2, num_ab_r3_edges],      # Spatial cutoff edges
    edge_attr=[num_ab_r3_edges, 100],     # RAAD edge features
  },
  
  # Line Graph (Edge-Edge) Connections
  (ag_edge, edge_connect, ag_edge)={
    edge_index=[2, num_ag_line_edges],    # Edges between edge nodes
    edge_attr=[num_ag_line_edges, 1],     # Edge type encoding
  },
  (ab_edge, edge_connect, ab_edge)={
    edge_index=[2, num_ab_line_edges],    # Edges between edge nodes
    edge_attr=[num_ab_line_edges, 1],     # Edge type encoding
  },
  
  # Cross-Hierarchy Connections
  
  # Atom-to-Residue mapping
  (ag_atom, belongs_to, ag_res)={ 
    edge_index=[2, num_ag_atoms]          # Each atom belongs to one residue
  },
  (ab_atom, belongs_to, ab_res)={ 
    edge_index=[2, num_ab_atoms]          # Each atom belongs to one residue
  },
  
  # Residue-to-Edge mapping
  (ag_res, connected_by, ag_edge)={ 
    edge_index=[2, num_ag_res_edge_connections]  # Residues connected by edge nodes
  },
  (ab_res, connected_by, ab_edge)={ 
    edge_index=[2, num_ab_res_edge_connections]  # Residues connected by edge nodes
  },
  
  # Inter-Protein Interactions
  (ag_res, interacts, ab_res)={ 
    edge_index=[2, num_interaction_pairs]  # Bipartite antigen-antibody interactions
  }
)

"""



"""
Create dataset for encoder combining AsEP graphs with PDB structure files.
Modified to create HeteroData objects for each complex sample.
"""
import os
import sys
import math
import warnings
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio.PDB import PDBParser, Select
from Bio.PDB.PDBIO import PDBIO
from biopandas.pdb import PandasPdb
from scipy.spatial.transform import Rotation as R
from torch_geometric.data import Data, HeteroData
from torch_scatter import scatter_add

warnings.filterwarnings("ignore")

# Add project directories to path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__), '../../walle'))

# ==================== ATOM GRAPH CONSTRUCTION ====================

class PDBProcessor:
    """Processes PDB files to extract atom information and build graphs"""
    def __init__(self, pdb_path, include_hydrogens=False):
        self.pdb_path = pdb_path
        self.include_hydrogens = include_hydrogens
        self.parser = PDBParser(QUIET=True)
        self.structure = self.parser.get_structure("pdb", pdb_path)
        self.atoms = []
        self.coords = []
        
    def extract_atoms(self):
        """Extract atoms and coordinates from PDB file"""
        for model in self.structure:
            for chain in model:
                for residue in chain:
                    # Skip heterogens and water
                    if residue.id[0] != ' ':
                        continue
                        
                    for atom in residue:
                        if not self.include_hydrogens and atom.element == 'H':
                            continue
                        self.atoms.append({
                            'name': atom.get_name(),
                            'element': atom.element,
                            'residue': residue.resname,
                            'residue_id': residue.id[1],
                            'chain_id': chain.id,
                            'coord': atom.get_coord()
                        })
                        self.coords.append(atom.get_coord())
        return self
    
    def get_atom_features(self):
        """Create feature vectors for atoms"""
        # Define feature mappings
        elements = ['C', 'N', 'O', 'S', 'P', 'H', 'OTHER']
        residues = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 
                    'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 
                    'THR', 'TRP', 'TYR', 'VAL', 'OTHER']
        
        features = []
        for atom in self.atoms:
            # One-hot encode element
            elem_idx = elements.index(atom['element']) if atom['element'] in elements else -1
            elem_onehot = torch.zeros(len(elements))
            elem_onehot[elem_idx if elem_idx != -1 else -1] = 1
            
            # One-hot encode residue
            res_idx = residues.index(atom['residue']) if atom['residue'] in residues else -1
            res_onehot = torch.zeros(len(residues))
            res_onehot[res_idx if res_idx != -1 else -1] = 1
            
            # Combine features (total: 7 + 21 = 28 dimensions)
            features.append(torch.cat([elem_onehot, res_onehot]))
        
        return torch.stack(features)
    
    def build_graph(self, cutoff=4.5):
        """Build atom graph with edges based on spatial proximity"""
        coords = torch.tensor(self.coords, dtype=torch.float)
        n_atoms = len(self.atoms)
        
        # Compute distance matrix
        dist_matrix = torch.cdist(coords, coords)
        
        # Create edges (within cutoff and excluding self-loops)
        src, dst = torch.nonzero((dist_matrix < cutoff) & (dist_matrix > 0), as_tuple=True)
        
        # Create edge attributes (distance + RBF expansion)
        distances = dist_matrix[src, dst].unsqueeze(1)
        offsets = torch.linspace(0, 20, 16)  # 16 RBF functions
        widths = torch.ones_like(offsets) * 1.0
        rbf_expanded = torch.exp(-0.5 * ((distances - offsets) / widths) ** 2)
        edge_attr = torch.cat([distances, rbf_expanded], dim=1)  # 1 + 16 = 17 dimensions
        
        # # Create residue mapping for aggregation
        # residue_indices = []
        # for atom in self.atoms:
        #     residue_indices.append(atom['residue_id'])

        # """
        # TODO: -make the residue indices sequential for atom-2-res mapping
        # """
        # residue_indices = make_sequential(residue_indices)

        # Create residue mapping for aggregation
        residue_indices = []
        residue_id_to_index = {}
        
        # First pass: map residue IDs to sequential indices
        unique_residues = sorted(set(atom['residue_id'] for atom in self.atoms))
        residue_id_to_index = {res_id: idx for idx, res_id in enumerate(unique_residues)}
        
        # Second pass: assign indices to atoms
        for atom in self.atoms:
            residue_indices.append(residue_id_to_index[atom['residue_id']])

        
        # Create PyG Data object
        return Data(
            x=self.get_atom_features(),  # (n_atoms, 28)
            pos=coords,                  # (n_atoms, 3)
            edge_index=torch.stack([src, dst]),  # (2, n_edges)
            edge_attr=edge_attr,         # (n_edges, 17)
            residue_indices=torch.tensor(residue_indices)  # For aggregation
        )

def make_sequential(lst):
    # Get the unique values and sort them
    unique_values = sorted(set(lst))

    # Create a mapping from the original values to sequential values
    value_to_sequential = {value: idx for idx, value in enumerate(unique_values)}

    # Transform the original list using the mapping
    sequential_list = [value_to_sequential[value] for value in lst]

    return sequential_list


def process_pdb_file(pdb_path, include_hydrogens=False):
    """
    Process a PDB file into a graph representation
    Args:
        pdb_path: Path to PDB file
        include_hydrogens: Whether to include hydrogen atoms
    Returns:
        PyG Data object with atom graph
    """
    processor = PDBProcessor(pdb_path, include_hydrogens)
    processor.extract_atoms()
    return processor.build_graph()


# ==================== RESIDUE GRAPH CONSTRUCTION ====================

class ResidueGraphBuilder:
    """Builds multi-relational residue graphs from PDB structures with full RAAD features"""
    
    def __init__(self, k_nn=10, spatial_cutoff=8.0, sequential_cutoffs=[1, 2]):
        self.k_nn = k_nn
        self.spatial_cutoff = spatial_cutoff
        self.sequential_cutoffs = sequential_cutoffs
        self.parser = PDBParser(QUIET=True)
        
        # RBF parameters
        self.rbf_centers = torch.linspace(0, 20, 16)
        self.rbf_width = 1.0

    def extract_residues(self, pdb_path):
        """Extract residue information from PDB file"""
        structure = self.parser.get_structure("pdb", pdb_path)
        residues = []
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    # Skip heterogens and water
                    if residue.id[0] != ' ':
                        continue
                    
                    # Get CA atom position
                    if 'CA' in residue:
                        ca_coord = residue['CA'].get_coord()
                    else:
                        # Skip residues without CA
                        continue
                    
                    residues.append({
                        'resname': residue.resname,
                        'chain_id': chain.id,
                        'res_id': residue.id[1],
                        'ca_coord': ca_coord,
                        'residue_obj': residue
                    })
        
        return residues
        
    def compute_rbf(self, distances):
        """Compute radial basis function encoding"""
        if isinstance(distances, (int, float)):
            distances = torch.tensor([distances], dtype=torch.float)
        elif not isinstance(distances, torch.Tensor):
            distances = torch.tensor(distances, dtype=torch.float)
        
        # Expand distances to match RBF centers
        if distances.dim() == 0:
            distances = distances.unsqueeze(0)
        
        rbf = torch.exp(-0.5 * ((distances.unsqueeze(-1) - self.rbf_centers) / self.rbf_width) ** 2)
        return rbf.squeeze(0) if rbf.shape[0] == 1 else rbf

    def compute_positional_encoding(self, position, max_len=1000):
        """Compute sinusoidal positional encoding"""
        pe = torch.zeros(16)
        position = torch.tensor([position], dtype=torch.float)
        
        div_term = torch.exp(torch.arange(0, 16, 2).float() * 
                           -(math.log(10000.0) / 16))
        
        pe[0::2] = torch.sin(position * div_term)
        pe[1::2] = torch.cos(position * div_term)
        return pe

    def compute_local_coordinate_frame(self, ca_coord, c_coord, n_coord):
        """Compute local coordinate frame Q_i from CA, C, N atoms"""
        ca = torch.tensor(ca_coord, dtype=torch.float)
        c = torch.tensor(c_coord, dtype=torch.float)
        n = torch.tensor(n_coord, dtype=torch.float)
        
        # Create coordinate frame
        v1 = c - ca  # CA -> C
        v2 = n - ca  # CA -> N
        
        # Gram-Schmidt orthogonalization
        u1 = F.normalize(v1, dim=0)
        u2_temp = v2 - torch.dot(v2, u1) * u1
        u2 = F.normalize(u2_temp, dim=0)
        u3 = torch.cross(u1, u2)
        
        Q = torch.stack([u1, u2, u3], dim=1)  # 3x3 matrix
        return Q

    def compute_dihedral_angles(self, residue, prev_residue=None, next_residue=None):
        """Compute 6 backbone angles: phi, psi, omega, alpha, beta, gamma"""
        angles = torch.zeros(6)
        
        try:
            # Current residue atoms
            n_curr = torch.tensor(residue['N'], dtype=torch.float)
            ca_curr = torch.tensor(residue['CA'], dtype=torch.float)
            c_curr = torch.tensor(residue['C'], dtype=torch.float)
            
            # Phi angle (previous C - current N - current CA - current C)
            if prev_residue is not None and 'C' in prev_residue:
                c_prev = torch.tensor(prev_residue['C'], dtype=torch.float)
                phi = self.compute_dihedral(c_prev, n_curr, ca_curr, c_curr)
                angles[0] = phi
            
            # Psi angle (current N - current CA - current C - next N)
            if next_residue is not None and 'N' in next_residue:
                n_next = torch.tensor(next_residue['N'], dtype=torch.float)
                psi = self.compute_dihedral(n_curr, ca_curr, c_curr, n_next)
                angles[1] = psi
            
            # Omega angle (previous CA - previous C - current N - current CA)
            if prev_residue is not None and 'CA' in prev_residue and 'C' in prev_residue:
                ca_prev = torch.tensor(prev_residue['CA'], dtype=torch.float)
                c_prev = torch.tensor(prev_residue['C'], dtype=torch.float)
                omega = self.compute_dihedral(ca_prev, c_prev, n_curr, ca_curr)
                angles[2] = omega
            
            # Bond angles alpha, beta, gamma
            if 'O' in residue:
                o_curr = torch.tensor(residue['O'], dtype=torch.float)
                # Alpha: N-CA-C angle
                alpha = self.compute_bond_angle(n_curr, ca_curr, c_curr)
                angles[3] = alpha
                
                # Beta: CA-C-O angle  
                beta = self.compute_bond_angle(ca_curr, c_curr, o_curr)
                angles[4] = beta
                
                # Gamma: C-CA-N angle
                gamma = self.compute_bond_angle(c_curr, ca_curr, n_curr)
                angles[5] = gamma
                
        except Exception as e:
            print(f"Warning: Could not compute all angles: {e}")
        
        return angles

    def compute_dihedral(self, p1, p2, p3, p4):
        """Compute dihedral angle between 4 points"""
        v1 = p2 - p1
        v2 = p3 - p2  
        v3 = p4 - p3
        
        n1 = torch.cross(v1, v2)
        n2 = torch.cross(v2, v3)
        
        n1 = F.normalize(n1, dim=0)
        n2 = F.normalize(n2, dim=0)
        
        cos_angle = torch.clamp(torch.dot(n1, n2), -1, 1)
        angle = torch.acos(cos_angle)
        
        # Check sign
        if torch.dot(torch.cross(n1, n2), F.normalize(v2, dim=0)) < 0:
            angle = -angle
            
        return angle

    def compute_bond_angle(self, p1, p2, p3):
        """Compute bond angle at p2"""
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = torch.dot(F.normalize(v1, dim=0), F.normalize(v2, dim=0))
        cos_angle = torch.clamp(cos_angle, -1, 1)
        return torch.acos(cos_angle)

    def compute_residue_features(self, residues):
        """Compute 105-dimensional node features for each residue"""
        aa_types = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 
                    'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 
                    'THR', 'TRP', 'TYR', 'VAL']
        
        features = []
        for idx, res in enumerate(residues):
            feature_parts = []
            
            # 1. Residue type embedding (20-D)
            res_type = res['resname'] if res['resname'] in aa_types else 'ALA'  # fallback
            res_idx = aa_types.index(res_type)
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
                if atom.element != 'H':  # Skip hydrogens
                    coord_dict[atom.name] = atom.coord
            
            # 3. Bond/dihedral angles (12-D: 6 angles × 2 for sin/cos)
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
            
            # 4. RBF distances (48-D: 3 distances × 16 RBF each)
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
                        # Use default distance if atom missing
                        default_dist = 1.5  # Typical bond length
                        rbf_dist = self.compute_rbf(default_dist)
                        distance_features.append(rbf_dist)
                
                feature_parts.append(torch.cat(distance_features))
            else:
                # Fallback if CA missing
                feature_parts.append(torch.zeros(48))
            
            # 5. Local coordinate frame directions (9-D)
            if all(atom in coord_dict for atom in ['CA', 'C', 'N']):
                Q = self.compute_local_coordinate_frame(
                    coord_dict['CA'], coord_dict['C'], coord_dict['N']
                )
                frame_features = Q.flatten()
                feature_parts.append(frame_features)
            else:
                # Identity matrix as fallback
                feature_parts.append(torch.eye(3).flatten())
            
            # Concatenate all features (should sum to 105)
            full_feature = torch.cat(feature_parts)
            assert full_feature.shape[0] == 105, f"Feature dimension mismatch: {full_feature.shape[0]} != 105"
            features.append(full_feature)
        
        return torch.stack(features)

    def compute_edge_features(self, i, j, residues, ca_coords):
        """Compute 100-dimensional edge features between residues i and j"""
        feature_parts = []
        
        # Get residue objects
        res_i = residues[i]
        res_j = residues[j]
        
        # Get atomic coordinates
        coord_i = {}
        coord_j = {}
        for atom in res_i['residue_obj']:
            if atom.element != 'H':
                coord_i[atom.name] = torch.tensor(atom.coord, dtype=torch.float)
        for atom in res_j['residue_obj']:
            if atom.element != 'H':
                coord_j[atom.name] = torch.tensor(atom.coord, dtype=torch.float)
        
        # 1. Edge type encoding (4-D) - will be set by calling function
        # For now, create placeholder - this should be set based on relation type
        edge_type_onehot = torch.zeros(4)
        feature_parts.append(edge_type_onehot)
        
        # 2. Relative positional encoding (16-D)
        rel_pos = j - i if res_i['chain_id'] == res_j['chain_id'] else 0
        rel_pos_enc = self.compute_positional_encoding(rel_pos)
        feature_parts.append(rel_pos_enc)
        
        # 3. RBF distances to 4 backbone atoms (64-D: 4 × 16)
        ca_i = ca_coords[i]
        distance_features = []
        
        for atom_name in ['CA', 'C', 'N', 'O']:
            if atom_name in coord_j:
                atom_j = coord_j[atom_name]
                dist = torch.norm(ca_i - atom_j)
                rbf_dist = self.compute_rbf(dist)
                distance_features.append(rbf_dist)
            else:
                # Use CA-CA distance as fallback
                dist = torch.norm(ca_i - ca_coords[j])
                rbf_dist = self.compute_rbf(dist)
                distance_features.append(rbf_dist)
        
        feature_parts.append(torch.cat(distance_features))
        
        # 4. Direction vectors in local frame (12-D: 4 directions × 3D)
        if all(atom in coord_i for atom in ['CA', 'C', 'N']):
            Q_i = self.compute_local_coordinate_frame(
                coord_i['CA'], coord_i['C'], coord_i['N']
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
                    
                    # Transform to local coordinate frame
                    local_direction = Q_i.T @ direction
                    direction_features.append(local_direction)
                else:
                    direction_features.append(torch.zeros(3))
            
            feature_parts.append(torch.cat(direction_features))
        else:
            # Fallback to zero directions
            feature_parts.append(torch.zeros(12))
        
        # 5. Quaternion orientation (4-D)
        if (all(atom in coord_i for atom in ['CA', 'C', 'N']) and 
            all(atom in coord_j for atom in ['CA', 'C', 'N'])):
            
            Q_i = self.compute_local_coordinate_frame(
                coord_i['CA'], coord_i['C'], coord_i['N']
            )
            Q_j = self.compute_local_coordinate_frame(
                coord_j['CA'], coord_j['C'], coord_j['N']
            )
            
            # Compute relative rotation Q_i^T @ Q_j
            rel_rotation = Q_i.T @ Q_j
            
            # Convert to quaternion
            quaternion = self.rotation_matrix_to_quaternion(rel_rotation)
            feature_parts.append(quaternion)
        else:
            # Identity quaternion as fallback
            feature_parts.append(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        
        # Concatenate all features (should sum to 100)
        full_feature = torch.cat(feature_parts)
        
        # Handle the edge type encoding separately since it depends on relation type
        return full_feature

    def rotation_matrix_to_quaternion(self, R):
        """Convert rotation matrix to quaternion"""
        # Ensure it's a valid rotation matrix
        R = R.detach().numpy()
        try:
            rot = R.from_matrix(R)
            quat = rot.as_quat()  # Returns [x, y, z, w]
            # Convert to [w, x, y, z] format
            quat = np.array([quat[3], quat[0], quat[1], quat[2]])
        except:
            # Fallback to identity quaternion
            quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        return torch.tensor(quat, dtype=torch.float)

    def build_multi_relational_graph(self, residues):
        """Build multi-relational residue graph with complete features"""
        n_residues = len(residues)
        ca_coords = torch.tensor([r['ca_coord'] for r in residues], dtype=torch.float)
        
        # Initialize edge lists for different relations
        edge_lists = {r: [] for r in range(4)}  # r0, r1, r2, r3
        
        # Sequential edges (r0: seq±1, r1: seq±2)  
        for i in range(n_residues):
            for j in range(n_residues):
                if i == j:
                    continue
                
                # Check if same chain
                if residues[i]['chain_id'] == residues[j]['chain_id']:
                    seq_dist = abs(residues[i]['res_id'] - residues[j]['res_id'])
                    
                    if seq_dist == 1:
                        edge_lists[0].append([i, j])  # r0
                    elif seq_dist == 2:
                        edge_lists[1].append([i, j])  # r1
        
        # Spatial edges
        dist_matrix = torch.cdist(ca_coords, ca_coords)
        
        # r2: k-NN edges
        for i in range(n_residues):
            distances = dist_matrix[i]
            # Exclude self and get k nearest
            _, indices = torch.topk(-distances, k=min(self.k_nn + 1, n_residues))
            for j in indices[1:self.k_nn + 1]:  # Skip self
                if j != i:
                    edge_lists[2].append([i, j.item()])
        
        # r3: spatial proximity (within cutoff, not already connected)
        spatial_edges = torch.nonzero((dist_matrix < self.spatial_cutoff) & (dist_matrix > 0)).tolist()
        for i, j in spatial_edges:
            # Check if not already connected by other relations
            already_connected = any([i, j] in edge_lists[r] for r in range(3))
            if not already_connected:
                edge_lists[3].append([i, j])

        
        # Compute edge features with relation types
        edge_features = {}
        relation_names = ['seq±1', 'seq±2', 'k-NN', 'spatial']

        for rel_type, edges in edge_lists.items():
            for src, dst in edges:
                edge_key = (src, dst, rel_type)
                edge_feat = self.compute_edge_features(src, dst, residues, ca_coords)
                
                # Set the edge type encoding (first 4 dimensions)
                edge_type_onehot = torch.zeros(4)
                edge_type_onehot[rel_type] = 1
                edge_feat[:4] = edge_type_onehot
                
                edge_features[edge_key] = edge_feat
        
        return edge_lists, ca_coords, edge_features

    def build_residue_graph(self, pdb_path):
        """Build complete residue graph from PDB file"""
        residues = self.extract_residues(pdb_path)
        node_features = self.compute_residue_features(residues)
        edge_lists, ca_coords, edge_features = self.build_multi_relational_graph(residues)
        
        # Convert to PyG format
        all_edges = []
        all_edge_attrs = []
        
        for rel_type, edges in edge_lists.items():
            for edge in edges:
                all_edges.append(edge)
                all_edge_attrs.append(edge_features[(edge[0], edge[1], rel_type)])
        
        if all_edges:
            edge_index = torch.tensor(all_edges).T
            edge_attr = torch.stack(all_edge_attrs)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 100))
        
        return Data(
            x=node_features,      # (n_residues, 105)
            pos=ca_coords,        # (n_residues, 3)  
            edge_index=edge_index,  # (2, n_edges)
            edge_attr=edge_attr,   # (n_edges, 100)
            edge_lists=edge_lists,  # Dict of edge lists by relation type
            edge_features=edge_features  # Original edge features dict
        )




# ==================== HETERODATA CONSTRUCTION ====================


def build_hetero_data(complex_data):
    """
    Create a HeteroData object for the complex sample
    Fixed to avoid problematic edge mapping attributes that cause DataLoader issues
    """
    hetero_data = HeteroData()
    
    # Extract components
    ag_data = complex_data['antigen']
    ab_data = complex_data['antibody']
    interaction_data = complex_data['interactions']
    
    # Add antigen residue data only
    hetero_data['ag_res'].x = ag_data['residue_graph'].x
    hetero_data['ag_res'].plm = ag_data['plm_embeddings']
    hetero_data['ag_res'].y = ag_data['epitope_labels']
    hetero_data['ag_res'].pos = ag_data['coordinates']
    
    # Add antibody residue data only
    hetero_data['ab_res'].x = ab_data['residue_graph'].x
    hetero_data['ab_res'].plm = ab_data['plm_embeddings']
    hetero_data['ab_res'].y = ab_data['paratope_labels']
    hetero_data['ab_res'].pos = ab_data['coordinates']
    
    # Skip edge graphs - residue-only tensors

    # Add edge_lists to residue nodes for EdgeMP processing
    if hasattr(ag_data['residue_graph'], 'edge_lists'):
        hetero_data['ag_res'].edge_lists = ag_data['residue_graph'].edge_lists
    if hasattr(ab_data['residue_graph'], 'edge_lists'):
        hetero_data['ab_res'].edge_lists = ab_data['residue_graph'].edge_lists

    # Add edge_features to residue nodes for EdgeMP processing
    if hasattr(ag_data['residue_graph'], 'edge_features'):
        # Convert edge_features dict to a more DataLoader-friendly format
        edge_features_list = []
        edge_keys_list = []
        for key, feat in ag_data['residue_graph'].edge_features.items():
            edge_keys_list.append(list(key))  # Convert tuple to list
            edge_features_list.append(feat)
        
        if edge_features_list:
            hetero_data['ag_res'].edge_features_tensor = torch.stack(edge_features_list)
            hetero_data['ag_res'].edge_keys_tensor = torch.tensor(edge_keys_list, dtype=torch.long)

    if hasattr(ab_data['residue_graph'], 'edge_features'):
        # Convert edge_features dict to a more DataLoader-friendly format
        edge_features_list = []
        edge_keys_list = []
        for key, feat in ab_data['residue_graph'].edge_features.items():
            edge_keys_list.append(list(key))  # Convert tuple to list
            edge_features_list.append(feat)
        
        if edge_features_list:
            hetero_data['ab_res'].edge_features_tensor = torch.stack(edge_features_list)
            hetero_data['ab_res'].edge_keys_tensor = torch.tensor(edge_keys_list, dtype=torch.long)
    
    # Skip atom bonds - residue-only tensors
    
    # Add residue relations with attributes
    for rel_type in [0, 1, 2, 3]:
        rel_name = f"r{rel_type}"
        hetero_data = add_residue_relation(
            hetero_data,
            ag_data['residue_graph'],
            'ag_res',
            rel_type,
            rel_name
        )
        hetero_data = add_residue_relation(
            hetero_data,
            ab_data['residue_graph'],
            'ab_res',
            rel_type,
            rel_name
        )
    
    # Skip edge graph connections and cross connections - residue-only tensors
    
    # Add interaction edges
    if 'interaction_pairs' in interaction_data and interaction_data['interaction_pairs'].numel() > 0:
        hetero_data['ag_res', 'interacts', 'ab_res'].edge_index = interaction_data['interaction_pairs']
    
    # Add metadata
    hetero_data.complex_id = complex_data['complex_id']
    
    # Remove atom-related node stores completely for residue-only tensors
    if 'ag_atom' in hetero_data.node_types:
        del hetero_data['ag_atom']
    if 'ab_atom' in hetero_data.node_types:
        del hetero_data['ab_atom']
    
    # Remove edge graph node stores for residue-only tensors
    if 'ag_edge' in hetero_data.node_types:
        del hetero_data['ag_edge']
    if 'ab_edge' in hetero_data.node_types:
        del hetero_data['ab_edge']
    
    # Clean up any empty node stores
    clean_hetero_data(hetero_data)
    
    # Initialize missing components
    initialize_missing_components(hetero_data)
    
    return hetero_data


def add_cross_connections(hetero_data, protein_data, prefix):
    """
    Add cross connections between different hierarchy levels - residue only
    """
    # Skip atom connections - residue-only tensors
    
    # Residue to edge graph connections using tensor-based mapping
    if f'{prefix}_edge' in hetero_data.node_types and hasattr(hetero_data[f'{prefix}_edge'], 'edge_mapping'):
        edge_mapping = hetero_data[f'{prefix}_edge'].edge_mapping
        
        # Create connections: each edge node connects to its source and destination residues
        if edge_mapping.numel() > 0:
            # edge_mapping shape: [num_edge_nodes, 3] where columns are [src, dst, rel]
            edge_nodes = torch.arange(edge_mapping.size(0))
            
            # Connect to source residues
            src_connections = torch.stack([edge_mapping[:, 0], edge_nodes])  # [residue_idx, edge_idx]
            # Connect to destination residues  
            dst_connections = torch.stack([edge_mapping[:, 1], edge_nodes])  # [residue_idx, edge_idx]
            
            # Combine both connections
            res_to_edge = torch.cat([src_connections, dst_connections], dim=1)
            hetero_data[f'{prefix}_res', 'connected_by', f'{prefix}_edge'].edge_index = res_to_edge
    
    return hetero_data




def add_edges_with_attrs(hetero_data, graph_data, edge_type, attr_names):
    """Add edges with attributes to HeteroData"""
    for attr in attr_names:
        if hasattr(graph_data, attr) and getattr(graph_data, attr) is not None:
            hetero_data[edge_type][attr] = getattr(graph_data, attr)
    return hetero_data

def add_residue_relation(hetero_data, residue_graph, prefix, rel_type, rel_name):
    """Add residue relation edges with attributes"""
    if rel_type in residue_graph.edge_lists:
        edges = residue_graph.edge_lists[rel_type]
        if edges:
            edge_index = torch.tensor(edges).t().contiguous()
            hetero_data[prefix, rel_name, prefix].edge_index = edge_index
            
            # Add edge attributes if available
            edge_attrs = []
            for edge in edges:
                edge_key = (edge[0], edge[1], rel_type)
                if edge_key in residue_graph.edge_features:
                    edge_attrs.append(residue_graph.edge_features[edge_key])
            
            if edge_attrs:
                hetero_data[prefix, rel_name, prefix].edge_attr = torch.stack(edge_attrs)
    return hetero_data



def clean_hetero_data(hetero_data):
    """Remove empty node stores from HeteroData"""
    for node_type in list(hetero_data.node_types):
        store = hetero_data[node_type]
        if store.num_nodes == 0:
            # Find and remove all associated edges
            for edge_type in list(hetero_data.edge_types):
                if node_type in edge_type:
                    del hetero_data[edge_type]
            # Remove the node store
            del hetero_data[node_type]


def initialize_missing_components(hetero_data):
    # Define expected edge types - residue only
    expected_edge_types = [
        ('ag_res', 'r0', 'ag_res'),
        ('ag_res', 'r1', 'ag_res'),
        ('ag_res', 'r2', 'ag_res'),
        ('ag_res', 'r3', 'ag_res'),
        ('ab_res', 'r0', 'ab_res'),
        ('ab_res', 'r1', 'ab_res'),
        ('ab_res', 'r2', 'ab_res'),
        ('ab_res', 'r3', 'ab_res'),
        ('ag_res', 'interacts', 'ab_res')
    ]

    # Initialize missing edge types
    for edge_type in expected_edge_types:
        if edge_type not in hetero_data.edge_types:
            hetero_data[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)
            # Add edge_attr if required by your model
            if edge_type[1] in ['r0', 'r1', 'r2', 'r3']:
                hetero_data[edge_type].edge_attr = torch.empty((0, 100), dtype=torch.float)
    
    return hetero_data



# ==================== DATASET CREATION ====================

class HierarchicalDatasetCreator:
    """Creates hierarchical dataset combining AsEP data with PDB structures"""
    
    def __init__(self, 
                 asep_data_path: str,
                 ag_pdb_dir: str,
                 ab_pdb_dir: str,
                 output_path: str,
                 antiberty_path: str,
                 ag_plm_embeddings=None):
        """
        Initialize the dataset creator.
        
        Args:
            asep_data_path: Path to AsEP preprocessed data pickle file
            ag_pdb_dir: Directory containing antigen PDB files (surf)
            ab_pdb_dir: Directory containing antibody PDB files (cdr)
            output_path: Path to save the processed dataset
            antiberty_path: Path to antibody PLM embeddings
            ag_plm_embeddings: Optional custom antigen PLM embeddings
        """
        self.asep_data_path = asep_data_path
        self.ag_pdb_dir = ag_pdb_dir
        self.ab_pdb_dir = ab_pdb_dir
        self.output_path = output_path
        self.antiberty_path = antiberty_path
        self.ag_plm_embeddings = ag_plm_embeddings
        
        # Initialize graph builders - residue only
        self.residue_builder = ResidueGraphBuilder()
        
        # AA mapping for sequence processing
        self.AA_MAP = {
            "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G",
            "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
            "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V",
            "TRP": "W", "TYR": "Y"
        }

    def load_asep_data(self):
        """Load and process AsEP dataset"""
        print(f"Loading AsEP data from {self.asep_data_path}")
        
        # Load original AsEP graphs
        asep_graphs = torch.load(self.asep_data_path)
        
        return asep_graphs

    def extract_plm_embeddings(self, asep_graphs, pdb_id, antiberty_embeddings):
        """Extract PLM embeddings from AsEP data with configurable antigen PLM source"""
        if pdb_id in asep_graphs:
            # Antigen PLM - use custom embeddings if provided, else default
            if self.ag_plm_embeddings and pdb_id in self.ag_plm_embeddings:
                ag_plm = self.ag_plm_embeddings[pdb_id]
            else:
                ag_plm = asep_graphs[pdb_id]["x_g"].numpy()  # Default ESM2-35M
            
            # Antibody PLM (unchanged)
            if pdb_id in antiberty_embeddings:
                ab_plm = antiberty_embeddings[pdb_id]
            else:
                ab_plm = asep_graphs[pdb_id]["x_b"].numpy()    
            
            return {
                'antigen': torch.tensor(ag_plm, dtype=torch.float),
                'antibody': torch.tensor(ab_plm, dtype=torch.float)
            }
        else:
            print(f"Warning: PLM embeddings not found for {pdb_id}")
            return None

    def extract_labels_and_interactions(self, asep_graphs, pdb_id):
        """Extract epitope/paratope labels and interaction matrix"""
        if pdb_id in asep_graphs:
            data = asep_graphs[pdb_id]
            
            # Extract labels
            epitope_labels = data.get("y_g", torch.zeros(data["x_g"].shape[0]))
            paratope_labels = data.get("y_b", torch.zeros(data["x_b"].shape[0]))
            
            # Extract bipartite interaction matrix from edge_index_bg
            if "edge_index_bg" in data:
                edge_index_bg = data["edge_index_bg"]
                # Convert to dense bipartite matrix
                n_ag = data["x_g"].shape[0]
                n_ab = data["x_b"].shape[0]
                bipartite_matrix = torch.zeros(n_ag, n_ab)
                
                if edge_index_bg.shape[1] > 0:
                    ab_indices = edge_index_bg[0]  # First row: antibody indices
                    ag_indices = edge_index_bg[1]  # Second row: antigen indices
                    bipartite_matrix[ag_indices, ab_indices] = 1
            else:
                n_ag = data["x_g"].shape[0] 
                n_ab = data["x_b"].shape[0]
                bipartite_matrix = torch.zeros(n_ag, n_ab)
            
            return {
                'epitope_labels': epitope_labels,
                'paratope_labels': paratope_labels,
                'bipartite_matrix': bipartite_matrix,
                'interaction_pairs': data.get("edge_index_bg", torch.zeros((2, 0)))
            }
        else:
            print(f"Warning: Labels not found for {pdb_id}")
            return None

    def process_complex(self, pdb_id: str, asep_graphs: dict, antiberty_embeddings):
        """Process a single antibody-antigen complex into HeteroData"""
        print(f"Processing complex: {pdb_id}")
        
        # File paths
        ag_pdb_path = os.path.join(self.ag_pdb_dir, f"{pdb_id}_surf.pdb")
        ab_pdb_path = os.path.join(self.ab_pdb_dir, f"{pdb_id}_cdr.pdb")
        
        
        # Build residue graphs only
        ag_residue_graph = self.residue_builder.build_residue_graph(ag_pdb_path)
        ab_residue_graph = self.residue_builder.build_residue_graph(ab_pdb_path)
        
        # Create minimal edge graphs for residue-only tensors
        ag_edge_graph = Data(x=None, edge_index=torch.zeros((2, 0), dtype=torch.long))
        ab_edge_graph = Data(x=None, edge_index=torch.zeros((2, 0), dtype=torch.long))
        
        # Extract PLM embeddings
        plm_embeddings = self.extract_plm_embeddings(asep_graphs, pdb_id, antiberty_embeddings)
        if plm_embeddings is None:
            return None
        
        # Extract labels and interactions
        labels_interactions = self.extract_labels_and_interactions(asep_graphs, pdb_id)
        if labels_interactions is None:
            return None
        
        # Create residue-only data structure
        complex_data = {
            'complex_id': pdb_id,
            'antigen': {
                'residue_graph': ag_residue_graph,
                'edge_graph': ag_edge_graph,
                'plm_embeddings': plm_embeddings['antigen'],
                'epitope_labels': labels_interactions['epitope_labels'],
                'coordinates': ag_residue_graph.pos
            },
            'antibody': {
                'residue_graph': ab_residue_graph,
                'edge_graph': ab_edge_graph,
                'plm_embeddings': plm_embeddings['antibody'],
                'paratope_labels': labels_interactions['paratope_labels'],
                'coordinates': ab_residue_graph.pos
            },
            'interactions': {
                'bipartite_matrix': labels_interactions['bipartite_matrix'],
                'interaction_pairs': labels_interactions['interaction_pairs']
            }
        }
        
        # Build HeteroData object
        return build_hetero_data(complex_data)
            
        # except Exception as e:
        #     print(f"Error processing {pdb_id}: {e}")
        #     return None
        

    def create_dataset(self, max_examples: Optional[int] = None):
        """
        - this function checks if the dataset already exists, and resumes the graph construction
        """

        """Create the complete hierarchical dataset as HeteroData objects with resume capability"""
        print("Creating hierarchical dataset...")
        
        # Load AsEP data
        asep_graphs = self.load_asep_data()
        antiberty_embeddings = torch.load(self.antiberty_path)

        # Get list of PDB IDs
        pdb_ids = list(asep_graphs.keys())
        pdb_ids.remove("5nj6_0P")
        pdb_ids.remove("5ies_0P")

        # Check for existing dataset
        processed_data = []
        existing_ids = set()
        if os.path.exists(self.output_path):
            try:
                processed_data = torch.load(self.output_path)
                existing_ids = {data.complex_id for data in processed_data}
                print(f"Found existing dataset with {len(processed_data)} complexes. Resuming...")
            except Exception as e:
                print(f"Error loading existing dataset: {e}. Starting fresh.")
                processed_data = []

        # Filter out already processed complexes
        remaining_ids = [pid for pid in pdb_ids if pid not in existing_ids]
        print(f"Total complexes to process: {len(remaining_ids)}")
        
        if max_examples:
            remaining_ids = remaining_ids[:max_examples]

        successful_count = 0
        total_count = len(remaining_ids)
        
        for i, pdb_id in enumerate(remaining_ids):
            print(f"\nProcessing {i+1}/{total_count}: {pdb_id}")
            
            hetero_data = self.process_complex(pdb_id, asep_graphs, antiberty_embeddings)
            if hetero_data is not None:
                processed_data.append(hetero_data)
                successful_count += 1
                
                # Save periodically (every 10 complexes)
                if successful_count % 10 == 0:
                    print(f"Temporary save after {successful_count} complexes")
                    torch.save(processed_data, self.output_path)
        
        print(f"\nDataset creation complete!")
        print(f"Successfully processed: {successful_count}/{total_count} complexes")
        
        # Final save
        print(f"Saving dataset to {self.output_path}")
        torch.save(processed_data, self.output_path)
        
        return processed_data



# ==================== UTILITY FUNCTIONS ====================

def load_plm_embeddings(embedding_path, embedding_key=None):
    """Load PLM embeddings from file with optional key selection"""
    embeddings = torch.load(embedding_path, map_location='cpu')
    if embedding_key:
        return embeddings[embedding_key]  # For esm2 file with multiple models
    return embeddings  # For esm3 file with single model

def load_hierarchical_dataset(dataset_path: str):
    """Load the processed hierarchical dataset"""
    return torch.load(dataset_path)



# ==================== MAIN EXECUTION ====================

def create_residue_graph_tensor(asep_data_path, ag_pdb_dir, ab_pdb_dir, antiberty_path, 
                                output_path, ag_plm_embeddings=None):
    """Create residue graph tensor with specified antigen PLM embeddings"""
    dataset_creator = HierarchicalDatasetCreator(
        asep_data_path=asep_data_path,
        ag_pdb_dir=ag_pdb_dir,
        ab_pdb_dir=ab_pdb_dir,
        output_path=output_path,
        antiberty_path=antiberty_path,
        ag_plm_embeddings=ag_plm_embeddings
    )
    return dataset_creator.create_dataset()

if __name__ == "__main__":
    sys.path.append(os.path.join(os.getcwd(), '../../'))

    # Configuration paths
    # Note: Adjust proj_dir based on your directory structure
    # Default assumes data is at {proj_dir}/data/asep/
    proj_dir = os.path.join(os.getcwd(), '../../')
    asep_data_path = os.path.join(proj_dir, "data/asep/processed/dict_pre_cal.pt")
    ag_pdb_dir = os.path.join(proj_dir, "data/asep/antigen/atmseq2surf")
    ab_pdb_dir = os.path.join(proj_dir, "data/asep/antibody/atmseq2cdr")
    antiberty_path = os.path.join(proj_dir, "data/asep/antibody/antiberty_embeddings/asep_antiberty_embeddings.pt")
    
    # PLM embedding paths
    ag_esm2_path = os.path.join(proj_dir, "data/asep/antigen/plm_embeddings/ag_esm2_embeddings_asep.pt")
    ag_esm3_path = os.path.join(proj_dir, "data/asep/antigen/plm_embeddings/ag_esm3_embeddings_asep.pt")
    
    # Output paths
    output_paths = [
        os.path.join(proj_dir, "data/asep/m3epi/res_graph_tensor_esm2_35m.pkl"),
        os.path.join(proj_dir, "data/asep/m3epi/res_graph_tensor_esm2_650m.pkl"),
        os.path.join(proj_dir, "data/asep/m3epi/res_graph_tensor_esm2_3b.pkl"),
        os.path.join(proj_dir, "data/asep/m3epi/res_graph_tensor_esm3_small.pkl")
    ]
    
    # Load all PLM embeddings once
    print("Loading PLM embeddings...")
    ag_esm2_embeddings = load_plm_embeddings(ag_esm2_path)
    ag_esm3_embeddings = load_plm_embeddings(ag_esm3_path)
    
    # Create output directory
    os.makedirs(os.path.dirname(output_paths[0]), exist_ok=True)
    
    # Dataset configurations: (output_path, embeddings, description)
    configs = [
        (output_paths[0], None, "ESM2-35M (default)"),
        (output_paths[1], ag_esm2_embeddings['esm2_650m'], "ESM2-650M"),
        (output_paths[2], ag_esm2_embeddings['esm2_3b'], "ESM2-3B"),
        (output_paths[3], ag_esm3_embeddings['esm3_small'], "ESM3-small")
    ]
    
    # Create 4 datasets sequentially
    for output_path, embeddings, description in configs:
        print(f"\n=== Creating {description} dataset ===")
        print(f"Output: {output_path.split('/')[-1]}")
        
        dataset = create_residue_graph_tensor(
            asep_data_path, ag_pdb_dir, ab_pdb_dir, antiberty_path, 
            output_path, embeddings
        )
        print(f"✓ Successfully created {len(dataset)} complexes for {description}")
    
    print(f"\n All 4 residue graph tensors created successfully!")






"""
NOTE:
- two complexes are excluded from the asep data:
1. 5nj6_0P.pdb 
2. 5ies_0P.pdb because its seqres2cdr_seq and atmseq2cdr have different lengths

"""

"""
example usage:

python data/construct_res_graphs_tensor.py

nohup python data/graph_construction.py \
    > graph_construct_output.log 2>&1 &

"""







"""

# test
def validate_heterodata(data):
    for chain in ['ag', 'ab']:
        res_key = f'{chain}_res'
        if res_key in data:
            for rel, edges in data[res_key].edge_lists.items():
                if isinstance(edges, torch.Tensor) and edges.nelement() == 0:
                    print(f"Warning: Empty edge list in {data['complex_id']} {res_key} relation {rel}")
                    # Add default self-loop edges
                    num_nodes = data[res_key].x.shape[0]
                    data[res_key].edge_lists[rel] = torch.stack([
                        torch.arange(num_nodes),
                        torch.arange(num_nodes)
                    ])

"""


