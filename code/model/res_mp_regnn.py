"""
Simplified Relation-Aware EGNN for Residue-level Message Passing (ResMP)
Compatible with existing HeteroData structure and dimensions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")


class RelationAwareEGNNLayer(nn.Module):
    """Simplified relation-aware EGNN layer compatible with existing architecture"""
    
    def __init__(self, node_dim, edge_dim, hidden_dim, num_relations=4, 
                 act_fn=nn.SiLU(), residual=True, normalize=False, 
                 coords_agg='mean', tanh=False):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.residual = residual
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        
        # Relation-specific message MLPs - simplified to use only standard inputs
        self.relation_message_mlps = nn.ModuleDict()
        self.relation_coord_mlps = nn.ModuleDict()
        
        for r in range(num_relations):
            # Message MLP for each relation: φ_m^(r)
            # Input: h[row] + h[col] + edge_feat + rbf_dist = node_dim*2 + edge_dim + 16
            self.relation_message_mlps[str(r)] = nn.Sequential(
                nn.Linear(2 * node_dim + edge_dim + 16, hidden_dim),
                act_fn,
                nn.Linear(hidden_dim, hidden_dim),
                act_fn
            )
            
            # Coordinate MLP for each relation: φ_x^(r) (outputs scalar)
            coord_layers = [
                nn.Linear(hidden_dim, hidden_dim),
                act_fn,
                nn.Linear(hidden_dim, 1, bias=False)
            ]
            if tanh:
                coord_layers.append(nn.Tanh())
            self.relation_coord_mlps[str(r)] = nn.Sequential(*coord_layers)
        
        # Shared node update MLP: φ_h
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, node_dim)
        )
        
        # RBF parameters for distance encoding
        self.rbf_centers = nn.Parameter(torch.linspace(0, 20, 16), requires_grad=False)
        self.rbf_width = 1.0
        
    def compute_rbf(self, distances):
        """Compute radial basis function encoding γ(d_ij)"""
        rbf = torch.exp(-0.5 * ((distances.unsqueeze(-1) - self.rbf_centers) / self.rbf_width) ** 2)
        return rbf
    
    def forward(self, h, x, hetero_data, node_type):
        """
        Forward pass of relation-aware EGNN layer
        
        Args:
            h: Node features (num_nodes, node_dim)
            x: Node coordinates (num_nodes, 3) - CA positions
            hetero_data: HeteroData object containing edge information
            node_type: 'ag_res' or 'ab_res'
            
        Returns:
            h_new: Updated node features
            x_new: Updated coordinates
        """
        num_nodes = h.shape[0]
        
        # Aggregate messages by relation (relation-aware processing)
        total_messages = torch.zeros(num_nodes, self.hidden_dim, device=h.device)
        coord_updates = torch.zeros_like(x)
        
        for rel in range(4):  # Relations r0-r3
            edge_type = (node_type, f'r{rel}', node_type)
            
            # Skip if this edge type doesn't exist or has no edges
            if edge_type not in hetero_data.edge_types:
                continue
                
            edge_index = hetero_data[edge_type].edge_index
            if edge_index.size(1) == 0:
                continue
                
            # Get edge attributes
            edge_attr = hetero_data[edge_type].edge_attr
            
            row, col = edge_index
            
            # Compute coordinate differences and distances
            coord_diff = x[row] - x[col]  # δ_ij
            distances = torch.norm(coord_diff, dim=1, keepdim=True)  # d_ij
            
            if self.normalize:
                coord_diff = coord_diff / (distances + self.epsilon)
            
            # Compute RBF distance encoding
            rbf_dist = self.compute_rbf(distances.squeeze())
            
            # Prepare message input for this specific relation
            message_input = torch.cat([
                h[row],           # source node features
                h[col],           # target node features  
                edge_attr,        # edge features
                rbf_dist          # RBF distance encoding
            ], dim=1)
            
            # Compute relation-specific messages: m_ij^(r)
            rel_str = str(rel)
            messages = self.relation_message_mlps[rel_str](message_input)
            
            # Compute relation-specific coordinate scaling: s_ij^(r)
            coord_weights = self.relation_coord_mlps[rel_str](messages)
            
            # Aggregate messages to nodes
            rel_messages = scatter_add(messages, col, dim=0, dim_size=num_nodes)
            total_messages += rel_messages
            
            # Aggregate coordinate updates
            coord_update = coord_diff * coord_weights
            if self.coords_agg == 'mean':
                # Normalize by degree for each node
                degree = scatter_add(torch.ones_like(col, dtype=torch.float), col, 
                                   dim=0, dim_size=num_nodes).unsqueeze(1)
                coord_update = coord_update / (degree[col] + self.epsilon)
            
            rel_coord_updates = scatter_add(coord_update, col, dim=0, dim_size=num_nodes)
            coord_updates += rel_coord_updates
        
        # Update node features using aggregated messages from all relations
        node_input = torch.cat([h, total_messages], dim=1)
        h_new = self.node_mlp(node_input)
        
        if self.residual:
            h_new = h + h_new
            
        # Update coordinates  
        x_new = x + coord_updates
        
        return h_new, x_new


class RelationAwareResMP(nn.Module):
    """
    Simplified Relation-Aware Residue-level Message Passing module
    Uses relation-specific parameters while maintaining compatibility
    """
    
    def __init__(self, node_dim=105, edge_dim=100, hidden_dim=128, num_layers=4,
                 num_relations=4, act_fn=nn.SiLU(), residual=True, 
                 normalize=False, dropout=0.1, layer_norm=True):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_relations = num_relations
        self.layer_norm = layer_norm
        
        # Input projection to working dimension
        self.node_proj_in = nn.Linear(node_dim, hidden_dim)
        
        # Relation-aware EGNN layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                RelationAwareEGNNLayer(
                    node_dim=hidden_dim,
                    edge_dim=edge_dim,
                    hidden_dim=hidden_dim,
                    num_relations=num_relations,
                    act_fn=act_fn,
                    residual=residual,
                    normalize=normalize
                )
            )
        
        # Layer normalization for each layer
        if layer_norm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim) for _ in range(num_layers)
            ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Output projection
        self.node_proj_out = nn.Linear(hidden_dim, node_dim)
    
    def forward(self, hetero_data, node_type):
        """
        Process residue-level interactions directly from HeteroData
        
        Args:
            hetero_data: PyG HeteroData object
            node_type: 'ag_res' or 'ab_res'
        """
        # Extract data directly from HeteroData
        h = self.node_proj_in(hetero_data[node_type].x)
        x = hetero_data[node_type].pos.clone()
        
        # Process through relation-aware layers
        for i, layer in enumerate(self.layers):
            h, x = layer(h, x, hetero_data, node_type)
            if self.layer_norm:
                h = self.layer_norms[i](h)
            if self.dropout:
                h = self.dropout(h)
        
        # Update HeteroData
        hetero_data[node_type].x = self.node_proj_out(h)
        hetero_data[node_type].pos = x
        
        return h, x


def test_relation_aware_res_mp():
    """Test RelationAwareResMP with HeteroData structure"""
    print("Testing RelationAwareResMP with HeteroData...")
    
    from torch_geometric.data import HeteroData
    
    # Create dummy HeteroData
    hetero_data = HeteroData()
    
    # Add residue nodes
    num_residues = 20
    hetero_data['ag_res'].x = torch.randn(num_residues, 105)
    hetero_data['ag_res'].pos = torch.randn(num_residues, 3)
    
    # Add residue relations
    for rel in range(4):
        edge_type = ('ag_res', f'r{rel}', 'ag_res')
        num_edges = 10 + rel * 5  # Different number of edges per relation
        
        # Create edge index
        edge_index = torch.randint(0, num_residues, (2, num_edges))
        hetero_data[edge_type].edge_index = edge_index
        
        # Create edge attributes
        hetero_data[edge_type].edge_attr = torch.randn(num_edges, 100)
    
    # Create RelationAwareResMP
    regnn_res_mp = RelationAwareResMP(
        node_dim=105,
        edge_dim=100,
        hidden_dim=128,
        num_layers=2,
        num_relations=4
    )
    
    # Test forward pass
    print(f"Input features shape: {hetero_data['ag_res'].x.shape}")
    print(f"Input coordinates shape: {hetero_data['ag_res'].pos.shape}")
    
    h_out, x_out = regnn_res_mp(hetero_data, 'ag_res')
    
    print(f"✓ RelationAwareResMP forward pass successful!")
    print(f"Output features: {h_out.shape}")
    print(f"Output coordinates: {x_out.shape}")
    
    # Calculate coordinate changes
    original_pos = torch.randn(num_residues, 3)  # Original positions for comparison
    pos_diff = torch.norm(original_pos - x_out, dim=1)
    print(f"Coordinate changes - mean: {pos_diff.mean():.4f}, max: {pos_diff.max():.4f}")
    
    return regnn_res_mp, h_out, x_out


if __name__ == "__main__":
    test_relation_aware_res_mp()


# """
# Relation-Aware ResMP (REGNN) using multi-relational message passing
# Processes pre-constructed residue graphs from create_dataset.py
# with multiple edge relations while maintaining E(3)-equivariance.
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.data import Data
# from typing import Dict, List, Optional, Tuple
# import warnings
# warnings.filterwarnings("ignore")


# def unsorted_segment_sum(data, segment_ids, num_segments):
#     """Segment sum operation for aggregating edge features to nodes"""
#     expand_dims = tuple(data.shape[1:])
#     result_shape = (num_segments,) + expand_dims
#     for _ in expand_dims:
#         segment_ids = segment_ids.unsqueeze(-1)
#     segment_ids = segment_ids.expand(-1, *expand_dims)
#     result = data.new_full(result_shape, 0)
#     result.scatter_add_(0, segment_ids, data)
#     return result


# def unsorted_segment_mean(data, segment_ids, num_segments):
#     """Segment mean operation for aggregating coordinate updates"""
#     expand_dims = tuple(data.shape[1:])
#     result_shape = (num_segments,) + expand_dims
#     for _ in expand_dims:
#         segment_ids = segment_ids.unsqueeze(-1)
#     segment_ids = segment_ids.expand(-1, *expand_dims)
#     result = data.new_full(result_shape, 0)
#     count = data.new_full(result_shape, 0)
#     result.scatter_add_(0, segment_ids, data)
#     count.scatter_add_(0, segment_ids, torch.ones_like(data))
#     return result / count.clamp(min=1)


# def coord2radial(edge_index, coord):
#     """
#     Compute radial features from coordinates
#     Adapted for single-channel coordinates (CA only)
#     """
#     row, col = edge_index
#     coord_diff = coord[row] - coord[col]  # [n_edge, 3]
    
#     # For single-channel, compute scalar distance
#     radial = torch.norm(coord_diff, dim=1, keepdim=True)  # [n_edge, 1]
    
#     return radial, coord_diff


# class RelationAwareMPNNLayer(nn.Module):
#     """Single layer of Relation-Aware MPNN adapted for separate protein graphs"""
    
#     def __init__(self, input_nf, output_nf, hidden_nf, dropout=0.1, 
#                  edges_in_d=100, num_relations=4):
#         super().__init__()
#         self.num_relations = num_relations
#         self.dropout = nn.Dropout(dropout)
        
#         # Message MLP - processes node and edge features
#         self.message_mlp = nn.Sequential(
#             nn.Linear(2 * input_nf + edges_in_d + 1, hidden_nf),  # +1 for radial distance
#             nn.ReLU(),
#             nn.Linear(hidden_nf, hidden_nf),
#             nn.ReLU()
#         )
        
#         # Node update MLP
#         self.node_mlp = nn.Sequential(
#             nn.Linear(input_nf + hidden_nf, hidden_nf),
#             nn.ReLU(),
#             nn.Linear(hidden_nf, output_nf)
#         )
        
#         # Coordinate update MLP
#         self.coord_mlp = nn.Sequential(
#             nn.Linear(hidden_nf, hidden_nf),
#             nn.ReLU(),
#             nn.Linear(hidden_nf, 1, bias=False)
#         )
        
#         # Edge update MLP  
#         self.edge_mlp = nn.Sequential(
#             nn.Linear(2 * output_nf + edges_in_d, hidden_nf),
#             nn.ReLU(),
#             nn.Linear(hidden_nf, edges_in_d)
#         )
    
#     def forward(self, h, coord, edge_attr_lists, edge_index_lists):
#         """
#         Forward pass of relation-aware MPNN layer
        
#         Args:
#             h: Node features [n_nodes, input_nf]
#             coord: Node coordinates [n_nodes, 3]
#             edge_attr_lists: List of edge attributes for each relation
#             edge_index_lists: List of edge indices for each relation
#         """
#         edge_feat_lists = []
#         coord_diff_lists = []
        
#         # Process each relation type
#         for i in range(len(edge_index_lists)):
#             if edge_index_lists[i].shape[1] > 0:
#                 # Compute radial features and coordinate differences
#                 radial, coord_diff = coord2radial(edge_index_lists[i], coord)
#                 coord_diff_lists.append(coord_diff)
                
#                 # Compute edge messages
#                 row, col = edge_index_lists[i]
#                 edge_input = torch.cat([h[row], h[col], edge_attr_lists[i], radial], dim=1)
#                 edge_feat = self.message_mlp(edge_input)
#                 edge_feat_lists.append(edge_feat)
#             else:
#                 # Handle empty edge lists
#                 coord_diff_lists.append(torch.empty(0, 3, device=coord.device))
#                 edge_feat_lists.append(torch.empty(0, h.shape[1], device=h.device))
        
#         # Update coordinates
#         coord_new = self.update_coords(coord, edge_index_lists, edge_feat_lists, coord_diff_lists)
        
#         # Update nodes
#         h_new = self.update_nodes(h, edge_index_lists, edge_feat_lists)
        
#         # Update edges
#         edge_attrs_new = self.update_edges(h_new, edge_index_lists, edge_attr_lists)
        
#         return h_new, coord_new, edge_attrs_new
    
#     def update_coords(self, coord, edge_index_lists, edge_feat_lists, coord_diff_lists):
#         """Update coordinates using edge features"""
#         coord_updates = torch.zeros_like(coord)
        
#         for i, (edge_index, edge_feats, coord_diff) in enumerate(zip(edge_index_lists, edge_feat_lists, coord_diff_lists)):
#             if edge_index.shape[1] > 0:
#                 row, col = edge_index
#                 coord_weights = self.coord_mlp(edge_feats)
#                 coord_update = coord_diff * coord_weights
#                 coord_updates += unsorted_segment_mean(coord_update, col, coord.shape[0])
        
#         return coord + coord_updates
    
#     def update_nodes(self, h, edge_index_lists, edge_feat_lists):
#         """Update node features by aggregating edge messages"""
#         total_messages = torch.zeros(h.shape[0], edge_feat_lists[0].shape[1], device=h.device)
        
#         for edge_index, edge_feats in zip(edge_index_lists, edge_feat_lists):
#             if edge_index.shape[1] > 0:
#                 row, col = edge_index
#                 messages = unsorted_segment_sum(edge_feats, col, h.shape[0])
#                 total_messages += messages
        
#         # Update nodes
#         node_input = torch.cat([h, total_messages], dim=1)
#         h_new = self.node_mlp(node_input)
        
#         return h_new
    
#     def update_edges(self, h_new, edge_index_lists, edge_attr_lists):
#         """Update edge attributes"""
#         edge_attrs_new = []
        
#         for edge_index, edge_attrs in zip(edge_index_lists, edge_attr_lists):
#             if edge_index.shape[1] > 0:
#                 row, col = edge_index
#                 edge_input = torch.cat([h_new[row], h_new[col], edge_attrs], dim=1)
#                 edge_attrs_updated = self.edge_mlp(edge_input)
#                 edge_attrs_new.append(edge_attrs_updated)
#             else:
#                 edge_attrs_new.append(edge_attrs)
        
#         return edge_attrs_new


# class RelationAwareResMP(nn.Module):
#     """
#     Relation-Aware ResMP using multi-relational EGNN
#     Processes pre-constructed residue graphs from create_dataset.py
#     """
    
#     def __init__(self, in_node_nf=105, hidden_nf=128, out_node_nf=105, 
#                  n_layers=4, dropout=0.1, edge_feats_dim=100, num_relations=4):
#         super().__init__()
#         self.n_layers = n_layers
#         self.num_relations = num_relations
#         self.dropout = nn.Dropout(dropout)
        
#         # Input/output projections
#         self.linear_in = nn.Linear(in_node_nf, hidden_nf)
#         self.linear_out = nn.Linear(hidden_nf, out_node_nf)
        
#         # Relation-aware MPNN layers
#         self.layers = nn.ModuleList()
#         for i in range(n_layers):
#             self.layers.append(
#                 RelationAwareMPNNLayer(
#                     input_nf=hidden_nf,
#                     output_nf=hidden_nf,
#                     hidden_nf=hidden_nf,
#                     dropout=dropout,
#                     edges_in_d=edge_feats_dim,
#                     num_relations=num_relations
#                 )
#             )
    
#     def forward(self, residue_graph, edge_features):
#         """
#         Forward method accepting pre-constructed PyG residue_graph from create_dataset.py
        
#         Args:
#             residue_graph: PyG Data object with:
#                 - x: Residue features (n_residues, node_dim)
#                 - pos: CA coordinates (n_residues, 3)
#                 - edge_lists: Dict mapping relation -> list of edges (already organized!)
#             edge_features: Dict mapping (src, dst, rel) -> edge features (from EdgeMP)
            
#         Returns:
#             h_out: Updated residue features (n_residues, node_dim)
#             x_out: Updated coordinates (n_residues, 3)
#         """
#         # Extract data from PyG graph
#         node_features = residue_graph.x
#         ca_coords = residue_graph.pos
#         edge_lists = residue_graph.edge_lists  # Already organized by relation!
        
#         # Project to working dimension
#         h = self.linear_in(node_features)
#         h = self.dropout(h)
#         x = ca_coords.clone()
        
#         # Prepare edge data for REGNN layers
#         edge_index_lists, edge_attr_lists = self.prepare_edge_data(edge_lists, edge_features, h.device)
        
#         # Process through layers
#         for layer in self.layers:
#             h, x, edge_attr_lists = layer(h, x, edge_attr_lists, edge_index_lists)
        
#         # Project to output dimension
#         h_out = self.dropout(h)
#         h_out = self.linear_out(h_out)
        
#         return h_out, x
    
#     def prepare_edge_data(self, edge_lists, edge_features, device):
#         """
#         Convert edge data format to lists required by RelationAwareMPNNLayer
        
#         Args:
#             edge_lists: Dict mapping relation -> list of [src, dst] edges (from dataset!)
#             edge_features: Dict mapping (src, dst, rel) -> edge features
#             device: Target device for tensors
            
#         Returns:
#             edge_index_lists: List of edge_index tensors for each relation
#             edge_attr_lists: List of edge attribute tensors for each relation
#         """
#         edge_index_lists = []
#         edge_attr_lists = []
        
#         for rel in range(self.num_relations):
#             if rel in edge_lists and len(edge_lists[rel]) > 0:
#                 # Convert to edge_index format
#                 edges = torch.tensor(edge_lists[rel], dtype=torch.long, device=device).t()
#                 edge_index_lists.append(edges)
                
#                 # Collect edge attributes for this relation
#                 edge_attrs = []
#                 for src, dst in edge_lists[rel]:
#                     edge_key = (src, dst, rel)
#                     if edge_key in edge_features:
#                         edge_attrs.append(edge_features[edge_key])
#                     else:
#                         # Fallback to zero features
#                         if edge_features:
#                             sample_edge = list(edge_features.values())[0]
#                             edge_attrs.append(torch.zeros_like(sample_edge))
#                         else:
#                             edge_attrs.append(torch.zeros(100, device=device))  # Default edge dim
                
#                 if edge_attrs:
#                     edge_attr_lists.append(torch.stack(edge_attrs))
#                 else:
#                     edge_attr_lists.append(torch.empty(0, 100, device=device))
#             else:
#                 # Empty relation
#                 edge_index_lists.append(torch.empty(2, 0, dtype=torch.long, device=device))
#                 edge_attr_lists.append(torch.empty(0, 100, device=device))
        
#         return edge_index_lists, edge_attr_lists


# def test_relation_aware_resmp():
#     """Test RelationAwareResMP with pre-constructed residue graph"""
#     print("Testing RelationAwareResMP with pre-constructed residue graph...")
    
#     # Create dummy residue graph as it would come from create_dataset.py
#     num_residues = 15
#     node_dim = 105
#     edge_dim = 64
    
#     # Create edge_lists as in your dataset
#     edge_lists = {
#         0: [[0, 1], [1, 2], [2, 3]],  # Sequential ±1
#         1: [[0, 2], [1, 3]],          # Sequential ±2
#         2: [[0, 4], [1, 5], [2, 6]],  # k-NN
#         3: [[0, 7], [1, 8]]           # Spatial
#     }
     
#     residue_graph = Data(
#         x=torch.randn(num_residues, node_dim),
#         pos=torch.randn(num_residues, 3),
#         edge_lists=edge_lists  
#     )
    
#     # Create dummy edge features (from EdgeMP)
#     edge_features = {}
#     for rel, edges in edge_lists.items():
#         for src, dst in edges:
#             edge_key = (src, dst, rel)
#             edge_features[edge_key] = torch.randn(edge_dim)
    
#     # Create RelationAwareResMP
#     model = RelationAwareResMP(
#         in_node_nf=node_dim,
#         hidden_nf=128,
#         out_node_nf=node_dim,
#         n_layers=2,
#         edge_feats_dim=edge_dim,
#         num_relations=4
#     )
    
#     # Test forward pass
#     h_out, x_out = model(residue_graph, edge_features)
    
#     print(f"✓ RelationAwareResMP forward pass successful!")
#     print(f"Input residues: {residue_graph.x.shape[0]}")
#     print(f"Output features: {h_out.shape}")
#     print(f"Output coordinates: {x_out.shape}")
    
#     # Calculate coordinate changes
#     pos_diff = torch.norm(residue_graph.pos - x_out, dim=1)
#     print(f"Coordinate changes - mean: {pos_diff.mean():.4f}, max: {pos_diff.max():.4f}")
    
#     return model, h_out, x_out


# if __name__ == "__main__":
#     test_relation_aware_resmp()







