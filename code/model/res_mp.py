import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.data import Data
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

class MultiRelationalEGNNLayer(nn.Module):
    """Single layer of multi-relational EGNN (EGNN-R)"""

    def __init__(self, node_dim, edge_dim, hidden_dim, num_relations=4,
                 act_fn=nn.SiLU(), residual=True, normalize=False,
                 coords_agg='mean', tanh=False, update_coords=True):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.residual = residual
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.update_coords = update_coords
        self.epsilon = 1e-8
        
        # Relation-specific message MLPs
        self.message_mlps = nn.ModuleDict()
        self.coord_mlps = nn.ModuleDict()
        
        for r in range(num_relations):
            # Message MLP: φ_m^(r) 
            # Input: h[row] + h[col] + edge_feat + rbf_dist = node_dim*2 + edge_dim + 16
            self.message_mlps[str(r)] = nn.Sequential(
                nn.Linear(2 * node_dim + edge_dim + 16, hidden_dim),  # +16 for RBF distance
                act_fn,
                nn.Linear(hidden_dim, hidden_dim),
                act_fn
            )
            
            # Coordinate MLP: φ_x^(r) (outputs scalar)
            coord_layers = [
                nn.Linear(hidden_dim, hidden_dim),
                act_fn,
                nn.Linear(hidden_dim, 1, bias=False)
            ]
            if tanh:
                coord_layers.append(nn.Tanh())
            self.coord_mlps[str(r)] = nn.Sequential(*coord_layers)
        
        # Node update MLP: φ_h (shared across relations)
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
        Forward pass of multi-relational EGNN layer
        
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
        
        # Aggregate messages by relation
        total_messages = torch.zeros_like(h)
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

            # print(len(edge_attr), len(edge_index[1]), edge_type) 
            # debug: the above lengths should be the same
            
            # Compute coordinate differences and distances
            coord_diff = x[row] - x[col]  # δ_ij
            distances = torch.norm(coord_diff, dim=1, keepdim=True)  # d_ij
            
            if self.normalize:
                coord_diff = coord_diff / (distances + self.epsilon)
            
            # Compute RBF distance encoding
            rbf_dist = self.compute_rbf(distances.squeeze())

            # Fix for RBF distance tensor dimensions
            if rbf_dist.dim() == 1:
                # If rbf_dist is 1D, expand it to match the batch dimension
                rbf_dist = rbf_dist.unsqueeze(0)  # Shape: [1, 16]
            
            # Prepare message input
            try:
                message_input = torch.cat([
                    h[row],           # source node features
                    h[col],           # target node features  
                    edge_attr,        # edge features
                    rbf_dist          # RBF distance encoding
                ], dim=1)
            except RuntimeError as e:
                print(f"[DEBUG ResMP] Message input concatenation error: {e}")
                print(f"[DEBUG ResMP] h[row] shape: {h[row].shape}")
                print(f"[DEBUG ResMP] h[col] shape: {h[col].shape}")  
                print(f"[DEBUG ResMP] edge_attr shape: {edge_attr.shape}")
                print(f"[DEBUG ResMP] rbf_dist shape: {rbf_dist.shape}")
                raise e
            
            # Compute messages: m_ij^(r)
            rel_str = str(rel)
            messages = self.message_mlps[rel_str](message_input)
            
            # Compute coordinate scaling: s_ij^(r)
            coord_weights = self.coord_mlps[rel_str](messages)
            
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
        
        # Update node features
        node_input = torch.cat([h, total_messages], dim=1)
        h_new = self.node_mlp(node_input)
        
        if self.residual:
            h_new = h + h_new

        # Update coordinates (can be disabled for ablation)
        x_new = x + coord_updates if self.update_coords else x

        return h_new, x_new


class ResMP(nn.Module):
    """
    Residue-level Message Passing module with multi-relational EGNN
    Processes HeteroData directly without requiring edge_lists
    """

    def __init__(self, node_dim=105, edge_dim=100, hidden_dim=128, num_layers=4,
                 num_relations=4, act_fn=nn.SiLU(), residual=True,
                 normalize=False, dropout=0.1, layer_norm=True, update_coords=True):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_relations = num_relations
        self.layer_norm = layer_norm

        # Input projection to working dimension
        self.node_proj_in = nn.Linear(node_dim, hidden_dim)

        # Multi-relational EGNN layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                MultiRelationalEGNNLayer(
                    node_dim=hidden_dim,
                    edge_dim=edge_dim,
                    hidden_dim=hidden_dim,
                    num_relations=num_relations,
                    act_fn=act_fn,
                    residual=residual,
                    normalize=normalize,
                    update_coords=update_coords
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
        
        # Process through layers
        for i, layer in enumerate(self.layers):
            h, x = layer(h, x, hetero_data, node_type)
            if self.layer_norm:
                h = self.layer_norms[i](h)
            if self.dropout:
                h = self.dropout(h)
        
        # Update HeteroData and return projected output
        h_out = self.node_proj_out(h)  # 128 -> 105
        hetero_data[node_type].x = h_out
        hetero_data[node_type].pos = x
        
        return h_out, x







def test_res_mp_with_hetero_data():
    """Test ResMP with HeteroData structure"""
    print("Testing ResMP with HeteroData...")
    
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
    
    # Create ResMP
    res_mp = ResMP(
        node_dim=105,
        edge_dim=100,
        hidden_dim=128,
        num_layers=2,
        num_relations=4
    )
    
    # Test forward pass
    h_out, x_out = res_mp(hetero_data, 'ag_res')
    
    print(f"✓ ResMP forward pass successful!")
    print(f"Input residues: {num_residues}")
    print(f"Output features: {h_out.shape}")
    print(f"Output coordinates: {x_out.shape}")
    
    # Calculate coordinate changes
    pos_diff = torch.norm(hetero_data['ag_res'].pos - x_out, dim=1)
    print(f"Coordinate changes - mean: {pos_diff.mean():.4f}, max: {pos_diff.max():.4f}")
    
    return res_mp, h_out, x_out


if __name__ == "__main__":
    test_res_mp_with_hetero_data()










# """
# Residue-level Message Passing (ResMP) using multi-relational EGNN
# Processes pre-constructed residue graphs from create_dataset.py
# with multiple edge relations while maintaining E(3)-equivariance.
# """
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_scatter import scatter_add
# from torch_geometric.data import Data
# from typing import Dict, List, Optional, Tuple
# import warnings
# warnings.filterwarnings("ignore")

# """
# TODO:  
# - perform ResMP with heterogenous residue graph using PyG HeteroData object
# """


# class MultiRelationalEGNNLayer(nn.Module):
#     """Single layer of multi-relational EGNN (EGNN-R)"""
    
#     def __init__(self, node_dim, edge_dim, hidden_dim, num_relations=4, 
#                  act_fn=nn.SiLU(), residual=True, normalize=False, 
#                  coords_agg='mean', tanh=False):
#         super().__init__()
#         self.node_dim = node_dim
#         self.edge_dim = edge_dim
#         self.hidden_dim = hidden_dim
#         self.num_relations = num_relations
#         self.residual = residual
#         self.normalize = normalize
#         self.coords_agg = coords_agg
#         self.tanh = tanh
#         self.epsilon = 1e-8
        
#         # Relation-specific message MLPs
#         self.message_mlps = nn.ModuleDict()
#         self.coord_mlps = nn.ModuleDict()
        
#         for r in range(num_relations):
#             # Message MLP: φ_m^(r) 
#             # Input: h[row] + h[col] + edge_feat + rbf_dist = node_dim*2 + edge_dim + 16
#             self.message_mlps[str(r)] = nn.Sequential(
#                 nn.Linear(2 * node_dim + edge_dim + 16, hidden_dim),  # +16 for RBF distance
#                 act_fn,
#                 nn.Linear(hidden_dim, hidden_dim),
#                 act_fn
#             )
            
#             # Coordinate MLP: φ_x^(r) (outputs scalar)
#             coord_layers = [
#                 nn.Linear(hidden_dim, hidden_dim),
#                 act_fn,
#                 nn.Linear(hidden_dim, 1, bias=False)
#             ]
#             if tanh:
#                 coord_layers.append(nn.Tanh())
#             self.coord_mlps[str(r)] = nn.Sequential(*coord_layers)
        
#         # Node update MLP: φ_h (shared across relations)
#         self.node_mlp = nn.Sequential(
#             nn.Linear(node_dim + hidden_dim, hidden_dim),
#             act_fn,
#             nn.Linear(hidden_dim, node_dim)
#         )
        
#         # RBF parameters for distance encoding
#         self.rbf_centers = nn.Parameter(torch.linspace(0, 20, 16), requires_grad=False)
#         self.rbf_width = 1.0
        
#     def compute_rbf(self, distances):
#         """Compute radial basis function encoding γ(d_ij)"""
#         rbf = torch.exp(-0.5 * ((distances.unsqueeze(-1) - self.rbf_centers) / self.rbf_width) ** 2)
#         return rbf
    
#     def forward(self, h, x, edge_lists, edge_features_dict):
#         """
#         Forward pass of multi-relational EGNN layer
        
#         Args:
#             h: Node features (num_nodes, node_dim)
#             x: Node coordinates (num_nodes, 3) - CA positions
#             edge_lists: Dict mapping relation -> list of [src, dst] edges (direct from dataset!)
#             edge_features_dict: Dict mapping (src, dst, rel) -> edge features
            
#         Returns:
#             h_new: Updated node features
#             x_new: Updated coordinates
#         """
#         num_nodes = h.shape[0]
        
#         # Aggregate messages by relation
#         total_messages = torch.zeros_like(h)
#         coord_updates = torch.zeros_like(x)
#         # print(edge_lists)
        
#         for rel, edges in edge_lists.items():
#             print(rel, len(edges[rel]))

#             """
#             FIXME: this part of the code handles when there are no edges 
#             for a particular relation type
#             """

#             # Skip if no edges or invalid edge list
#             if len(edges) == 0 or not isinstance(edges, list) or len(edges) < 2:
#                 print("Empty edge list: skipping edges for this relation..")
#                 continue
                
#             # # Convert to tensor with robust handling
#             # try:
#             #     edge_tensor = torch.tensor(edges, dtype=torch.long, device=h.device)
#             #     if edge_tensor.dim() == 1:
#             #         edge_tensor = edge_tensor.unsqueeze(0)
#             #     row, col = edge_tensor.t()
#             # except Exception as e:
#             #     print(f"Error processing edges for rel {rel}: {e}")
#             #     print(f"Edge list: {edges}")
#             #     continue
            
#             # if len(edges) == 0:
#             #     continue
                
#             # Convert to tensor (locally within this layer)

#             # print(len(edges[0]), len(edges[1]))

#             edge_tensor = torch.tensor(edges[rel], dtype=torch.long, device=h.device).t()
#             print(edge_tensor.shape)
#             row, col = edge_tensor

            
#             # Compute coordinate differences and distances
#             coord_diff = x[row] - x[col]  # δ_ij
#             distances = torch.norm(coord_diff, dim=1, keepdim=True)  # d_ij
            
#             if self.normalize:
#                 coord_diff = coord_diff / (distances + self.epsilon)
            
#             # Get edge features for this relation
#             edge_feats = []
#             for i, (src_idx, dst_idx) in enumerate(zip(row.tolist(), col.tolist())):
#                 edge_key = (src_idx, dst_idx, rel)
#                 if edge_key in edge_features_dict:
#                     edge_feats.append(edge_features_dict[edge_key])
#                 else:
#                     # Fallback to zero features if not found
#                     print("edge features not found...")
#                     edge_feats.append(torch.zeros(self.edge_dim, device=h.device))
            
#             if len(edge_feats) == 0:
#                 continue
                
#             edge_feats = torch.stack(edge_feats)
#             # print(edge_feats.shape)
            
#             # Compute RBF distance encoding
#             rbf_dist = self.compute_rbf(distances.squeeze())
            
#             # Prepare message input
#             message_input = torch.cat([
#                 h[row],           # source node features
#                 h[col],           # target node features  
#                 edge_feats,       # edge features from EdgeMP
#                 rbf_dist          # RBF distance encoding
#             ], dim=1)
            
#             # Compute messages: m_ij^(r)
#             rel_str = str(rel)
#             messages = self.message_mlps[rel_str](message_input)
            
#             # Compute coordinate scaling: s_ij^(r)
#             coord_weights = self.coord_mlps[rel_str](messages)
            
#             # Aggregate messages to nodes
#             rel_messages = scatter_add(messages, col, dim=0, dim_size=num_nodes)
#             total_messages += rel_messages
            
#             # Aggregate coordinate updates
#             coord_update = coord_diff * coord_weights
#             if self.coords_agg == 'mean':
#                 # Normalize by degree for each node
#                 degree = scatter_add(torch.ones_like(col, dtype=torch.float), col, 
#                                    dim=0, dim_size=num_nodes).unsqueeze(1)
#                 coord_update = coord_update / (degree[col] + self.epsilon)
            
#             rel_coord_updates = scatter_add(coord_update, col, dim=0, dim_size=num_nodes)
#             coord_updates += rel_coord_updates

#         print("update residue node features...")
        
#         # Update node features
#         node_input = torch.cat([h, total_messages], dim=1)
#         h_new = self.node_mlp(node_input)
        
#         if self.residual:
#             h_new = h + h_new
            
#         # Update coordinates  
#         x_new = x + coord_updates
        
#         return h_new, x_new


# class ResMP(nn.Module):
#     """
#     Residue-level Message Passing module with multi-relational EGNN
#     Processes pre-constructed residue graphs from create_dataset.py
#     """
    
#     def __init__(self, node_dim=105, edge_dim=100, hidden_dim=128, num_layers=4,
#                  num_relations=4, act_fn=nn.SiLU(), residual=True, 
#                  normalize=False, dropout=0.1, layer_norm=True):
#         super().__init__()
#         self.node_dim = node_dim
#         self.edge_dim = edge_dim
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#         self.num_relations = num_relations
#         self.layer_norm = layer_norm
        
#         # Input projection to working dimension
#         self.node_proj_in = nn.Linear(node_dim, hidden_dim)
        
#         # Multi-relational EGNN layers
#         self.layers = nn.ModuleList()
#         for _ in range(num_layers):
#             self.layers.append(
#                 MultiRelationalEGNNLayer(
#                     node_dim=hidden_dim,
#                     edge_dim=edge_dim,
#                     hidden_dim=hidden_dim,
#                     num_relations=num_relations,
#                     act_fn=act_fn,
#                     residual=residual,
#                     normalize=normalize
#                 )
#             )
        
#         # Layer normalization for each layer
#         if layer_norm:
#             self.layer_norms = nn.ModuleList([
#                 nn.LayerNorm(hidden_dim) for _ in range(num_layers)
#             ])
        
#         # Dropout
#         self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
#         # Output projection
#         self.node_proj_out = nn.Linear(hidden_dim, node_dim)
    


#     def forward(self, hetero_data, node_type):
#         """
#         Process residue-level interactions directly from HeteroData
        
#         Args:
#             hetero_data: PyG HeteroData object
#             node_type: 'ag_res' or 'ab_res'
#         """
#         # Extract data directly from HeteroData
#         h = self.node_proj_in(hetero_data[node_type].x)
#         x = hetero_data[node_type].pos.clone()
        
#         # Collect edge features by relation
#         edge_features = {}
#         for rel in range(4):  # Relations r0-r3
#             edge_type = (node_type, f'r{rel}', node_type)
#             if edge_type in hetero_data.edge_types:
#                 for i in range(hetero_data[edge_type].edge_index.size(1)):
#                     src = hetero_data[edge_type].edge_index[0, i].item()
#                     dst = hetero_data[edge_type].edge_index[1, i].item()
#                     # edge_features[(src, dst, rel)] = hetero_data[edge_type].x[i]
#                     edge_features[(src, dst, rel)] = hetero_data[edge_type].edge_attr[i]
        
#         # Process through layers
#         for i, layer in enumerate(self.layers):
#             h, x = layer(h, x, hetero_data[node_type].edge_lists, edge_features)
#             if self.layer_norm:
#                 h = self.layer_norms[i](h)
#             if self.dropout:
#                 h = self.dropout(h)
        
#         # Update HeteroData
#         hetero_data[node_type].x = self.node_proj_out(h)
#         hetero_data[node_type].pos = x
        
#         return h, x

# def test_res_mp():
#     """Test ResMP with pre-constructed residue graph"""
#     print("Testing ResMP with pre-constructed residue graph...")
    
#     # Create dummy residue graph as it would come from create_dataset.py
#     num_residues = 20
#     node_dim = 105
#     edge_dim = 64
    
#     residue_graph = Data(
#         x=torch.randn(num_residues, node_dim),
#         pos=torch.randn(num_residues, 3),
#         edge_index=torch.randint(0, num_residues, (2, 40)),
#         edge_attr=torch.randn(40, edge_dim)
#     )
    
#     # Create dummy edge features (from EdgeMP)
#     edge_features = {}
#     for i in range(residue_graph.edge_index.shape[1]):
#         src, dst = residue_graph.edge_index[0, i].item(), residue_graph.edge_index[1, i].item()
#         rel = i % 4  # Simple relation assignment
#         edge_key = (src, dst, rel)
#         edge_features[edge_key] = torch.randn(edge_dim)
    
#     # Create ResMP
#     res_mp = ResMP(
#         node_dim=node_dim,
#         edge_dim=edge_dim,
#         hidden_dim=128,
#         num_layers=2,
#         num_relations=4
#     )
    
#     # Test forward pass
#     h_out, x_out = res_mp(residue_graph)
    
#     print(f"✓ ResMP forward pass successful!")
#     print(f"Input residues: {residue_graph.x.shape[0]}")
#     print(f"Output features: {h_out.shape}")
#     print(f"Output coordinates: {x_out.shape}")
    
#     # Calculate coordinate changes
#     pos_diff = torch.norm(residue_graph.pos - x_out, dim=1)
#     print(f"Coordinate changes - mean: {pos_diff.mean():.4f}, max: {pos_diff.max():.4f}")
    
#     return res_mp, h_out, x_out


# if __name__ == "__main__":
#     test_res_mp()






