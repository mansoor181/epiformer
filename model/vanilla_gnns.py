"""
Vanilla GNN implementations for hierarchical encoder blocks.
Provides GCN, GAT, and GIN layers compatible with AtomMP, EdgeMP, and ResMP components.

These implementations follow PyG conventions while maintaining compatibility with the 
hierarchical architecture's expected input/output formats.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, MessagePassing
from torch_geometric.utils import add_self_loops, degree
from typing import Optional, Union


class VanillaGCN(nn.Module):
    """
    Multi-layer GCN for node feature learning.
    Compatible with AtomMP, EdgeMP, and ResMP blocks.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 use_layer_norm: bool = True):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm
        
        # Build layers
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        if use_layer_norm:
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, output_dim))
        else:
            # Single layer case
            self.convs[0] = GCNConv(input_dim, output_dim)
            
        if use_layer_norm:
            self.layer_norms.append(nn.LayerNorm(output_dim))

    def forward(self, x, edge_index, edge_attr=None, pos=None):
        """
        Forward pass compatible with hierarchical encoder expectations.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features (ignored for GCN)
            pos: Node positions (returned unchanged for compatibility)
            
        Returns:
            h: Updated node features [num_nodes, output_dim]
            pos: Unchanged positions (for EGNN compatibility)
        """
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            if self.use_layer_norm and i < len(self.layer_norms):
                x = self.layer_norms[i](x)
            
            # Apply activation and dropout (except last layer)
            if i < len(self.convs) - 1:
                x = F.silu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Return format compatible with EGNN blocks
        return x, pos if pos is not None else None


class VanillaGAT(nn.Module):
    """
    Multi-layer GAT for attention-based node feature learning.
    Compatible with AtomMP, EdgeMP, and ResMP blocks.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 2,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 use_layer_norm: bool = True):
        super().__init__()
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm
        
        # Build layers
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(GATConv(input_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout))
        if use_layer_norm:
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout))
            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        # Output layer (single head for final output)
        if num_layers > 1:
            self.convs.append(GATConv(hidden_dim, output_dim, heads=1, dropout=dropout))
        else:
            # Single layer case
            self.convs[0] = GATConv(input_dim, output_dim, heads=1, dropout=dropout)
            
        if use_layer_norm:
            self.layer_norms.append(nn.LayerNorm(output_dim))

    def forward(self, x, edge_index, edge_attr=None, pos=None):
        """
        Forward pass compatible with hierarchical encoder expectations.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features (ignored for GAT)
            pos: Node positions (returned unchanged for compatibility)
            
        Returns:
            h: Updated node features [num_nodes, output_dim]
            pos: Unchanged positions (for EGNN compatibility)
        """
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            if self.use_layer_norm and i < len(self.layer_norms):
                x = self.layer_norms[i](x)
            
            # Apply activation and dropout (except last layer)
            if i < len(self.convs) - 1:
                x = F.silu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Return format compatible with EGNN blocks
        return x, pos if pos is not None else None


class VanillaGIN(nn.Module):
    """
    Multi-layer GIN for isomorphism-aware node feature learning.
    Compatible with AtomMP, EdgeMP, and ResMP blocks.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 use_layer_norm: bool = True,
                 eps: float = 0.0):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm
        
        # Build layers
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        # Input layer MLP
        input_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINConv(input_mlp, eps=eps))
        if use_layer_norm:
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            hidden_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(hidden_mlp, eps=eps))
            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        # Output layer
        if num_layers > 1:
            output_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, output_dim)
            )
            self.convs.append(GINConv(output_mlp, eps=eps))
        else:
            # Single layer case
            single_mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, output_dim)
            )
            self.convs[0] = GINConv(single_mlp, eps=eps)
            
        if use_layer_norm:
            self.layer_norms.append(nn.LayerNorm(output_dim))

    def forward(self, x, edge_index, edge_attr=None, pos=None):
        """
        Forward pass compatible with hierarchical encoder expectations.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features (ignored for GIN)
            pos: Node positions (returned unchanged for compatibility)
            
        Returns:
            h: Updated node features [num_nodes, output_dim]
            pos: Unchanged positions (for EGNN compatibility)
        """
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            if self.use_layer_norm and i < len(self.layer_norms):
                x = self.layer_norms[i](x)
            
            # Apply activation and dropout (except last layer)
            if i < len(self.convs) - 1:
                x = F.silu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Return format compatible with EGNN blocks
        return x, pos if pos is not None else None


class RelationalGCN(nn.Module):
    """
    Multi-relational GCN for handling different edge types.
    Specifically designed for ResMP with multiple relation types.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_relations: int,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 use_layer_norm: bool = True):
        super().__init__()
        
        self.num_layers = num_layers
        self.num_relations = num_relations
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm
        
        # Build layers
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        # Input layer - separate weights per relation
        self.convs.append(nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) for _ in range(num_relations)
        ]))
        if use_layer_norm:
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim) for _ in range(num_relations)
            ]))
            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(nn.ModuleList([
                nn.Linear(hidden_dim, output_dim) for _ in range(num_relations)
            ]))
        else:
            # Single layer case
            self.convs[0] = nn.ModuleList([
                nn.Linear(input_dim, output_dim) for _ in range(num_relations)
            ])
            
        if use_layer_norm:
            self.layer_norms.append(nn.LayerNorm(output_dim))

    def forward(self, x, edge_indices_dict, edge_attrs_dict=None, pos=None):
        """
        Forward pass for multi-relational graphs.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_indices_dict: Dict of {relation_type: edge_index}
            edge_attrs_dict: Dict of {relation_type: edge_attr} (ignored)
            pos: Node positions (returned unchanged for compatibility)
            
        Returns:
            h: Updated node features [num_nodes, output_dim]
            pos: Unchanged positions (for EGNN compatibility)
        """
        
        for layer_idx, conv_list in enumerate(self.convs):
            # Aggregate messages from each relation type
            messages = []
            
            for rel_idx, (rel_type, edge_index) in enumerate(edge_indices_dict.items()):
                if edge_index.size(1) > 0:  # Check if edges exist
                    # Apply relation-specific transformation
                    h_rel = conv_list[rel_idx](x)
                    
                    # Message passing
                    row, col = edge_index
                    deg = degree(col, x.size(0), dtype=x.dtype)
                    deg_inv_sqrt = deg.pow(-0.5)
                    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
                    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
                    
                    # Aggregate messages
                    messages.append(torch.zeros_like(h_rel).scatter_add(0, col.unsqueeze(1).expand(-1, h_rel.size(1)), 
                                                                      norm.unsqueeze(1) * h_rel[row]))
            
            # Combine messages from all relations
            if messages:
                x = sum(messages) / len(messages)
            else:
                # No edges, just apply transformation
                x = conv_list[0](x) if len(conv_list) > 0 else x
            
            # Apply normalization, activation, dropout
            if self.use_layer_norm and layer_idx < len(self.layer_norms):
                x = self.layer_norms[layer_idx](x)
            
            if layer_idx < len(self.convs) - 1:
                x = F.silu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x, pos if pos is not None else None
    



# Factory function for creating vanilla GNNs
def create_vanilla_gnn(gnn_type: str, **kwargs):
    """
    Factory function to create vanilla GNN instances.
    
    Args:
        gnn_type: Type of GNN ('gcn', 'gat', 'gin', 'rgcn')
        **kwargs: Arguments passed to GNN constructor
        
    Returns:
        GNN instance
    """
    gnn_type = gnn_type.lower()
    
    if gnn_type == 'gcn':
        return VanillaGCN(**kwargs)
    elif gnn_type == 'gat':
        return VanillaGAT(**kwargs)
    elif gnn_type == 'gin':
        return VanillaGIN(**kwargs)
    elif gnn_type == 'rgcn':
        return RelationalGCN(**kwargs)
    else:
        raise ValueError(f"Unknown GNN type: {gnn_type}. Supported: gcn, gat, gin, rgcn")


# Wrapper classes for compatibility with existing EGNN interface
class VanillaAtomMP(nn.Module):
    """AtomMP wrapper for vanilla GNNs"""
    
    def __init__(self, gnn_type='gcn', in_node_nf=28, hidden_nf=64, out_node_nf=32, 
                 n_layers=3, dropout=0.1, **kwargs):
        super().__init__()
        self.gnn = create_vanilla_gnn(
            gnn_type,
            input_dim=in_node_nf,
            hidden_dim=hidden_nf,
            output_dim=out_node_nf,
            num_layers=n_layers,
            dropout=dropout,
            **kwargs
        )
    
    def forward(self, hetero_data, node_type):
        """Forward pass compatible with HierarchicalEncoder"""
        node_data = hetero_data[node_type]
        x = node_data.x
        pos = node_data.pos
        edge_index = hetero_data[(node_type, 'atom_bond', node_type)].edge_index
        edge_attr = hetero_data[(node_type, 'atom_bond', node_type)].edge_attr
        
        h, _ = self.gnn(x, edge_index, edge_attr, pos)
        return h


class VanillaEdgeMP(nn.Module):
    """EdgeMP wrapper for vanilla GNNs"""
    
    def __init__(self, gnn_type='gcn', input_dim=100, hidden_dims=[64, 64], 
                 num_layers=3, dropout=0.1, **kwargs):
        super().__init__()
        output_dim = hidden_dims[-1] if hidden_dims else input_dim
        self.gnn = create_vanilla_gnn(
            gnn_type,
            input_dim=input_dim,
            hidden_dim=hidden_dims[0] if hidden_dims else input_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            **kwargs
        )
    
    def forward(self, x, edge_index, edge_attr):
        """Forward pass compatible with EdgeMP interface"""
        h, _ = self.gnn(x, edge_index, edge_attr)
        return h


class VanillaResMP(nn.Module):
    """ResMP wrapper for vanilla GNNs with multi-relational support"""
    
    def __init__(self, gnn_type='gcn', node_dim=105, hidden_dim=128, num_layers=4, 
                 num_relations=4, dropout=0.1, **kwargs):
        super().__init__()
        
        if gnn_type == 'rgcn':
            self.gnn = create_vanilla_gnn(
                gnn_type,
                input_dim=node_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                num_relations=num_relations,
                num_layers=num_layers,
                dropout=dropout,
                **kwargs
            )
            self.is_relational = True
        else:
            # For non-relational GNNs, we'll flatten all edges
            self.gnn = create_vanilla_gnn(
                gnn_type,
                input_dim=node_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                **kwargs
            )
            self.is_relational = False
    
    def forward(self, hetero_data, node_type):
        """Forward pass compatible with HierarchicalEncoder"""
        node_data = hetero_data[node_type]
        x = node_data.x
        pos = node_data.pos
        
        if self.is_relational:
            # Collect edges by relation type
            edge_indices_dict = {}
            relation_names = ['r0', 'r1', 'r2', 'r3']
            
            for i, rel_name in enumerate(relation_names):
                edge_key = (node_type, rel_name, node_type)
                if edge_key in hetero_data.edge_types:
                    edge_indices_dict[i] = hetero_data[edge_key].edge_index
                else:
                    # Create empty edge index for missing relations
                    edge_indices_dict[i] = torch.empty((2, 0), dtype=torch.long, device=x.device)
            
            h, _ = self.gnn(x, edge_indices_dict, pos=pos)
        else:
            # Flatten all edges for non-relational GNNs
            all_edges = []
            relation_names = ['r0', 'r1', 'r2', 'r3']
            
            for rel_name in relation_names:
                edge_key = (node_type, rel_name, node_type)
                if edge_key in hetero_data.edge_types:
                    all_edges.append(hetero_data[edge_key].edge_index)
            
            if all_edges:
                edge_index = torch.cat(all_edges, dim=1)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)
            
            h, _ = self.gnn(x, edge_index, pos=pos)
        
        return h, pos


def test_vanilla_gnns():
    """Test vanilla GNN implementations"""
    print("Testing Vanilla GNNs...")
    
    # Test data
    num_nodes = 50
    input_dim = 28
    hidden_dim = 64
    output_dim = 32
    num_edges = 100
    
    x = torch.randn(num_nodes, input_dim)
    pos = torch.randn(num_nodes, 3)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 17)
    
    # Test each GNN type
    gnn_types = ['gcn', 'gat', 'gin']
    
    for gnn_type in gnn_types:
        print(f"\nTesting {gnn_type.upper()}...")
        
        gnn = create_vanilla_gnn(
            gnn_type,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=2
        )
        
        h, pos_out = gnn(x, edge_index, edge_attr, pos)
        
        print(f"  Input: {x.shape}")
        print(f"  Output: {h.shape}")
        print(f"  Positions unchanged: {torch.equal(pos, pos_out) if pos_out is not None else 'N/A'}")
        
        # Test parameter count
        params = sum(p.numel() for p in gnn.parameters())
        print(f"  Parameters: {params:,}")
    
    # Test relational GCN
    print(f"\nTesting Relational GCN...")
    
    num_relations = 4
    edge_indices_dict = {
        0: torch.randint(0, num_nodes, (2, 20)),
        1: torch.randint(0, num_nodes, (2, 15)),
        2: torch.randint(0, num_nodes, (2, 25)),
        3: torch.randint(0, num_nodes, (2, 10))
    }
    
    rgcn = RelationalGCN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_relations=num_relations,
        num_layers=2
    )
    
    h, pos_out = rgcn(x, edge_indices_dict, pos=pos)
    
    print(f"  Input: {x.shape}")
    print(f"  Output: {h.shape}")
    print(f"  Relations: {num_relations}")
    
    params = sum(p.numel() for p in rgcn.parameters())
    print(f"  Parameters: {params:,}")
    
    print("\n✅ All vanilla GNN tests completed successfully!")


if __name__ == "__main__":
    test_vanilla_gnns()