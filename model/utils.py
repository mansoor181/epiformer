import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.data import Data
import numpy as np
from math import pi


import os
import random
import numpy as np
import torch
import wandb
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime
from omegaconf import OmegaConf



# encoder helper functions

def rbf(values, offsets, widths):
    """
    Radial Basis Function expansion.
    Args:
        values: Input distances (N,)
        offsets: Centers of RBF (K,)
        widths: Widths of RBF (K,)
    Returns:
        Expanded features (N, K)
    """
    diff = values.view(-1, 1) - offsets.view(1, -1)
    return torch.exp(-0.5 * (diff / widths) ** 2)

def get_angles(pos_i, pos_j, pos_k):
    """
    Calculate angles between three points.
    Returns:
        angles in radians (N,)
    """
    vec_ji = pos_j - pos_i
    vec_jk = pos_j - pos_k
    cos_angle = torch.sum(vec_ji * vec_jk, dim=-1) / (
        torch.norm(vec_ji, dim=-1) * torch.norm(vec_jk, dim=-1) + 1e-10
    )
    return torch.acos(torch.clamp(cos_angle, -1.0, 1.0))

def quaternion_from_matrix(matrix):
    """
    Convert rotation matrix to quaternion.
    Args:
        matrix: (N, 3, 3)
    Returns:
        quaternion: (N, 4) in (w, x, y, z) format
    """
    trace = matrix[:, 0, 0] + matrix[:, 1, 1] + matrix[:, 2, 2]
    q = torch.zeros(matrix.size(0), 4, device=matrix.device)
    
    mask0 = trace > 0
    if mask0.any():
        s = 0.5 / torch.sqrt(trace[mask0] + 1.0)
        q[mask0, 0] = 0.25 / s
        q[mask0, 1] = (matrix[mask0, 2, 1] - matrix[mask0, 1, 2]) * s
        q[mask0, 2] = (matrix[mask0, 0, 2] - matrix[mask0, 2, 0]) * s
        q[mask0, 3] = (matrix[mask0, 1, 0] - matrix[mask0, 0, 1]) * s
        
    mask1 = (matrix[:, 0, 0] > matrix[:, 1, 1]) & (matrix[:, 0, 0] > matrix[:, 2, 2])
    if mask1.any():
        s = 2.0 * torch.sqrt(1.0 + matrix[mask1, 0, 0] - matrix[mask1, 1, 1] - matrix[mask1, 2, 2])
        q[mask1, 0] = (matrix[mask1, 2, 1] - matrix[mask1, 1, 2]) / s
        q[mask1, 1] = 0.25 * s
        q[mask1, 2] = (matrix[mask1, 0, 1] + matrix[mask1, 1, 0]) / s
        q[mask1, 3] = (matrix[mask1, 0, 2] + matrix[mask1, 2, 0]) / s
        
    mask2 = matrix[:, 1, 1] > matrix[:, 2, 2]
    if mask2.any():
        s = 2.0 * torch.sqrt(1.0 + matrix[mask2, 1, 1] - matrix[mask2, 0, 0] - matrix[mask2, 2, 2])
        q[mask2, 0] = (matrix[mask2, 0, 2] - matrix[mask2, 2, 0]) / s
        q[mask2, 1] = (matrix[mask2, 0, 1] + matrix[mask2, 1, 0]) / s
        q[mask2, 2] = 0.25 * s
        q[mask2, 3] = (matrix[mask2, 1, 2] + matrix[mask2, 2, 1]) / s
        
    mask3 = ~(mask0 | mask1 | mask2)
    if mask3.any():
        s = 2.0 * torch.sqrt(1.0 + matrix[mask3, 2, 2] - matrix[mask3, 0, 0] - matrix[mask3, 1, 1])
        q[mask3, 0] = (matrix[mask3, 1, 0] - matrix[mask3, 0, 1]) / s
        q[mask3, 1] = (matrix[mask3, 0, 2] + matrix[mask3, 2, 0]) / s
        q[mask3, 2] = (matrix[mask3, 1, 2] + matrix[mask3, 2, 1]) / s
        q[mask3, 3] = 0.25 * s
        
    return q





def train_test_split(data: Any, seed: int, test_ratio: float = 0.2) -> tuple:
    """
    Perform a single random train/test split.
    Returns (train_data, test_data).
    TODO:
    - use sklearn's train-test split function
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    perm = rng.permutation(n)
    cut = int(n * (1 - test_ratio))
    train_idx, test_idx = perm[:cut], perm[cut:]
    train_data = [data[i] for i in train_idx]
    test_data  = [data[i] for i in test_idx]
    return train_data, test_data


def seed_everything(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    """Get the device to use for computations."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def load_data(data_path: str) -> Dict:
    """
    Load preprocessed data from pickle file.
    
    Args:
        data_path: Path to the pickle file containing the data
        
    Returns:
        Dictionary containing:
        - complex_code: PDB complex identifiers
        - coord_AG: Antigen coordinates
        - label_AG: Antigen labels (epitope/non-epitope)
        - coord_AB: Antibody coordinates
        - label_AB: Antibody labels (paratope/non-paratope) 
        - edge_AGAB: Edges between AG-AB nodes
        - edge_AB: Edges within antibody graph
        - edge_AG: Edges within antigen graph
        - vertex_AB: Antibody node features
        - vertex_AG: Antigen node features
        - AbLang_AB: Antibody language model embeddings
        - ESM1b_AG: Antigen language model embeddings
    """
    data = torch.load(data_path)
    return data


def initialize_wandb(cfg):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb_cfg = cfg_dict.get('wandb', {})

    # Pull tags out, ensure it's a list of str
    raw_tags = wandb_cfg.get('tags', None)
    tags = None
    if raw_tags:
        # force everything to str, skip non-iterables
        tags = tuple(str(t) for t in raw_tags if isinstance(t, (str, int, float)))
    
    wandb.init(
        project=wandb_cfg['project'],
        entity=wandb_cfg['entity'],
        name=wandb_cfg.get('name', None),
        group=wandb_cfg.get('group', None),
        # only pass tags if non-empty tuple
        **({'tags': tags} if tags else {}),
        notes=wandb_cfg.get('notes', None),
        config=cfg_dict,
        dir=wandb_cfg.get('save_dir', './wandb'),
        mode=wandb_cfg.get('mode', 'online'),
        resume=wandb_cfg.get('resume', 'allow'),
        anonymous=wandb_cfg.get('anonymous', 'allow'),
    )




def save_model(model: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               epoch: int,
               loss: float,
               metrics: Dict[str, float],
               path: Union[str, Path]) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch number  
        loss: Current loss value
        metrics: Dictionary of metric values
        path: Path to save checkpoint
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics
    }, path)

def load_model(model: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               path: Union[str, Path]) -> tuple:
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model to load weights into
        optimizer: Optimizer to load state into 
        path: Path to checkpoint file
        
    Returns:
        Tuple containing:
        - epoch number
        - model with loaded weights
        - optimizer with loaded state
        - loss value
        - metrics dictionary
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return (
        checkpoint['epoch'],
        model,
        optimizer,
        checkpoint['loss'],
        checkpoint['metrics']
    )

def get_run_dir(base_dir: Union[str, Path], run_name: str) -> Path:
    """Create and return directory for current run."""
    run_dir = Path(base_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def setup_logging(run_dir: Path) -> None:
    """Setup logging to file."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(run_dir / 'train.log'),
            logging.StreamHandler()
        ]
    )

def calculate_class_weights(labels: torch.Tensor) -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        labels: Binary labels tensor
        
    Returns:
        Tensor of class weights
    """
    num_samples = len(labels)
    num_positives = labels.sum().item()
    num_negatives = num_samples - num_positives
    
    pos_weight = num_samples / (2 * num_positives)
    neg_weight = num_samples / (2 * num_negatives)
    
    weights = torch.zeros_like(labels, dtype=torch.float)
    weights[labels == 1] = pos_weight
    weights[labels == 0] = neg_weight
    
    return weights

def format_metrics(metrics: Dict[str, float], prefix: str = '') -> str:
    """Format metrics dictionary into string for printing."""
    return ' | '.join([
        f"{prefix}{k}: {v:.4f}" for k, v in metrics.items()
    ])

