"""
Utilities for EpiFormer training and inference.

Basic utilities:
    Random seed setting
    Device selection
    Data loading
    Model saving/loading
Logging utilities:
    WandB initialization
    Directory creation
    Logging setup
Training helpers:
    Class weight calculation
    Metrics formatting
"""
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

try:
    import umap
except ImportError:
    umap = None

try:
    from sklearn.manifold import TSNE
except ImportError:
    TSNE = None




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



# Add this function to your utils.py
def select_device(gpu_id):
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        return device
    return torch.device('cpu')

# Update get_device function (optional but recommended)
def get_device(cfg):
    if hasattr(cfg, 'gpu_id') and torch.cuda.is_available():
        return select_device(cfg.gpu_id)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    """Initialize wandb logging. Returns False if wandb is not available."""
    if not WANDB_AVAILABLE:
        print("Warning: wandb not available, skipping initialization")
        return False

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb_cfg = cfg_dict.get('wandb', {})
    
    # Create descriptive notes based on model configuration
    model_cfg = cfg_dict.get('model', {})
    graph_type = model_cfg.get('graph_type', 'unknown')
    
    # Get encoder configuration
    ag_encoder = model_cfg.get('ag_encoder', {})
    ab_encoder = model_cfg.get('ab_encoder', {})
    
    # Build architecture description
    arch_parts = []
    if ag_encoder.get('atommp_enabled', False):
        atom_type = ag_encoder.get('atom_mp_type', 'egnn')
        arch_parts.append(f"AtomMP({atom_type})")
    if ag_encoder.get('edgemp_enabled', False):
        edge_type = ag_encoder.get('edgemp_type', 'egnn')
        arch_parts.append(f"EdgeMP({edge_type})")
    if ag_encoder.get('resmp_enabled', False):
        res_type = ag_encoder.get('resmp_type', 'egnn')
        arch_parts.append(f"ResMP({res_type})")
    
    architecture = " + ".join(arch_parts) if arch_parts else "No-Encoder"
    
    # Get loss and training info
    loss_name = cfg_dict.get('loss', {}).get('node_prediction', {}).get('name', 'unknown')
    mode = cfg_dict.get('mode', {}).get('mode', 'unknown')
    run_id = cfg_dict.get('run_id', 'unknown')
    
    # Create comprehensive notes
    notes = f"Graph: {graph_type.upper()} | Arch: {architecture} | Loss: {loss_name.upper()} | Mode: {mode} | Run: {run_id}"
    
    # Pull existing tags and add model-specific ones
    raw_tags = wandb_cfg.get('tags', [])
    if raw_tags and not isinstance(raw_tags, list):
        raw_tags = [raw_tags]
    
    # Add model-specific tags
    model_tags = [
        graph_type,
        f"arch-{architecture.replace(' + ', '-').replace('(', '').replace(')', '').lower()}",
        f"loss-{loss_name}",
        mode
    ]
    
    all_tags = (raw_tags or []) + model_tags
    tags = tuple(str(t) for t in all_tags if isinstance(t, (str, int, float)))
    
    wandb.init(
        project=wandb_cfg['project'],
        entity=wandb_cfg['entity'],
        name=wandb_cfg.get('name', None),
        group=wandb_cfg.get('group', None),
        tags=list(tags) if tags else None,
        notes=notes,
        config=cfg_dict,
        dir=wandb_cfg.get('save_dir', './wandb'),
        mode=wandb_cfg.get('mode', 'online'),
        resume=wandb_cfg.get('resume', 'allow'),
        anonymous=wandb_cfg.get('anonymous', 'allow'),
    )
    
    # Define metrics for combined train/validation plots
    wandb.define_metric("epoch")
    
    # All metrics that will be logged with train_ and val_ prefixes
    base_metrics = [
        "loss", "epitope_f1", "epitope_precision", "epitope_recall", "epitope_accuracy",
        "paratope_f1", "paratope_precision", "paratope_recall", "paratope_accuracy", 
        "edge_f1", "edge_precision", "edge_recall", "edge_accuracy",
        "epitope_loss", "paratope_loss", "edge_loss", "walle_loss"
    ]
    
    # Define step metric for all train/val variants
    for base_metric in base_metrics:
        wandb.define_metric(f"train_{base_metric}", step_metric="epoch")
        wandb.define_metric(f"val_{base_metric}", step_metric="epoch")
    
    # Also define other metrics
    wandb.define_metric("epoch_time", step_metric="epoch")
    wandb.define_metric("learning_rate", step_metric="epoch")
    wandb.define_metric("epi_threshold", step_metric="epoch")
    wandb.define_metric("para_threshold", step_metric="epoch")




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

# Pre-training utilities
def save_pretrained_weights(model, checkpoint_dir: str, epoch: int, loss: float, 
                          best_loss: float = None, is_best: bool = False):
    """Save pre-trained model weights (hierarchical model only, no projection heads)"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract hierarchical model weights
    hierarchical_state = model.hierarchical_model.state_dict()
    
    save_dict = {
        'epoch': epoch,
        'model_state_dict': hierarchical_state,
        'loss': loss,
        'best_loss': best_loss or loss
    }
    
    # Save regular checkpoint
    checkpoint_path = checkpoint_dir / f'pretrain_epoch_{epoch}.pt'
    torch.save(save_dict, checkpoint_path)
    
    # Save best checkpoint
    if is_best:
        best_path = checkpoint_dir / 'best_pretrained.pt'
        torch.save(save_dict, best_path)

def load_pretrained_weights(model, pretrain_path: str, strict: bool = True) -> bool:
    """Load pre-trained weights into hierarchical model for fine-tuning"""
    if not os.path.exists(pretrain_path):
        return False
    
    try:
        checkpoint = torch.load(pretrain_path, map_location='cpu')
        pretrained_state = checkpoint['model_state_dict']
        
        # Handle both HierarchicalModelWithHeads and HierarchicalModel
        target_model = model.hierarchical_model if hasattr(model, 'hierarchical_model') else model
        
        target_model.load_state_dict(pretrained_state, strict=strict)
        return True
        
    except Exception:
        return False

def get_best_pretrained_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Get path to best pre-trained checkpoint"""
    checkpoint_dir = Path(checkpoint_dir)
    
    # Look for best checkpoint first
    best_path = checkpoint_dir / 'best_pretrained.pt'
    if best_path.exists():
        return str(best_path)
    
    # Fallback to latest checkpoint
    checkpoints = list(checkpoint_dir.glob('pretrain_epoch_*.pt'))
    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
        return str(checkpoints[-1])
    
    return None


# import os
# import torch, random
# import numpy as np
# from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryMatthewsCorrCoef, BinaryAveragePrecision, BinaryAUROC, BinaryF1Score,  BinaryAccuracy

# # from model.loss import NTXentLoss  # Importing NTXentLoss from a custom module


# # Set up device configuration (use GPU if available, otherwise CPU)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# def seed_everything(seed):
#     # Set Python random seed
#     random.seed(seed)

#     # Set NumPy random seed
#     np.random.seed(seed)

#     # Set PyTorch random seed
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


# def generate_random_seed() -> int:
#     """ Generate a random seed using os.urandom
#     credit: https://stackoverflow.com/q/57416925
#     """
#     return int.from_bytes(
#         os.urandom(4),
#         byteorder="big"
#     )



def get_k_fold_data(K, i, X):
    """
    Split data into K folds for cross-validation and return train/val/test sets for fold i.
    
    Args:
        K (int): Number of folds
        i (int): Current fold index (0-based)
        X (array-like): Data to be split
        
    Returns:
        tuple: (X_train, X_val, X_test) - Training, validation and test sets for fold i
    """
    assert K > 1  # Ensure we have at least 2 folds
    fold_size = len(X) // K  # Calculate size of each fold

    X_train, X_val, X_test = None, None, None

    # Prepare fold indices
    tmp_list = list(range(K))
    idx_i = tmp_list.index(i)
    del tmp_list[idx_i]
    v = tmp_list[-1]  # Last remaining index will be validation set

    # Split data into folds
    for j in range(K):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part = X[idx]
        
        # Assign to test, val or train sets
        if j == i:
            X_test = X_part
        elif j == v:
            X_val = X_part
        elif X_train is None:
            X_train = X_part
        else:
            X_train = np.concatenate((X_train, X_part), axis=0)
    return X_train, X_val, X_test




# ----------------------------------------------------

# def evalution_prot(preds, targets):
#     """
#     Evaluate model predictions against targets using multiple metrics.
    
#     Args:
#         preds (torch.Tensor): Model predictions
#         targets (torch.Tensor): Ground truth labels
        
#     Returns:
#         tuple: (AUPRC, AUROC, precision, recall, f1, bacc, MCC)
#     """
#     # AUROC (Area Under Receiver Operating Characteristic curve)
#     auroc = BinaryAUROC().to(device)
#     auroc.update(preds, targets)
#     auroc_i = auroc.compute().item()
    
#     # AUPRC (Area Under Precision-Recall Curve)
#     auprc = BinaryAveragePrecision().to(device)
#     auprc.update(preds, targets)
#     auprc_i = auprc.compute().item()
    
#     # Precision
#     precision = BinaryPrecision().to(device)
#     precision_i = precision(preds, targets).item()
    
#     # Recall
#     recall = BinaryRecall().to(device)
#     recall_i = recall(preds, targets).item()
    
#     # F1 Score
#     f1 = BinaryF1Score().to(device)
#     f1_i = f1(preds, targets).item()
    
#     # Balanced Accuracy (using regular accuracy for binary classification)
#     bacc = BinaryAccuracy().to(device)
#     bacc_i = bacc(preds, targets).item()
    
#     # MCC (Matthews Correlation Coefficient)
#     mcc = BinaryMatthewsCorrCoef().to(device)
#     mcc_i = mcc(preds, targets).item()

#     return auprc_i, auroc_i, precision_i, recall_i, f1_i, bacc_i, mcc_i


# def consine_inter(A, B):
#     """
#     Compute cosine similarity between corresponding vectors in A and B.
    
#     Args:
#         A (torch.Tensor): First set of vectors
#         B (torch.Tensor): Second set of vectors
        
#     Returns:
#         torch.Tensor: Cosine similarity scores
#     """
#     dot_product = torch.sum(A * B, dim=1)
#     norm_A = torch.norm(A, dim=1)
#     norm_B = torch.norm(B, dim=1)
#     # Add small epsilon to avoid division by zero
#     cosine_similarity = dot_product / ((norm_A * norm_B) + 1e-8)
#     return cosine_similarity

# def dis_pairs(coord_1, coord_2):
#     """
#     Calculate Euclidean distance between two 3D coordinates.
    
#     Args:
#         coord_1 (list): First coordinate [x,y,z]
#         coord_2 (list): Second coordinate [x,y,z]
        
#     Returns:
#         float: Euclidean distance between the coordinates
#     """
#     # Extract coordinates
#     coord_1_x = coord_1[-3]
#     coord_1_y = coord_1[-2]
#     coord_1_z = coord_1[-1]
#     coord_2_x = coord_2[-3]
#     coord_2_y = coord_2[-2]
#     coord_2_z = coord_2[-1]
    
#     # Calculate Euclidean distance
#     distance = np.sqrt((float(coord_1_x) - float(coord_2_x)) ** 2 + 
#                (float(coord_1_y) - float(coord_2_y)) ** 2 + 
#                (float(coord_1_z) - float(coord_2_z)) ** 2)
#     return distance

# def index_mink(data, k):
#     """
#     Find indices of the k smallest values in a list.
    
#     Args:
#         data (list): Input data
#         k (int): Number of smallest values to find
        
#     Returns:
#         list: Indices of the k smallest values
#     """
#     Lst = data[:]  # Create a copy of the input list
#     index_k = []
#     for i in range(k):
#         index_i = Lst.index(min(Lst))  # Find index of current minimum
#         index_k.append(index_i)
#         Lst[index_i] = float('inf')  # Replace found minimum with infinity
#     return index_k

# def CreateGearnetGraph(data):
#     """
#     - Create Gearnet graph structures for AG and AB from input data.
#     - Create and combine 3 types of edges: sequential, radius, and kNN.
    
#     Args:
#         data (dict): Input data containing edge and coordinate information
        
#     Returns:
#         tuple: (ag_edge_ind, ab_edge_ind) - Graph structures for AG and AB
#     """
#     from torchdrug import data as drugdata
    
#     # Process AG (Antigen) graph
#     edge_AG_radius = (np.array(data["edge_AG"] + ([[1] * len(data["edge_AG"][0])])).T).tolist()
#     num_nodes_AG = max(max(np.array(edge_AG_radius)[:, 0]), max(np.array(edge_AG_radius)[:, 1])) + 1
    
#     # Create sequential edges
#     edge_AG_seq = []
#     for p in range(num_nodes_AG - 1):
#         edge_AG_seq.append([p, p + 1, 0])
    
#     # Create 10-nearest neighbor edges
#     edge_AG_10nearest = []
#     for p in range(num_nodes_AG):
#         dis_pq = []
#         for q in range(num_nodes_AG):
#             dis_pq.append(dis_pairs(data["coord_AG"][p], data["coord_AG"][q]))
#         near10_q = index_mink(dis_pq, 11)  # Get 11 nearest (including self)
#         del near10_q[near10_q.index(p)]  # Remove self
#         near10_AG_p = list(map(lambda x: [p, x, 2], near10_q))
#         edge_AG_10nearest = edge_AG_10nearest + near10_AG_p
    
#     # Combine all edge types
#     edge_AG = edge_AG_seq + edge_AG_radius + edge_AG_10nearest
#     graph_AG = drugdata.Graph(edge_AG, num_node=num_nodes_AG, num_relation=3).to(device)
#     node_embedding_AG = torch.tensor(data["vertex_AG"], dtype=torch.float).to(device)
#     ag_edge_ind = [graph_AG, node_embedding_AG]
    
#     # Process AB (Antibody) graph (similar to AG)
#     edge_AB_radius = (np.array(data["edge_AB"] + ([[1] * len(data["edge_AB"][0])])).T).tolist()
#     num_nodes_AB = max(max(np.array(edge_AB_radius)[:, 0]), max(np.array(edge_AB_radius)[:, 1])) + 1
    
#     edge_AB_seq = []
#     for p in range(num_nodes_AG - 1):  # Note: This uses num_nodes_AG which might be a bug
#         edge_AG_seq.append([p, p + 1, 0])
    
#     edge_AB_10nearest = []
#     for p in range(num_nodes_AB):
#         dis_pq = []
#         for q in range(num_nodes_AB):
#             dis_pq.append(dis_pairs(data["coord_AB"][p], data["coord_AB"][q]))
#         near10_q = index_mink(dis_pq, 11)
#         del near10_q[near10_q.index(p)]
#         near10_AB_p = list(map(lambda x: [p, x, 2], near10_q))
#         edge_AB_10nearest = edge_AB_10nearest + near10_AB_p
    
#     edge_AB = torch.tensor(edge_AB_seq + edge_AB_radius + edge_AB_10nearest)
#     graph_AB = drugdata.Graph(edge_AB, num_node=num_nodes_AB, num_relation=3).to(device)
#     node_embedding_AB = torch.tensor(data["vertex_AB"], dtype=torch.float).to(device)
#     ab_edge_ind = [graph_AB, node_embedding_AB]

#     return ag_edge_ind, ab_edge_ind

# def CreateKnearestEdge(data):
#     """
#     Create k-nearest neighbor edges for AG and AB graphs.
    
#     Args:
#         data (dict): Input data containing coordinate information
        
#     Returns:
#         tuple: (edge_AG_10nearest, edge_AB_10nearest) - Edge lists for AG and AB
#     """
#     # Process AG (Antigen) edges
#     num_nodes_AG = data["vertex_AB"].shape[0]  # Note: This uses vertex_AB which might be a bug
#     edge_AG_10nearest_p = []
#     edge_AG_10nearest_q = []
    
#     for p in range(num_nodes_AG):
#         dis_pq = []
#         for q in range(num_nodes_AG):
#             dis_pq.append(dis_pairs(data["coord_AG"][p], data["coord_AG"][q]))
#         near10_q = index_mink(dis_pq, 11)
#         del near10_q[near10_q.index(p)]
#         near10_p = [p] * 10
#         edge_AG_10nearest_p = edge_AG_10nearest_p + near10_p
#         edge_AG_10nearest_q = edge_AG_10nearest_q + near10_q
#     edge_AG_10nearest = [edge_AG_10nearest_p, edge_AG_10nearest_q]
    
#     # Process AB (Antibody) edges (similar to AG)
#     num_nodes_AB = data["vertex_AB"].shape[0]
#     edge_AB_10nearest_p = []
#     edge_AB_10nearest_q = []
#     for p in range(num_nodes_AB):
#         dis_pq = []
#         for q in range(num_nodes_AB):
#             dis_pq.append(dis_pairs(data["coord_AB"][p], data["coord_AB"][q]))
#         near10_q = index_mink(dis_pq, 11)
#         del near10_q[near10_q.index(p)]
#         near10_p = [p] * 10
#         edge_AB_10nearest_p = edge_AB_10nearest_p + near10_p
#         edge_AB_10nearest_q = edge_AB_10nearest_q + near10_q
#     edge_AB_10nearest = [edge_AB_10nearest_p, edge_AB_10nearest_q]
#     return edge_AG_10nearest, edge_AB_10nearest









# # Set up device configuration (use GPU if available, otherwise CPU)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def get_k_fold_data(K, i, X):
#     """
#     Split data into K folds for cross-validation and return train/val/test sets for fold i.
    
#     Args:
#         K (int): Number of folds
#         i (int): Current fold index (0-based)
#         X (array-like): Data to be split
        
#     Returns:
#         tuple: (X_train, X_val, X_test) - Training, validation and test sets for fold i
#     """
#     assert K > 1  # Ensure we have at least 2 folds
#     fold_size = len(X) // K  # Calculate size of each fold

#     X_train, X_val, X_test = None, None, None

#     # Prepare fold indices
#     tmp_list = list(range(K))
#     idx_i = tmp_list.index(i)
#     del tmp_list[idx_i]
#     v = tmp_list[-1]  # Last remaining index will be validation set

#     # Split data into folds
#     for j in range(K):
#         idx = slice(j * fold_size, (j + 1) * fold_size)
#         X_part = X[idx]
        
#         # Assign to test, val or train sets
#         if j == i:
#             X_test = X_part
#         elif j == v:
#             X_val = X_part
#         elif X_train is None:
#             X_train = X_part
#         else:
#             X_train = np.concatenate((X_train, X_part), axis=0)
#     return X_train, X_val, X_test


# def consine_inter(A, B):
#     """
#     Compute cosine similarity between corresponding vectors in A and B.
    
#     Args:
#         A (torch.Tensor): First set of vectors
#         B (torch.Tensor): Second set of vectors
        
#     Returns:
#         torch.Tensor: Cosine similarity scores
#     """
#     dot_product = torch.sum(A * B, dim=1)
#     norm_A = torch.norm(A, dim=1)
#     norm_B = torch.norm(B, dim=1)
#     # Add small epsilon to avoid division by zero
#     cosine_similarity = dot_product / ((norm_A * norm_B) + 1e-8)
#     return cosine_similarity

# def dis_pairs(coord_1, coord_2):
#     """
#     Calculate Euclidean distance between two 3D coordinates.
    
#     Args:
#         coord_1 (list): First coordinate [x,y,z]
#         coord_2 (list): Second coordinate [x,y,z]
        
#     Returns:
#         float: Euclidean distance between the coordinates
#     """
#     # Extract coordinates
#     coord_1_x = coord_1[-3]
#     coord_1_y = coord_1[-2]
#     coord_1_z = coord_1[-1]
#     coord_2_x = coord_2[-3]
#     coord_2_y = coord_2[-2]
#     coord_2_z = coord_2[-1]
    
#     # Calculate Euclidean distance
#     distance = np.sqrt((float(coord_1_x) - float(coord_2_x)) ** 2 + 
#                (float(coord_1_y) - float(coord_2_y)) ** 2 + 
#                (float(coord_1_z) - float(coord_2_z)) ** 2)
#     return distance

# def index_mink(data, k):
#     """
#     Find indices of the k smallest values in a list.
    
#     Args:
#         data (list): Input data
#         k (int): Number of smallest values to find
        
#     Returns:
#         list: Indices of the k smallest values
#     """
#     Lst = data[:]  # Create a copy of the input list
#     index_k = []
#     for i in range(k):
#         index_i = Lst.index(min(Lst))  # Find index of current minimum
#         index_k.append(index_i)
#         Lst[index_i] = float('inf')  # Replace found minimum with infinity
#     return index_k

# def CreateGearnetGraph(data):
#     """
#     - Create Gearnet graph structures for AG and AB from input data.
#     - Create and combine 3 types of edges: sequential, radius, and kNN.
    
#     Args:
#         data (dict): Input data containing edge and coordinate information
        
#     Returns:
#         tuple: (ag_edge_ind, ab_edge_ind) - Graph structures for AG and AB
#     """
#     from torchdrug import data as drugdata
    
#     # Process AG (Antigen) graph
#     edge_AG_radius = (np.array(data["edge_AG"] + ([[1] * len(data["edge_AG"][0])])).T).tolist()
#     num_nodes_AG = max(max(np.array(edge_AG_radius)[:, 0]), max(np.array(edge_AG_radius)[:, 1])) + 1
    
#     # Create sequential edges
#     edge_AG_seq = []
#     for p in range(num_nodes_AG - 1):
#         edge_AG_seq.append([p, p + 1, 0])
    
#     # Create 10-nearest neighbor edges
#     edge_AG_10nearest = []
#     for p in range(num_nodes_AG):
#         dis_pq = []
#         for q in range(num_nodes_AG):
#             dis_pq.append(dis_pairs(data["coord_AG"][p], data["coord_AG"][q]))
#         near10_q = index_mink(dis_pq, 11)  # Get 11 nearest (including self)
#         del near10_q[near10_q.index(p)]  # Remove self
#         near10_AG_p = list(map(lambda x: [p, x, 2], near10_q))
#         edge_AG_10nearest = edge_AG_10nearest + near10_AG_p
    
#     # Combine all edge types
#     edge_AG = edge_AG_seq + edge_AG_radius + edge_AG_10nearest
#     graph_AG = drugdata.Graph(edge_AG, num_node=num_nodes_AG, num_relation=3).to(device)
#     node_embedding_AG = torch.tensor(data["vertex_AG"], dtype=torch.float).to(device)
#     ag_edge_ind = [graph_AG, node_embedding_AG]
    
#     # Process AB (Antibody) graph (similar to AG)
#     edge_AB_radius = (np.array(data["edge_AB"] + ([[1] * len(data["edge_AB"][0])])).T).tolist()
#     num_nodes_AB = max(max(np.array(edge_AB_radius)[:, 0]), max(np.array(edge_AB_radius)[:, 1])) + 1
    
#     edge_AB_seq = []
#     for p in range(num_nodes_AG - 1):  # Note: This uses num_nodes_AG which might be a bug
#         edge_AG_seq.append([p, p + 1, 0])
    
#     edge_AB_10nearest = []
#     for p in range(num_nodes_AB):
#         dis_pq = []
#         for q in range(num_nodes_AB):
#             dis_pq.append(dis_pairs(data["coord_AB"][p], data["coord_AB"][q]))
#         near10_q = index_mink(dis_pq, 11)
#         del near10_q[near10_q.index(p)]
#         near10_AB_p = list(map(lambda x: [p, x, 2], near10_q))
#         edge_AB_10nearest = edge_AB_10nearest + near10_AB_p
    
#     edge_AB = torch.tensor(edge_AB_seq + edge_AB_radius + edge_AB_10nearest)
#     graph_AB = drugdata.Graph(edge_AB, num_node=num_nodes_AB, num_relation=3).to(device)
#     node_embedding_AB = torch.tensor(data["vertex_AB"], dtype=torch.float).to(device)
#     ab_edge_ind = [graph_AB, node_embedding_AB]

#     return ag_edge_ind, ab_edge_ind

# def CreateKnearestEdge(data):
#     """
#     Create k-nearest neighbor edges for AG and AB graphs.
    
#     Args:
#         data (dict): Input data containing coordinate information
        
#     Returns:
#         tuple: (edge_AG_10nearest, edge_AB_10nearest) - Edge lists for AG and AB
#     """
#     # Process AG (Antigen) edges
#     num_nodes_AG = data["vertex_AB"].shape[0]  # Note: This uses vertex_AB which might be a bug
#     edge_AG_10nearest_p = []
#     edge_AG_10nearest_q = []
    
#     for p in range(num_nodes_AG):
#         dis_pq = []
#         for q in range(num_nodes_AG):
#             dis_pq.append(dis_pairs(data["coord_AG"][p], data["coord_AG"][q]))
#         near10_q = index_mink(dis_pq, 11)
#         del near10_q[near10_q.index(p)]
#         near10_p = [p] * 10
#         edge_AG_10nearest_p = edge_AG_10nearest_p + near10_p
#         edge_AG_10nearest_q = edge_AG_10nearest_q + near10_q
#     edge_AG_10nearest = [edge_AG_10nearest_p, edge_AG_10nearest_q]
    
#     # Process AB (Antibody) edges (similar to AG)
#     num_nodes_AB = data["vertex_AB"].shape[0]
#     edge_AB_10nearest_p = []
#     edge_AB_10nearest_q = []
#     for p in range(num_nodes_AB):
#         dis_pq = []
#         for q in range(num_nodes_AB):
#             dis_pq.append(dis_pairs(data["coord_AB"][p], data["coord_AB"][q]))
#         near10_q = index_mink(dis_pq, 11)
#         del near10_q[near10_q.index(p)]
#         near10_p = [p] * 10
#         edge_AB_10nearest_p = edge_AB_10nearest_p + near10_p
#         edge_AB_10nearest_q = edge_AB_10nearest_q + near10_q
#     edge_AB_10nearest = [edge_AB_10nearest_p, edge_AB_10nearest_q]
#     return edge_AG_10nearest, edge_AB_10nearest


# ==================== Debug Utilities ====================

def save_embedding_plots(ag_features, ab_features, epi_labels, para_labels, 
                        save_dir, epoch_or_suffix="final"):
    """
    Create and save tSNE and UMAP visualizations of antigen/antibody embeddings.
    
    Args:
        ag_features: Antigen embeddings [N_ag, feature_dim]
        ab_features: Antibody embeddings [N_ab, feature_dim] 
        epi_labels: Epitope labels [N_ag]
        para_labels: Paratope labels [N_ab]
        save_dir: Directory to save plots
        epoch_or_suffix: Suffix for filename (e.g., epoch number or "final")
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to numpy if needed
    if hasattr(ag_features, 'detach'):
        ag_features = ag_features.detach().cpu().numpy()
    if hasattr(ab_features, 'detach'):
        ab_features = ab_features.detach().cpu().numpy()
    if hasattr(epi_labels, 'detach'):
        epi_labels = epi_labels.detach().cpu().numpy()
    if hasattr(para_labels, 'detach'):
        para_labels = para_labels.detach().cpu().numpy()
    
    # Create UMAP plots if available
    if umap is not None:
        try:
            # Antigen UMAP
            reducer = umap.UMAP(n_components=2, random_state=42)
            ag_umap = reducer.fit_transform(ag_features)
            
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            scatter = plt.scatter(ag_umap[:, 0], ag_umap[:, 1], c=epi_labels, 
                                cmap='coolwarm', alpha=0.7, s=20)
            plt.colorbar(scatter, label='Epitope')
            plt.title(f'Antigen Embeddings (UMAP) - {epoch_or_suffix}')
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            
            # Antibody UMAP
            ab_umap = reducer.fit_transform(ab_features)
            plt.subplot(1, 2, 2)
            scatter = plt.scatter(ab_umap[:, 0], ab_umap[:, 1], c=para_labels, 
                                cmap='coolwarm', alpha=0.7, s=20)
            plt.colorbar(scatter, label='Paratope')
            plt.title(f'Antibody Embeddings (UMAP) - {epoch_or_suffix}')
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/embeddings_umap_{epoch_or_suffix}.png", 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"UMAP visualization failed: {e}")
    
    # Create tSNE plots if available
    if TSNE is not None:
        try:
            # Antigen tSNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(ag_features)-1))
            ag_tsne = tsne.fit_transform(ag_features)
            
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            scatter = plt.scatter(ag_tsne[:, 0], ag_tsne[:, 1], c=epi_labels, 
                                cmap='coolwarm', alpha=0.7, s=20)
            plt.colorbar(scatter, label='Epitope')
            plt.title(f'Antigen Embeddings (tSNE) - {epoch_or_suffix}')
            plt.xlabel('tSNE 1')
            plt.ylabel('tSNE 2')
            
            # Antibody tSNE
            tsne_ab = TSNE(n_components=2, random_state=42, perplexity=min(30, len(ab_features)-1))
            ab_tsne = tsne_ab.fit_transform(ab_features)
            plt.subplot(1, 2, 2)
            scatter = plt.scatter(ab_tsne[:, 0], ab_tsne[:, 1], c=para_labels, 
                                cmap='coolwarm', alpha=0.7, s=20)
            plt.colorbar(scatter, label='Paratope')
            plt.title(f'Antibody Embeddings (tSNE) - {epoch_or_suffix}')
            plt.xlabel('tSNE 1')
            plt.ylabel('tSNE 2')
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/embeddings_tsne_{epoch_or_suffix}.png", 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"tSNE visualization failed: {e}")


def get_data_splits_by_mode(data, mode_cfg):
    """
    Split data according to training mode configuration.
    
    Args:
        data: Dataset to split
        mode_cfg: Mode configuration from hydra
        
    Returns:
        Tuple of (train_data, val_data, test_data)
        val_data may be None for certain modes
    """
    mode = mode_cfg.mode
    
    # Get split ratios
    train_ratio = getattr(mode_cfg, 'train_ratio', 0.7)
    val_ratio = getattr(mode_cfg, 'val_ratio', 0.15)
    test_ratio = getattr(mode_cfg, 'test_ratio', 0.15)
    seed = getattr(mode_cfg, 'random_seed', 42)
    
    # Use subset if specified
    if getattr(mode_cfg, 'use_test_subset', False) and hasattr(mode_cfg, 'test_subset_path'):
        # Note: This would need to be implemented based on your data loading structure
        pass
    
    # Perform split
    rng = np.random.default_rng(seed)
    n = len(data)
    indices = rng.permutation(n)
    
    if mode == "test":
        # Train/test split only
        split_point = int(n * train_ratio)
        train_indices = indices[:split_point]
        test_indices = indices[split_point:]
        
        train_data = [data[i] for i in train_indices]
        val_data = None
        test_data = [data[i] for i in test_indices]
        
    else:
        # Train/val/test split
        train_split = int(n * train_ratio)
        val_split = int(n * (train_ratio + val_ratio))
        
        train_indices = indices[:train_split]
        val_indices = indices[train_split:val_split]
        test_indices = indices[val_split:]
        
        train_data = [data[i] for i in train_indices]
        val_data = [data[i] for i in val_indices]
        test_data = [data[i] for i in test_indices]
    
    return train_data, val_data, test_data


def compute_simple_debug_stats(logits, loss_components=None, model=None):
    """
    Compute simple debug statistics for training monitoring.
    
    Args:
        logits: Model logits before sigmoid
        loss_components: Dict of loss components (optional)
        model: Model for gradient computation (optional)
        
    Returns:
        Dict of debug statistics
    """
    stats = {}
    
    # Logit statistics
    with torch.no_grad():
        stats['logit_mean'] = logits.mean().item()
        stats['logit_std'] = logits.std().item() 
        stats['logit_min'] = logits.min().item()
        stats['logit_max'] = logits.max().item()
        
        # Probability statistics after sigmoid
        probs = torch.sigmoid(logits)
        near_zero = (probs < 0.01).float().mean().item() * 100
        near_one = (probs > 0.99).float().mean().item() * 100
        stats['prob_near_zero_pct'] = near_zero
        stats['prob_near_one_pct'] = near_one
        stats['prob_collapse_warning'] = near_zero > 90 or near_one > 90
    
    # Loss component breakdown
    if loss_components is not None:
        stats.update(loss_components)
    
    # Gradient norms (if model provided)
    if model is not None:
        total_norm = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                
                # Track encoder-specific gradients
                if 'ag_encoder' in name:
                    stats['ag_encoder_grad_norm'] = stats.get('ag_encoder_grad_norm', 0) + param_norm.item() ** 2
                elif 'ab_encoder' in name:
                    stats['ab_encoder_grad_norm'] = stats.get('ab_encoder_grad_norm', 0) + param_norm.item() ** 2
        
        stats['total_grad_norm'] = total_norm ** 0.5
        stats['ag_encoder_grad_norm'] = stats.get('ag_encoder_grad_norm', 0) ** 0.5
        stats['ab_encoder_grad_norm'] = stats.get('ab_encoder_grad_norm', 0) ** 0.5
        
        # Ratio to detect if AB encoder is learning
        if stats['ag_encoder_grad_norm'] > 0:
            stats['ab_ag_grad_ratio'] = stats['ab_encoder_grad_norm'] / stats['ag_encoder_grad_norm']
        else:
            stats['ab_ag_grad_ratio'] = 0
    
    return stats