
"""
TODO: 
- save the config with the model checkpoints to reproduce the same setup
"""

"""
Enhanced checkpointing with resume support and run-specific directories
"""
import torch
import random
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
import time
import os

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False

    def __call__(self, current_value):
        if self.best_value is None:
            self.best_value = current_value
            return False

        if self.mode == 'min':
            improved = current_value < self.best_value - self.min_delta
        else:
            improved = current_value > self.best_value + self.min_delta

        if improved:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop

    def state_dict(self):
        return {
            'counter': self.counter,
            'best_value': self.best_value,
            'early_stop': self.early_stop
        }
    
    def load_state_dict(self, state):
        self.counter = state['counter']
        self.best_value = state['best_value']
        self.early_stop = state['early_stop']


class ModelCheckpoint:
    """
    Unified checkpointing system with resume support
    - Saves in run-specific directories
    - Maintains best and last checkpoints
    - Handles all modes (train, dev, test)
    """
    def __init__(self, dirpath, run_id, filename="model", monitor='val_loss', 
                 mode='min', save_top_k=1, config=None):
        self.base_dir = Path(dirpath)
        self.run_id = run_id
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.best_k_models = {}
        self.best_value = None
        self.best_other_states = {}  # Store best other_states for threshold persistence
        self.config = config
        self.run_dir = self.base_dir / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def save(self, model, optimizer, scheduler, epoch, fold, value, completed=False, other_states=None):
        """Save checkpoint with full state including scheduler"""
        checkpoint = {
            'epoch': epoch,
            'fold': fold,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,  # Save scheduler state
            'monitor_value': value,
            'config': OmegaConf.to_container(self.config, resolve=True),
            'run_id': self.run_id,
            'completed': completed,
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'numpy_rng_state': np.random.get_state(),
            'random_state': random.getstate(),
            'timestamp': time.time(),
        }
        
        if other_states:
            checkpoint.update(other_states)
        
        # Save last checkpoint
        last_path = self.run_dir / f"{self.filename}_last.pt"
        torch.save(checkpoint, last_path)
        
        # Save best checkpoint
        if self.best_value is None or \
           (self.mode == 'min' and value < self.best_value) or \
           (self.mode == 'max' and value > self.best_value):
            
            self.best_value = value
            if other_states:
                self.best_other_states = other_states.copy()  # Store best other_states
            best_path = self.run_dir / f"{self.filename}_best.pt"
            torch.save(checkpoint, best_path)
            self.best_k_models[best_path] = value
            
            # Maintain top-k models
            if len(self.best_k_models) > self.save_top_k:
                if self.mode == 'min':
                    worst_path = max(self.best_k_models, key=self.best_k_models.get)
                else:
                    worst_path = min(self.best_k_models, key=self.best_k_models.get)
                if worst_path.exists():
                    worst_path.unlink()
                del self.best_k_models[worst_path]


    """
    TODO: add new method to load the best model
    """

    def get_best_model_path(self):
        """Get the path to the best model checkpoint"""
        return self.run_dir / f"{self.filename}_best.pt"
    
    def load_best_model(self, model, optimizer, scheduler, device):
        """
        Load the best model, optimizer, and scheduler states
        """


        best_path = self.run_dir / f"{self.filename}_best.pt"
        if best_path.exists():
            checkpoint = self.load_checkpoint(best_path, device)
            model.load_state_dict(checkpoint['model_state_dict'])
            # # Load model state with strict=False to handle dynamic PLM projection layers
            # missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            # # Log dynamic PLM projection layers (expected to be missing/unexpected)
            # plm_keys = [k for k in unexpected_keys if 'plm_proj' in k]
            # if plm_keys:
            #     print(f"Note: Ignoring unexpected PLM projection keys: {plm_keys}")
            #     unexpected_keys = [k for k in unexpected_keys if 'plm_proj' not in k]
            
            # # Warn about any other unexpected issues
            # if missing_keys:
            #     print(f"Warning: Missing keys when loading model: {missing_keys}")
            # if unexpected_keys:
            #     print(f"Warning: Unexpected keys when loading model: {unexpected_keys}")

            # Thresholds are now fixed from model config - no need to load from checkpoint
            
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            return model, optimizer, scheduler
        else:
            raise FileNotFoundError(f"No best model checkpoint found at {best_path}")
        


    @staticmethod
    def load_checkpoint(path, device, config_override=None):
        """Load checkpoint with device mapping and config override"""
        # checkpoint = torch.load(path, map_location=device)
        
        # # Handle config override for resuming
        # if config_override:
        #     saved_config = OmegaConf.create(checkpoint['config'])
        #     merged_config = OmegaConf.merge(saved_config, config_override)
        #     checkpoint['config'] = OmegaConf.to_container(merged_config, resolve=True)
        
        # # Restore random states
        # torch.set_rng_state(checkpoint['torch_rng_state'])
        # if checkpoint['cuda_rng_state'] and torch.cuda.is_available():
        #     torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
        # np.random.set_state(checkpoint['numpy_rng_state'])
        # random.setstate(checkpoint['random_state'])

        checkpoint = torch.load(path, map_location="cpu")
        
        # Handle config override for resuming
        if config_override:
            saved_config = OmegaConf.create(checkpoint['config'])
            merged_config = OmegaConf.merge(saved_config, config_override)
            checkpoint['config'] = OmegaConf.to_container(merged_config, resolve=True)
        
        # Restore random states
        if 'torch_rng_state' in checkpoint:
            # Convert to ByteTensor if needed
            if not isinstance(checkpoint['torch_rng_state'], torch.ByteTensor):
                checkpoint['torch_rng_state'] = torch.ByteTensor(checkpoint['torch_rng_state'])
            torch.set_rng_state(checkpoint['torch_rng_state'])

        if 'cuda_rng_state' in checkpoint and torch.cuda.is_available():
            if not isinstance(checkpoint['cuda_rng_state'], list):
                checkpoint['cuda_rng_state'] = [checkpoint['cuda_rng_state']]
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])

        # Move model to device
        if 'model_state_dict' in checkpoint:
            for k in list(checkpoint['model_state_dict'].keys()):
                checkpoint['model_state_dict'][k] = checkpoint['model_state_dict'][k].to(device)
        
        
        return checkpoint



