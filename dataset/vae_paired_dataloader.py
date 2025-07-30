#!/usr/bin/env python3
"""
Paired data loader for VAE training.

This module implements a data loader that ensures training only on pairs where both 
control and perturbed data exist for a given experiment/perturbation combination.
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Set
import scipy.sparse as sp
from pathlib import Path


class VAEPairedDataset(Dataset):
    """
    Dataset for VAE training with paired control/perturbation data.
    
    This dataset ensures that for each training example, there exists both control 
    and perturbed data for the same experiment/perturbation combination.
    """
    
    def __init__(self, 
                 expression_data: np.ndarray,
                 experiment_ids: np.ndarray,
                 perturbation_ids: np.ndarray,
                 control_perturbation_id: int = 0,
                 min_cells_per_pair: int = 5):
        """
        Initialize the paired dataset.
        
        Args:
            expression_data: Gene expression matrix [n_cells, n_genes]
            experiment_ids: Experiment IDs (SRX accessions) for each cell
            perturbation_ids: Perturbation IDs for each cell
            control_perturbation_id: ID representing control/untreated condition
            min_cells_per_pair: Minimum number of cells required for each condition in a pair
        """
        self.expression_data = torch.FloatTensor(expression_data)
        self.experiment_ids = torch.LongTensor(experiment_ids)
        self.perturbation_ids = torch.LongTensor(perturbation_ids)
        self.control_perturbation_id = control_perturbation_id
        
        # Find valid pairs (experiment, perturbation) where both control and perturbed exist
        self.valid_pairs = self._find_valid_pairs(min_cells_per_pair)
        
        # Create mapping from pair to cell indices
        self.pair_to_cells = self._create_pair_mapping()
        
        # Create balanced dataset indices
        self.valid_indices = self._create_balanced_indices()
        
        print(f"Found {len(self.valid_pairs)} valid experiment-perturbation pairs")
        print(f"Total training examples: {len(self.valid_indices)}")
    
    def _find_valid_pairs(self, min_cells_per_pair: int) -> List[Tuple[int, int]]:
        """Find experiment-perturbation pairs with both control and perturbed data."""
        # Count cells for each (experiment, perturbation) combination
        pair_counts = {}
        for i in range(len(self.experiment_ids)):
            exp_id = self.experiment_ids[i].item()
            pert_id = self.perturbation_ids[i].item()
            key = (exp_id, pert_id)
            pair_counts[key] = pair_counts.get(key, 0) + 1
        
        # Find experiments with both control and at least one perturbation
        valid_pairs = []
        experiments_with_controls = set()
        
        # First, find experiments that have controls
        for (exp_id, pert_id), count in pair_counts.items():
            if pert_id == self.control_perturbation_id and count >= min_cells_per_pair:
                experiments_with_controls.add(exp_id)
        
        # Then, find perturbations in those experiments
        for (exp_id, pert_id), count in pair_counts.items():
            if (exp_id in experiments_with_controls and 
                pert_id != self.control_perturbation_id and 
                count >= min_cells_per_pair):
                valid_pairs.append((exp_id, pert_id))
        
        return valid_pairs
    
    def _create_pair_mapping(self) -> Dict[Tuple[int, int], List[int]]:
        """Create mapping from (experiment, perturbation) pairs to cell indices."""
        pair_to_cells = {}
        
        for i in range(len(self.experiment_ids)):
            exp_id = self.experiment_ids[i].item()
            pert_id = self.perturbation_ids[i].item()
            key = (exp_id, pert_id)
            
            if key not in pair_to_cells:
                pair_to_cells[key] = []
            pair_to_cells[key].append(i)
        
        return pair_to_cells
    
    def _create_balanced_indices(self) -> List[int]:
        """Create balanced dataset with equal representation of control and perturbed."""
        valid_indices = []
        
        for exp_id, pert_id in self.valid_pairs:
            # Get control cells for this experiment
            control_key = (exp_id, self.control_perturbation_id)
            control_indices = self.pair_to_cells.get(control_key, [])
            
            # Get perturbed cells for this experiment-perturbation
            pert_key = (exp_id, pert_id)
            pert_indices = self.pair_to_cells.get(pert_key, [])
            
            # Add all available cells from both conditions
            valid_indices.extend(control_indices)
            valid_indices.extend(pert_indices)
        
        return valid_indices
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training example."""
        cell_idx = self.valid_indices[idx]
        
        return {
            'expression': self.expression_data[cell_idx],
            'experiment_id': self.experiment_ids[cell_idx],
            'perturbation_id': self.perturbation_ids[cell_idx]
        }
    
    def get_paired_examples(self, exp_id: int, pert_id: int, n_samples: int = 10) -> Dict[str, torch.Tensor]:
        """
        Get paired control and perturbed examples for a specific experiment-perturbation.
        
        Args:
            exp_id: Experiment ID
            pert_id: Perturbation ID (not control)
            n_samples: Number of examples to sample from each condition
            
        Returns:
            Dictionary with control and perturbed examples
        """
        # Get control cells
        control_key = (exp_id, self.control_perturbation_id)
        control_indices = self.pair_to_cells.get(control_key, [])
        
        # Get perturbed cells
        pert_key = (exp_id, pert_id)
        pert_indices = self.pair_to_cells.get(pert_key, [])
        
        if not control_indices or not pert_indices:
            raise ValueError(f"No paired data found for experiment {exp_id}, perturbation {pert_id}")
        
        # Sample cells
        n_control = min(n_samples, len(control_indices))
        n_pert = min(n_samples, len(pert_indices))
        
        control_sample = np.random.choice(control_indices, n_control, replace=False)
        pert_sample = np.random.choice(pert_indices, n_pert, replace=False)
        
        return {
            'control_expression': self.expression_data[control_sample],
            'control_experiment_ids': self.experiment_ids[control_sample],
            'control_perturbation_ids': self.perturbation_ids[control_sample],
            'perturbed_expression': self.expression_data[pert_sample],
            'perturbed_experiment_ids': self.experiment_ids[pert_sample],
            'perturbed_perturbation_ids': self.perturbation_ids[pert_sample]
        }


def load_vae_data(data_path: str, 
                  expression_key: str = 'expression',
                  experiment_key: str = 'SRX_accession',
                  perturbation_key: str = 'perturbation') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data from the batch file format for VAE training.
    
    Args:
        data_path: Path to the data file (.npz format from download.py)
        expression_key: Key for expression data in the file
        experiment_key: Key for experiment IDs in the file
        perturbation_key: Key for perturbation IDs in the file
        
    Returns:
        Tuple of (expression_data, experiment_ids, perturbation_ids)
    """
    data = np.load(data_path, allow_pickle=True)
    
    # Load expression data
    if expression_key in data:
        expression_data = data[expression_key]
        if sp.issparse(expression_data):
            expression_data = expression_data.toarray()
    else:
        raise KeyError(f"Expression data key '{expression_key}' not found in {data_path}")
    
    # Load metadata
    if 'obs' in data:
        obs = data['obs'].item()  # obs is stored as a dictionary
        
        if experiment_key in obs:
            experiment_ids = obs[experiment_key]
        else:
            raise KeyError(f"Experiment key '{experiment_key}' not found in obs")
            
        if perturbation_key in obs:
            perturbation_ids = obs[perturbation_key]
        else:
            raise KeyError(f"Perturbation key '{perturbation_key}' not found in obs")
    else:
        raise KeyError("'obs' metadata not found in data file")
    
    return expression_data, experiment_ids, perturbation_ids


def create_vae_dataloaders(data_path: str,
                          batch_size: int = 256,
                          test_split: float = 0.2,
                          val_split: float = 0.1,
                          min_cells_per_pair: int = 5,
                          num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/validation/test data loaders for VAE training.
    
    Args:
        data_path: Path to the data file
        batch_size: Batch size for training
        test_split: Fraction of data to use for testing
        val_split: Fraction of remaining data to use for validation
        min_cells_per_pair: Minimum cells required per condition pair
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load data
    expression_data, experiment_ids, perturbation_ids = load_vae_data(data_path)
    
    # Create dataset
    dataset = VAEPairedDataset(
        expression_data=expression_data,
        experiment_ids=experiment_ids,
        perturbation_ids=perturbation_ids,
        min_cells_per_pair=min_cells_per_pair
    )
    
    # Split dataset
    n_total = len(dataset)
    n_test = int(test_split * n_total)
    n_val = int(val_split * (n_total - n_test))
    n_train = n_total - n_test - n_val
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, 
                       help='Path to the data file (.npz format)')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--min_cells_per_pair', type=int, default=5)
    
    args = parser.parse_args()
    
    # Test data loading
    try:
        train_loader, val_loader, test_loader = create_vae_dataloaders(
            args.data_path,
            batch_size=args.batch_size,
            min_cells_per_pair=args.min_cells_per_pair
        )
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Test batch loading
        batch = next(iter(train_loader))
        print(f"Batch shapes:")
        print(f"  Expression: {batch['expression'].shape}")
        print(f"  Experiment IDs: {batch['experiment_id'].shape}")
        print(f"  Perturbation IDs: {batch['perturbation_id'].shape}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
