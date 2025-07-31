#!/usr/bin/env python3
"""
Enhanced data loaders for Flexible VAE training.

This module provides robust, phase-aware data loading:
- Phase 1: Self-supervised pretraining on large datasets (no perturbation labels)
- Phase 2: Perturbation fine-tuning with paired control/perturbation data

Features:
- Efficient memory usage with optional sparse matrix support
- Robust pairing validation for Phase 2
- Flexible data formats (npz, h5, csv)
- Data quality checks and filtering
- Balanced sampling strategies

Authors: [Your name]
Created: July 2025
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Set, Union
import scipy.sparse as sp
from pathlib import Path
import logging
import warnings

logger = logging.getLogger(__name__)


class FlexibleDataset(Dataset):
    """
    Base dataset class for Phase 1 pretraining.
    
    Supports large-scale scRNA-seq data without perturbation labels.
    Optimized for memory efficiency and fast loading.
    """
    
    def __init__(self, 
                 expression_data: Union[np.ndarray, sp.spmatrix],
                 experiment_ids: np.ndarray,
                 gene_names: Optional[List[str]] = None,
                 cell_metadata: Optional[pd.DataFrame] = None,
                 normalize: bool = True,
                 log_transform: bool = True,
                 min_genes_per_cell: int = 200,
                 min_cells_per_gene: int = 3):
        """
        Initialize the flexible dataset.
        
        Args:
            expression_data: Gene expression matrix [n_cells, n_genes] 
            experiment_ids: Experiment/batch IDs for each cell
            gene_names: Optional gene names for validation
            cell_metadata: Optional metadata for additional filtering
            normalize: Whether to normalize expression data
            log_transform: Whether to apply log(x + 1) transformation
            min_genes_per_cell: Filter cells with too few expressed genes
            min_cells_per_gene: Filter genes expressed in too few cells
        """
        self.gene_names = gene_names
        self.cell_metadata = cell_metadata
        
        # Convert sparse to dense if needed for filtering
        if sp.issparse(expression_data):
            self.is_sparse = True
            # Apply basic filtering first
            expression_data = self._filter_sparse_data(
                expression_data, min_genes_per_cell, min_cells_per_gene
            )
        else:
            self.is_sparse = False
            expression_data = self._filter_dense_data(
                expression_data, min_genes_per_cell, min_cells_per_gene
            )
        
        # Preprocessing
        if normalize:
            expression_data = self._normalize_data(expression_data)
        
        if log_transform:
            if self.is_sparse:
                expression_data.data = np.log1p(expression_data.data)
            else:
                expression_data = np.log1p(expression_data)
        
        # Store as tensors
        if self.is_sparse:
            # Convert to dense for PyTorch (can optimize later with sparse tensors)
            self.expression_data = torch.FloatTensor(expression_data.toarray())
        else:
            self.expression_data = torch.FloatTensor(expression_data)
        
        self.experiment_ids = torch.LongTensor(experiment_ids)
        
        # Validate dimensions
        assert len(self.expression_data) == len(self.experiment_ids), \
            "Mismatch between expression and experiment ID dimensions"
        
        logger.info(f"Dataset initialized: {len(self)} cells, {self.expression_data.shape[1]} genes")
        logger.info(f"Experiments: {len(torch.unique(self.experiment_ids))}")
    
    def _filter_sparse_data(self, data: sp.spmatrix, min_genes: int, min_cells: int):
        """Filter sparse expression data."""
        # Filter cells (rows) with too few expressed genes
        genes_per_cell = np.array((data > 0).sum(axis=1)).flatten()
        cell_mask = genes_per_cell >= min_genes
        data = data[cell_mask]
        
        # Filter genes (columns) expressed in too few cells
        cells_per_gene = np.array((data > 0).sum(axis=0)).flatten()
        gene_mask = cells_per_gene >= min_cells
        data = data[:, gene_mask]
        
        logger.info(f"Filtered to {data.shape[0]} cells and {data.shape[1]} genes")
        return data
    
    def _filter_dense_data(self, data: np.ndarray, min_genes: int, min_cells: int):
        """Filter dense expression data."""
        # Filter cells with too few expressed genes
        genes_per_cell = (data > 0).sum(axis=1)
        cell_mask = genes_per_cell >= min_genes
        data = data[cell_mask]
        
        # Filter genes expressed in too few cells
        cells_per_gene = (data > 0).sum(axis=0)
        gene_mask = cells_per_gene >= min_cells
        data = data[:, gene_mask]
        
        logger.info(f"Filtered to {data.shape[0]} cells and {data.shape[1]} genes")
        return data
    
    def _normalize_data(self, data: Union[np.ndarray, sp.spmatrix]):
        """Normalize expression data to counts per 10,000."""
        if sp.issparse(data):
            # Sparse normalization
            data = data.copy()
            counts_per_cell = np.array(data.sum(axis=1)).flatten()
            data.data = data.data / counts_per_cell[data.nonzero()[0]] * 10000
        else:
            # Dense normalization
            counts_per_cell = data.sum(axis=1, keepdims=True)
            data = data / counts_per_cell * 10000
        
        return data
    
    def __len__(self):
        return len(self.expression_data)
    
    def __getitem__(self, idx):
        return {
            'expression': self.expression_data[idx],
            'experiment_id': self.experiment_ids[idx]
        }


class PairedPerturbationDataset(Dataset):
    """
    Enhanced dataset for Phase 2 perturbation training.
    
    Ensures robust pairing between control and perturbation conditions,
    with careful validation and balanced sampling.
    """
    
    def __init__(self, 
                 expression_data: Union[np.ndarray, sp.spmatrix],
                 experiment_ids: np.ndarray,
                 perturbation_ids: np.ndarray,
                 target_gene_ids: Optional[np.ndarray] = None,
                 control_perturbation_id: int = 0,
                 min_cells_per_condition: int = 10,
                 max_pairs_per_experiment: Optional[int] = None,
                 balance_pairs: bool = True,
                 normalize: bool = True,
                 log_transform: bool = True):
        """
        Initialize the paired perturbation dataset.
        
        Args:
            expression_data: Gene expression matrix [n_cells, n_genes]
            experiment_ids: Experiment/batch IDs for each cell
            perturbation_ids: Perturbation IDs for each cell  
            target_gene_ids: Target gene IDs (if different from perturbation_ids)
            control_perturbation_id: ID representing control condition
            min_cells_per_condition: Minimum cells required per condition
            max_pairs_per_experiment: Maximum pairs to keep per experiment
            balance_pairs: Whether to balance control/perturbation ratios
            normalize: Whether to normalize expression data
            log_transform: Whether to apply log transformation
        """
        self.control_perturbation_id = control_perturbation_id
        self.min_cells_per_condition = min_cells_per_condition
        
        # Use perturbation_ids as target_gene_ids if not provided
        if target_gene_ids is None:
            target_gene_ids = perturbation_ids
            
        # Data preprocessing
        if sp.issparse(expression_data):
            expression_data = expression_data.toarray()
        
        if normalize:
            counts_per_cell = expression_data.sum(axis=1, keepdims=True)
            expression_data = expression_data / counts_per_cell * 10000
        
        if log_transform:
            expression_data = np.log1p(expression_data)
        
        # Store original data
        self.all_expression = torch.FloatTensor(expression_data)
        self.all_experiment_ids = experiment_ids
        self.all_perturbation_ids = perturbation_ids
        self.all_target_gene_ids = target_gene_ids
        
        # Find and validate pairs
        self.valid_pairs = self._find_valid_pairs()
        
        # Create pair mappings
        self.pair_data = self._create_pair_mappings(max_pairs_per_experiment, balance_pairs)
        
        logger.info(f"PairedDataset initialized: {len(self.pair_data)} training pairs")
        logger.info(f"Valid experiment-perturbation combinations: {len(self.valid_pairs)}")
    
    def _find_valid_pairs(self) -> List[Tuple[int, int]]:
        """
        Find valid (experiment, perturbation) pairs where both control and 
        perturbation conditions have sufficient cells.
        """
        valid_pairs = []
        
        # Group by experiment
        unique_experiments = np.unique(self.all_experiment_ids)
        
        for exp_id in unique_experiments:
            exp_mask = self.all_experiment_ids == exp_id
            exp_perturbations = self.all_perturbation_ids[exp_mask]
            
            # Check if control exists in this experiment
            control_count = np.sum(exp_perturbations == self.control_perturbation_id)
            if control_count < self.min_cells_per_condition:
                continue
            
            # Check each perturbation in this experiment
            unique_perturbations = np.unique(exp_perturbations)
            for pert_id in unique_perturbations:
                if pert_id == self.control_perturbation_id:
                    continue  # Skip control
                
                pert_count = np.sum(exp_perturbations == pert_id)
                if pert_count >= self.min_cells_per_condition:
                    valid_pairs.append((exp_id, pert_id))
        
        logger.info(f"Found {len(valid_pairs)} valid experiment-perturbation pairs")
        return valid_pairs
    
    def _create_pair_mappings(self, max_pairs: Optional[int], balance: bool) -> List[Dict]:
        """Create training examples from valid pairs."""
        pair_data = []
        
        for exp_id, pert_id in self.valid_pairs:
            # Get control and perturbation cell indices
            exp_mask = self.all_experiment_ids == exp_id
            
            control_mask = exp_mask & (self.all_perturbation_ids == self.control_perturbation_id)
            pert_mask = exp_mask & (self.all_perturbation_ids == pert_id)
            
            control_indices = np.where(control_mask)[0]
            pert_indices = np.where(pert_mask)[0]
            
            # Get target gene ID for this perturbation
            target_gene_id = self.all_target_gene_ids[pert_mask][0]  # Should be same for all
            
            # Create all possible pairs
            if balance:
                # Balance by taking equal numbers of control/perturbation
                n_pairs = min(len(control_indices), len(pert_indices))
                control_indices = np.random.choice(control_indices, n_pairs, replace=False)
                pert_indices = np.random.choice(pert_indices, n_pairs, replace=False)
            
            # Limit pairs per experiment if specified
            if max_pairs and len(control_indices) > max_pairs:
                selected = np.random.choice(len(control_indices), max_pairs, replace=False)
                control_indices = control_indices[selected]
                pert_indices = pert_indices[selected]
            
            # Create pair mappings
            for ctrl_idx, pert_idx in zip(control_indices, pert_indices):
                pair_data.append({
                    'control_idx': ctrl_idx,
                    'perturbed_idx': pert_idx,
                    'experiment_id': exp_id,
                    'perturbation_id': pert_id,
                    'target_gene_id': target_gene_id
                })
        
        return pair_data
    
    def __len__(self):
        return len(self.pair_data)
    
    def __getitem__(self, idx):
        pair = self.pair_data[idx]
        
        return {
            'control_expression': self.all_expression[pair['control_idx']],
            'perturbed_expression': self.all_expression[pair['perturbed_idx']],
            'experiment_id': torch.LongTensor([pair['experiment_id']]).squeeze(),
            'perturbation_id': torch.LongTensor([pair['perturbation_id']]).squeeze(),
            'target_gene_id': torch.LongTensor([pair['target_gene_id']]).squeeze()
        }


def load_data_from_file(file_path: str, 
                       format_type: str = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load expression data from various file formats.
    
    Args:
        file_path: Path to data file
        format_type: File format ('npz', 'h5', 'csv', 'tsv') - auto-detected if None
        
    Returns:
        Tuple of (expression_data, experiment_ids, metadata_dict)
    """
    file_path = Path(file_path)
    
    if format_type is None:
        format_type = file_path.suffix.lower()
    
    if format_type == '.npz':
        data = np.load(file_path)
        expression_data = data['expression']
        experiment_ids = data['experiment_ids']
        
        # Load optional metadata
        metadata = {}
        for key in data.keys():
            if key not in ['expression', 'experiment_ids']:
                metadata[key] = data[key]
                
        return expression_data, experiment_ids, metadata
    
    elif format_type in ['.h5', '.hdf5']:
        import h5py
        with h5py.File(file_path, 'r') as f:
            expression_data = f['expression'][:]
            experiment_ids = f['experiment_ids'][:]
            
            metadata = {}
            for key in f.keys():
                if key not in ['expression', 'experiment_ids']:
                    metadata[key] = f[key][:]
        
        return expression_data, experiment_ids, metadata
    
    elif format_type in ['.csv', '.tsv']:
        separator = '\t' if format_type == '.tsv' else ','
        df = pd.read_csv(file_path, sep=separator, index_col=0)
        
        # Assume last column is experiment_ids, rest is expression
        expression_data = df.iloc[:, :-1].values
        experiment_ids = df.iloc[:, -1].values
        
        metadata = {'gene_names': df.columns[:-1].tolist()}
        
        return expression_data, experiment_ids, metadata
    
    else:
        raise ValueError(f"Unsupported file format: {format_type}")


def create_flexible_dataloaders(data_path: str,
                              phase: int,
                              batch_size: int = 256,
                              val_split: float = 0.1,
                              test_split: float = 0.1,
                              **dataset_kwargs) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create data loaders for flexible VAE training.
    
    Args:
        data_path: Path to data file
        phase: Training phase (1 or 2)
        batch_size: Batch size for data loaders
        val_split: Fraction for validation set
        test_split: Fraction for test set (optional)
        **dataset_kwargs: Additional arguments for dataset initialization
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load data
    expression_data, experiment_ids, metadata = load_data_from_file(data_path)
    
    if phase == 1:
        # Phase 1: Pretraining dataset
        dataset = FlexibleDataset(
            expression_data=expression_data,
            experiment_ids=experiment_ids,
            gene_names=metadata.get('gene_names'),
            **dataset_kwargs
        )
        
    elif phase == 2:
        # Phase 2: Paired perturbation dataset
        if 'perturbation_ids' not in metadata:
            raise ValueError("Phase 2 requires perturbation_ids in data")
        
        dataset = PairedPerturbationDataset(
            expression_data=expression_data,
            experiment_ids=experiment_ids,
            perturbation_ids=metadata['perturbation_ids'],
            target_gene_ids=metadata.get('target_gene_ids'),
            **dataset_kwargs
        )
    
    else:
        raise ValueError(f"Unsupported phase: {phase}")
    
    # Split dataset
    total_size = len(dataset)
    test_size = int(total_size * test_split) if test_split > 0 else 0
    val_size = int(total_size * val_split)
    train_size = total_size - val_size - test_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size, test_size] if test_size > 0 else [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = None
    if test_size > 0:
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
    
    logger.info(f"Created data loaders: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader) if test_loader else 0}")
    
    return train_loader, val_loader, test_loader


# Convenience functions for backward compatibility
def create_vae_dataloaders(*args, **kwargs):
    """Backward compatibility wrapper."""
    warnings.warn("create_vae_dataloaders is deprecated. Use create_flexible_dataloaders instead.")
    return create_flexible_dataloaders(*args, **kwargs)


# Legacy class alias
VAEPairedDataset = PairedPerturbationDataset
