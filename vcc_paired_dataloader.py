#!/usr/bin/env python3
"""
VCC Paired DataLoader that loads control and perturbed cell sets together.
This is designed for training models that use control cells as conditioning.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import scanpy as sc
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import os
import json


class VCCPairedDataset(Dataset):
    """
    Dataset that returns paired control and perturbed cell sets.
    Each item contains a set of control cells and a set of perturbed cells
    for the same target gene.
    """
    
    def __init__(
        self,
        data_path: str,
        set_size: int = 16,  # Number of cells per set
        tokenizer=None,  # Tokenizer for converting expressions to tokens
        max_cells: Optional[int] = None,
        match_by_batch: bool = True,  # Whether to match controls from same batch
        use_hvgs: bool = True,  # Whether to use only HVG genes
    ):
        """
        Args:
            data_path: Path to adata_Training.h5ad
            set_size: Number of cells to sample per set
            tokenizer: Optional tokenizer for discretizing expression values
            max_cells: Maximum cells to load (for debugging)
            match_by_batch: Whether to match controls from the same batch
            use_hvgs: Whether to use only highly variable genes
        """
        print(f"Loading VCC paired data from: {data_path}")
        self.adata = sc.read_h5ad(data_path)
        self.set_size = set_size
        self.tokenizer = tokenizer
        self.match_by_batch = match_by_batch
        self.use_hvgs = use_hvgs
        
        # Load HVG information if requested
        self.hvg_indices = None
        self.hvg_names = None
        self.gene_name_to_hvg_idx = None
        
        if use_hvgs:
            hvg_info_path = Path(data_path).parent / 'hvg_info.json'
            if not hvg_info_path.exists():
                raise FileNotFoundError(
                    f"HVG info not found at {hvg_info_path}. "
                    "Please run preprocess_hvgs.py --process-vcc first."
                )
            
            with open(hvg_info_path, 'r') as f:
                hvg_info = json.load(f)
            
            self.hvg_indices = hvg_info['hvg_indices']
            self.hvg_names = hvg_info['hvg_names']
            self.gene_name_to_hvg_idx = hvg_info['gene_name_to_hvg_idx']
            
            print(f"Using {len(self.hvg_indices)} HVG genes")
            
            # Filter genes to only HVGs
            self.adata = self.adata[:, self.hvg_indices]
            self.gene_names = self.hvg_names
        else:
            self.gene_names = self.adata.var.index.tolist()
            
        if max_cells is not None and len(self.adata) > max_cells:
            self.adata = self.adata[:max_cells]
            
        # Build indices
        self._build_indices()
        self._build_perturbation_groups()
        
    def _build_indices(self):
        """Build indices for fast access to control and perturbed cells."""
        self.control_indices = []
        self.perturbed_indices = []
        self.batch_control_indices = {}  # batch -> list of control indices
        
        for idx in range(len(self.adata)):
            if self.adata.obs.iloc[idx]['target_gene'] == 'non-targeting':
                self.control_indices.append(idx)
                batch = self.adata.obs.iloc[idx]['batch']
                if batch not in self.batch_control_indices:
                    self.batch_control_indices[batch] = []
                self.batch_control_indices[batch].append(idx)
            else:
                self.perturbed_indices.append(idx)
                
        print(f"Found {len(self.control_indices)} control cells and "
              f"{len(self.perturbed_indices)} perturbed cells")
        
    def _build_perturbation_groups(self):
        """Group perturbed cells by target gene."""
        self.perturbation_groups = {}
        self.perturbation_batches = {}
        
        for idx in self.perturbed_indices:
            target_gene = self.adata.obs.iloc[idx]['target_gene']
            batch = self.adata.obs.iloc[idx]['batch']
            
            if target_gene not in self.perturbation_groups:
                self.perturbation_groups[target_gene] = []
                self.perturbation_batches[target_gene] = {}
                
            self.perturbation_groups[target_gene].append(idx)
            
            if batch not in self.perturbation_batches[target_gene]:
                self.perturbation_batches[target_gene][batch] = []
            self.perturbation_batches[target_gene][batch].append(idx)
            
        # Filter out genes with too few cells
        min_cells = self.set_size * 2  # Need at least 2 sets worth
        self.valid_genes = [
            gene for gene, indices in self.perturbation_groups.items()
            if len(indices) >= min_cells
        ]
        
        print(f"Found {len(self.valid_genes)} genes with enough perturbed cells")
        
    def __len__(self):
        return len(self.valid_genes)
    
    def __getitem__(self, idx):
        """
        Return a paired set of control and perturbed cells.
        
        Returns:
            dict with:
                - control_expr: (set_size, n_genes) control cell expressions
                - perturbed_expr: (set_size, n_genes) perturbed cell expressions
                - target_gene: Name of the perturbed gene
                - target_gene_idx: Index of target gene in expression matrix
                - control_mean: Mean expression of control set
                - log2fc: Log2 fold change (perturbed_mean / control_mean)
                - perturbation_magnitude: L2 norm of the perturbation effect
        """
        target_gene = self.valid_genes[idx]
        
        # Get perturbed cells
        pert_indices = self.perturbation_groups[target_gene]
        if len(pert_indices) >= self.set_size:
            selected_pert = np.random.choice(pert_indices, self.set_size, replace=False)
        else:
            selected_pert = np.random.choice(pert_indices, self.set_size, replace=True)
            
        # Get control cells (preferably from same batches)
        if self.match_by_batch:
            # Try to get controls from the same batches as perturbed cells
            control_pool = []
            for pidx in selected_pert:
                batch = self.adata.obs.iloc[pidx]['batch']
                if batch in self.batch_control_indices:
                    control_pool.extend(self.batch_control_indices[batch])
            
            # If not enough batch-matched controls, use all controls
            if len(control_pool) < self.set_size:
                control_pool = self.control_indices
        else:
            control_pool = self.control_indices
            
        if len(control_pool) >= self.set_size:
            selected_ctrl = np.random.choice(control_pool, self.set_size, replace=False)
        else:
            selected_ctrl = np.random.choice(control_pool, self.set_size, replace=True)
            
        # Extract expressions
        if hasattr(self.adata.X, 'toarray'):
            control_expr = self.adata.X[selected_ctrl].toarray()
            pert_expr = self.adata.X[selected_pert].toarray()
        else:
            control_expr = self.adata.X[selected_ctrl]
            pert_expr = self.adata.X[selected_pert]
            
        # Compute statistics
        control_mean = control_expr.mean(axis=0)
        pert_mean = pert_expr.mean(axis=0)
        log2fc = np.log2((pert_mean + 1) / (control_mean + 1))
        
        # Get target gene index
        if self.use_hvgs and self.gene_name_to_hvg_idx:
            target_idx = self.gene_name_to_hvg_idx.get(target_gene, -1)
        else:
            try:
                target_idx = self.gene_names.index(target_gene)
            except ValueError:
                target_idx = -1
        
        # Apply tokenizer if provided
        if self.tokenizer is not None:
            control_expr = self.tokenizer(control_expr)
            pert_expr = self.tokenizer(pert_expr)
            
        result = {
            'control_expr': torch.FloatTensor(control_expr),
            'perturbed_expr': torch.FloatTensor(pert_expr),
            'target_gene': target_gene,
            'target_gene_idx': target_idx,
            'control_mean': torch.FloatTensor(control_mean),
            'log2fc': torch.FloatTensor(log2fc),
            'perturbation_magnitude': float(np.linalg.norm(log2fc)),
        }
        
        return result
    
    def get_control_mean(self):
        """Get overall mean of control cells."""
        if hasattr(self.adata.X[self.control_indices], 'toarray'):
            return self.adata.X[self.control_indices].toarray().mean(axis=0)
        else:
            return self.adata.X[self.control_indices].mean(axis=0)


class VCCValidationDataset(Dataset):
    """
    Dataset for validation that focuses on specific genes from a CSV file.
    """
    
    def __init__(
        self,
        vcc_dataset,  # VCCDataset instance
        validation_csv: str,
        set_size: int = 16,
        tokenizer=None,
        n_samples_per_gene: int = 100,  # How many samples to generate per gene
    ):
        """
        Args:
            vcc_dataset: Base VCCDataset instance (should use HVGs)
            validation_csv: Path to CSV with validation genes
            set_size: Number of cells per set
            tokenizer: Optional tokenizer
            n_samples_per_gene: Number of samples to generate per gene
        """
        from vcc_dataloader import VCCDataset
        self.vcc_dataset = vcc_dataset
        self.set_size = set_size
        self.tokenizer = tokenizer
        self.n_samples_per_gene = n_samples_per_gene
        
        # Load validation genes
        self.val_df = pd.read_csv(validation_csv)
        self.validation_genes = self.val_df['target_gene'].tolist()
        
        # Build control indices
        self.control_indices = self.vcc_dataset.get_control_cells()
        
        # Check which validation genes have data
        self.available_genes = []
        for gene in self.validation_genes:
            pert_cells = self.vcc_dataset.get_perturbed_cells(gene)
            if len(pert_cells) >= self.set_size:
                self.available_genes.append(gene)
                
        print(f"Found {len(self.available_genes)}/{len(self.validation_genes)} "
              f"validation genes with enough data")
        
        # Create fixed indices for reproducible validation
        self.samples = []
        for gene in self.available_genes:
            for _ in range(self.n_samples_per_gene):
                self.samples.append(gene)
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a validation sample."""
        target_gene = self.samples[idx]
        
        # Get perturbed cells for this gene
        pert_indices = self.vcc_dataset.get_perturbed_cells(target_gene)
        selected_pert = np.random.choice(pert_indices, self.set_size, replace=False)
        
        # Get random control cells
        selected_ctrl = np.random.choice(self.control_indices, self.set_size, replace=False)
        
        # Extract expressions
        control_expr = self.vcc_dataset.expression[selected_ctrl]
        pert_expr = self.vcc_dataset.expression[selected_pert]
        
        # Compute statistics
        control_mean = control_expr.mean(axis=0)
        pert_mean = pert_expr.mean(axis=0)
        log2fc = np.log2((pert_mean + 1) / (control_mean + 1))
        
        # Get target gene index
        target_idx = self.vcc_dataset.get_gene_index(target_gene)
        if target_idx is None:
            target_idx = -1
            
        # Apply tokenizer if provided
        if self.tokenizer is not None:
            control_expr = self.tokenizer(control_expr)
            pert_expr = self.tokenizer(pert_expr)
            
        return {
            'control_expr': torch.FloatTensor(control_expr),
            'perturbed_expr': torch.FloatTensor(pert_expr),
            'target_gene': target_gene,
            'target_gene_idx': target_idx,
            'control_mean': torch.FloatTensor(control_mean),
            'log2fc': torch.FloatTensor(log2fc),
            'perturbation_magnitude': float(np.linalg.norm(log2fc)),
        }


def create_vcc_paired_dataloader(
    data_dir: Optional[str] = None,
    set_size: int = 16,
    batch_size: int = 32,
    tokenizer=None,
    num_workers: int = 4,
    max_cells: Optional[int] = None,
    match_by_batch: bool = True,
    use_hvgs: bool = True,
) -> Tuple[VCCPairedDataset, DataLoader]:
    """
    Create a paired dataloader for VCC data.
    
    Args:
        data_dir: Directory containing VCC data
        set_size: Number of cells per set
        batch_size: Batch size for DataLoader
        tokenizer: Optional tokenizer for discretizing expressions
        num_workers: Number of workers for DataLoader
        max_cells: Maximum cells to load (for debugging)
        match_by_batch: Whether to match controls from same batch
        use_hvgs: Whether to use only HVG genes
        
    Returns:
        dataset: VCCPairedDataset instance
        dataloader: DataLoader instance
    """
    # Find data directory if not provided
    if data_dir is None:
        from vcc_dataloader import find_vcc_data_dir
        data_dir = find_vcc_data_dir()
        if data_dir is None:
            raise FileNotFoundError(
                "Could not find VCC data directory. Please provide data_dir or "
                "set VCC_DATA_DIR environment variable."
            )
    
    data_path = Path(data_dir) / "adata_Training.h5ad"
    
    dataset = VCCPairedDataset(
        str(data_path),
        set_size=set_size,
        tokenizer=tokenizer,
        max_cells=max_cells,
        match_by_batch=match_by_batch,
        use_hvgs=use_hvgs,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return dataset, dataloader


def create_vcc_validation_dataloader(
    data_dir: Optional[str] = None,
    set_size: int = 16,
    batch_size: int = 32,
    tokenizer=None,
    n_samples_per_gene: int = 100,
    max_cells: Optional[int] = None,
    use_hvgs: bool = True,
) -> Tuple[VCCValidationDataset, DataLoader]:
    """
    Create a validation dataloader for VCC data.
    
    Args:
        data_dir: Directory containing VCC data
        set_size: Number of cells per set
        batch_size: Batch size for DataLoader
        tokenizer: Optional tokenizer
        n_samples_per_gene: Number of samples per validation gene
        max_cells: Maximum cells to load (for debugging)
        use_hvgs: Whether to use only HVG genes
        
    Returns:
        dataset: VCCValidationDataset instance
        dataloader: DataLoader instance
    """
    from vcc_dataloader import VCCDataset, find_vcc_data_dir
    
    # Find data directory
    if data_dir is None:
        data_dir = find_vcc_data_dir()
        if data_dir is None:
            raise FileNotFoundError("Could not find VCC data directory")
            
    # Load base dataset
    data_path = Path(data_dir) / "adata_Training.h5ad"
    vcc_dataset = VCCDataset(str(data_path), max_cells=max_cells, use_hvgs=use_hvgs)
    
    # Create validation dataset
    validation_csv = Path(data_dir) / "pert_counts_Validation.csv"
    if not validation_csv.exists():
        raise FileNotFoundError(f"Validation CSV not found: {validation_csv}")
        
    dataset = VCCValidationDataset(
        vcc_dataset,
        str(validation_csv),
        set_size=set_size,
        tokenizer=tokenizer,
        n_samples_per_gene=n_samples_per_gene
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep validation order consistent
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return dataset, dataloader


if __name__ == "__main__":
    # Test the paired dataloader
    print("Testing VCC Paired DataLoader...")
    
    try:
        # Create dummy tokenizer for testing
        class DummyTokenizer:
            def __call__(self, x):
                # Simple binning tokenizer
                return (x * 10).long().clamp(0, 63)
        
        tokenizer = [DummyTokenizer()]
        
        # Test paired dataset
        dataset, dataloader = create_vcc_paired_dataloader(
            set_size=8,
            batch_size=4,
            tokenizer=tokenizer,
            max_cells=10000
        )
        
        print(f"\nDataset info:")
        print(f"  Total perturbation groups: {len(dataset)}")
        
        # Test single sample
        sample = dataset[0]
        print(f"\nSample:")
        print(f"  Target gene: {sample['target_gene']}")
        print(f"  Control shape: {sample['control_expr'].shape}")
        print(f"  Perturbed shape: {sample['perturbed_expr'].shape}")
        print(f"  Log2FC mean: {sample['log2fc'].mean():.3f}")
        print(f"  Perturbation magnitude: {sample['perturbation_magnitude']:.3f}")
        
        # Test batch
        for batch in dataloader:
            print(f"\nBatch:")
            print(f"  Control shape: {batch['control_expr'].shape}")
            print(f"  Perturbed shape: {batch['perturbed_expr'].shape}")
            print(f"  Target genes: {batch['target_gene']}")
            print(f"  Log2FC: {batch['log2fc']}")
            break
        
        # Test validation dataset
        print("\n" + "="*50)
        print("Testing validation dataset...")
        
        val_dataset, val_dataloader = create_vcc_validation_dataloader(
            set_size=8,
            batch_size=4,
            tokenizer=tokenizer,
            n_samples_per_gene=10,
            max_cells=5000
        )
        
        val_sample = val_dataset[0]
        print(f"\nValidation sample:")
        print(f"  Target gene: {val_sample['target_gene']}")
        print(f"  Control shape: {val_sample['control_expr'].shape}")
        print(f"  Perturbed shape: {val_sample['perturbed_expr'].shape}")
        print(f"  Perturbation magnitude: {val_sample['perturbation_magnitude']:.3f}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc() 