#!/usr/bin/env python3
"""
Simple VCC DataLoader for fine-tuning experiments.
Designed to be easily modified and extended.
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
import hashlib
from datetime import datetime


def load_hvg_info_with_cache(data_path: str) -> Dict:
    """
    Load HVG information with caching for faster subsequent loads.
    
    Args:
        data_path: Path to the h5ad file
        
    Returns:
        Dictionary containing hvg_indices, hvg_names, and gene_name_to_hvg_idx
    """
    # Original HVG info path
    hvg_info_path = Path(data_path).parent / 'hvg_info.json'
    
    # Cache directory and file
    cache_dir = Path("/workspace/data/vcc_data")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a unique cache key based on the original file path and modification time
    cache_key = hashlib.md5(str(hvg_info_path).encode()).hexdigest()
    cache_file = cache_dir / f"hvg_cache_{cache_key}.json"
    
    # Check if cache exists and is valid
    if cache_file.exists() and hvg_info_path.exists():
        try:
            # Check if original file is newer than cache
            cache_mtime = cache_file.stat().st_mtime
            orig_mtime = hvg_info_path.stat().st_mtime
            
            if cache_mtime >= orig_mtime:
                # Load from cache
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    
                # Verify cache has all required fields
                if all(key in cache_data for key in ['hvg_indices', 'hvg_names', 'gene_name_to_hvg_idx']):
                    print(f"Loading HVG info from cache: {cache_file}")
                    return cache_data
        except Exception as e:
            print(f"Warning: Failed to load cache: {e}")
    
    # Load from original source
    if not hvg_info_path.exists():
        raise FileNotFoundError(
            f"HVG info not found at {hvg_info_path}. "
            "Please run preprocess_hvgs.py --process-vcc first."
        )
    
    print(f"Loading HVG info from source: {hvg_info_path}")
    with open(hvg_info_path, 'r') as f:
        hvg_info = json.load(f)
    
    # Save to cache
    try:
        # Add metadata to cache
        cache_data = hvg_info.copy()
        cache_data['_cache_metadata'] = {
            'source_path': str(hvg_info_path),
            'cached_at': datetime.now().isoformat(),
            'n_hvgs': len(hvg_info['hvg_indices'])
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        print(f"Cached HVG info to: {cache_file}")
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")
    
    return hvg_info


def load_dense_arrays_with_cache(data_path: str, hvg_indices: List[int], max_cells: Optional[int] = None) -> Optional[np.ndarray]:
    """
    Load dense expression arrays with caching for faster subsequent loads.
    
    Args:
        data_path: Path to the h5ad file
        hvg_indices: List of HVG indices for filtering
        max_cells: Maximum number of cells to load
        
    Returns:
        Dense expression array or None if not cached
    """
    # Cache directory and file
    cache_dir = Path("/workspace/data/vcc_data")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create cache key based on data path, HVG indices, and max_cells
    cache_key_data = f"{data_path}_{len(hvg_indices)}_{max_cells}"
    cache_key = hashlib.md5(cache_key_data.encode()).hexdigest()
    dense_cache_file = cache_dir / f"dense_cache_{cache_key}.npz"
    
    # Check if cache exists and is valid
    if dense_cache_file.exists():
        try:
            # Check if original h5ad file is newer than cache
            cache_mtime = dense_cache_file.stat().st_mtime
            orig_mtime = Path(data_path).stat().st_mtime
            
            if cache_mtime >= orig_mtime:
                print(f"Loading dense arrays from cache: {dense_cache_file}")
                with np.load(dense_cache_file) as cached:
                    return cached['expression_data']
        except Exception as e:
            print(f"Warning: Failed to load dense array cache: {e}")
    
    return None


def save_dense_arrays_to_cache(data_path: str, hvg_indices: List[int], expression_data: np.ndarray, max_cells: Optional[int] = None):
    """
    Save dense expression arrays to cache.
    
    Args:
        data_path: Path to the h5ad file
        hvg_indices: List of HVG indices for filtering
        expression_data: Dense expression array to cache
        max_cells: Maximum number of cells loaded
    """
    # Cache directory
    cache_dir = Path("/workspace/data/vcc_data")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create cache key based on data path, HVG indices, and max_cells
    cache_key_data = f"{data_path}_{len(hvg_indices)}_{max_cells}"
    cache_key = hashlib.md5(cache_key_data.encode()).hexdigest()
    dense_cache_file = cache_dir / f"dense_cache_{cache_key}.npz"
    
    try:
        np.savez_compressed(dense_cache_file, expression_data=expression_data)
        print(f"Cached dense arrays to: {dense_cache_file}")
    except Exception as e:
        print(f"Warning: Failed to save dense array cache: {e}")


class VCCDataset(Dataset):
    """
    Simple dataset for VCC data.
    
    Returns:
        - expression: Gene expression vector (n_hvgs,)
        - target_gene: Name of the perturbed gene (or 'non-targeting')
        - is_control: Boolean indicating if this is a control cell
        - cell_id: Original cell barcode
        - batch: Batch identifier
    """
    
    def __init__(
        self,
        data_path: str,
        subset: Optional[str] = None,  # 'perturbed', 'control', or None for all
        max_cells: Optional[int] = None,  # Limit number of cells for debugging
        use_hvgs: bool = True,  # Whether to use only HVG genes
    ):
        """
        Args:
            data_path: Path to adata_Training.h5ad
            subset: Whether to load only perturbed, control, or all cells
            max_cells: Maximum number of cells to load (for debugging)
            use_hvgs: Whether to use only highly variable genes
        """
        # Validate path
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        print(f"Loading VCC data from: {data_path}")
        self.data_path = data_path
        self.subset = subset
        self.max_cells = max_cells
        
        # Load HVG information if requested
        self.use_hvgs = use_hvgs
        self.hvg_indices = None
        self.hvg_names = None
        self.gene_name_to_hvg_idx = None
        self.expression = None
        
        if use_hvgs:
            hvg_info = load_hvg_info_with_cache(data_path)
            
            self.hvg_indices = hvg_info['hvg_indices']
            self.hvg_names = hvg_info['hvg_names']
            self.gene_name_to_hvg_idx = hvg_info['gene_name_to_hvg_idx']
            
            print(f"Using {len(self.hvg_indices)} HVG genes")
            
            # Try to load dense arrays from cache if all HVG keys are present
            if all(key in hvg_info for key in ['hvg_indices', 'hvg_names', 'gene_name_to_hvg_idx']):
                # Create cache key that includes subset info
                cache_key_data = f"{data_path}_{len(self.hvg_indices)}_{max_cells}_{subset}"
                cache_key = hashlib.md5(cache_key_data.encode()).hexdigest()
                
                cache_dir = Path("/workspace/data/vcc_data")
                dense_cache_file = cache_dir / f"dense_cache_{cache_key}.npz"
                
                # Check if cache exists and is valid
                if dense_cache_file.exists():
                    try:
                        # Check if original h5ad file is newer than cache
                        cache_mtime = dense_cache_file.stat().st_mtime
                        orig_mtime = Path(data_path).stat().st_mtime
                        
                        if cache_mtime >= orig_mtime:
                            print(f"Loading dense arrays from cache: {dense_cache_file}")
                            with np.load(dense_cache_file, allow_pickle=True) as cached:
                                self.expression = cached['expression_data']
                                self.cell_ids = cached['cell_ids'].tolist()
                                self.target_genes = cached['target_genes'].tolist()
                                self.batches = cached['batches'].tolist()
                                self.gene_names = self.hvg_names
                                print(f"Using cached data: {len(self.cell_ids)} cells, {len(self.gene_names)} genes")
                    except Exception as e:
                        print(f"Warning: Failed to load dense array cache: {e}")
                        self.expression = None
        
        # Load adata only if we don't have cached dense arrays
        if self.expression is None:
            self.adata = sc.read_h5ad(data_path)
            
            if use_hvgs:
                # Filter genes to only HVGs
                self.adata = self.adata[:, self.hvg_indices]
                self.gene_names = self.hvg_names
            else:
                self.gene_names = self.adata.var.index.tolist()
            
            # Filter by subset if requested
            if subset == 'control':
                mask = self.adata.obs['target_gene'] == 'non-targeting'
                self.adata = self.adata[mask]
                print(f"Loaded {len(self.adata)} control cells")
            elif subset == 'perturbed':
                mask = self.adata.obs['target_gene'] != 'non-targeting'
                self.adata = self.adata[mask]
                print(f"Loaded {len(self.adata)} perturbed cells")
            else:
                print(f"Loaded all {len(self.adata)} cells")
            
            # Limit cells if requested
            if max_cells is not None and len(self.adata) > max_cells:
                self.adata = self.adata[:max_cells]
                print(f"Limited to {max_cells} cells")
            
            # Store metadata
            self.cell_ids = self.adata.obs.index.tolist()
            self.target_genes = self.adata.obs['target_gene'].tolist()
            self.batches = self.adata.obs['batch'].tolist()
            
            # Convert expression to dense array for simplicity
            print("Converting expression data to dense array...")
            if hasattr(self.adata.X, 'toarray'):
                self.expression = self.adata.X.toarray()
            else:
                self.expression = self.adata.X.copy()
            
            # Cache the dense arrays if using HVGs
            if use_hvgs and all(key in hvg_info for key in ['hvg_indices', 'hvg_names', 'gene_name_to_hvg_idx']):
                try:
                    cache_key_data = f"{data_path}_{len(self.hvg_indices)}_{max_cells}_{subset}"
                    cache_key = hashlib.md5(cache_key_data.encode()).hexdigest()
                    
                    cache_dir = Path("/workspace/data/vcc_data")
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    dense_cache_file = cache_dir / f"dense_cache_{cache_key}.npz"
                    
                    np.savez_compressed(
                        dense_cache_file, 
                        expression_data=self.expression,
                        cell_ids=np.array(self.cell_ids),
                        target_genes=np.array(self.target_genes),
                        batches=np.array(self.batches)
                    )
                    print(f"Cached dense arrays to: {dense_cache_file}")
                except Exception as e:
                    print(f"Warning: Failed to save dense array cache: {e}")
            
            print(f"Dataset ready: {len(self)} cells, {len(self.gene_names)} genes")
    
    def __len__(self):
        if hasattr(self, 'adata') and self.adata is not None:
            return len(self.adata)
        else:
            return len(self.cell_ids)
    
    def __getitem__(self, idx):
        """Return a single cell's data."""
        return {
            'expression': torch.FloatTensor(self.expression[idx]),
            'target_gene': self.target_genes[idx],
            'is_control': self.target_genes[idx] == 'non-targeting',
            'cell_id': self.cell_ids[idx],
            'batch': self.batches[idx],
            'idx': idx
        }
    
    def get_control_cells(self):
        """Get indices of all control cells."""
        return [i for i, gene in enumerate(self.target_genes) if gene == 'non-targeting']
    
    def get_perturbed_cells(self, target_gene: str):
        """Get indices of cells with a specific perturbation."""
        return [i for i, gene in enumerate(self.target_genes) if gene == target_gene]
    
    def get_unique_perturbations(self):
        """Get list of unique target genes (excluding controls)."""
        return sorted(set(g for g in self.target_genes if g != 'non-targeting'))
    
    def compute_control_mean(self):
        """Compute mean expression of control cells."""
        control_indices = self.get_control_cells()
        if not control_indices:
            raise ValueError("No control cells found in dataset")
        return self.expression[control_indices].mean(axis=0)
    
    def get_gene_index(self, gene_name: str) -> Optional[int]:
        """Get the index of a gene in the expression matrix."""
        if self.use_hvgs and self.gene_name_to_hvg_idx:
            # If using HVGs, check if gene is in HVG list
            return self.gene_name_to_hvg_idx.get(gene_name, None)
        else:
            # Otherwise, search in full gene list
            try:
                return self.gene_names.index(gene_name)
            except ValueError:
                return None


class VCCPerturbationDataset(Dataset):
    """
    Dataset that returns perturbation-control pairs.
    Useful for training models to learn perturbation effects.
    """
    
    def __init__(self, vcc_dataset: VCCDataset):
        """
        Args:
            vcc_dataset: Base VCCDataset instance
        """
        self.vcc_dataset = vcc_dataset
        
        # Build index of perturbations
        self.perturbations = []
        for gene in self.vcc_dataset.get_unique_perturbations():
            pert_cells = self.vcc_dataset.get_perturbed_cells(gene)
            self.perturbations.extend([(idx, gene) for idx in pert_cells])
        
        # Compute control statistics
        self.control_mean = self.vcc_dataset.compute_control_mean()
        self.control_indices = self.vcc_dataset.get_control_cells()
        
        print(f"Built perturbation dataset: {len(self.perturbations)} perturbed cells")
    
    def __len__(self):
        return len(self.perturbations)
    
    def __getitem__(self, idx):
        """Return a perturbation with its effect."""
        pert_idx, target_gene = self.perturbations[idx]
        
        # Get perturbed cell
        pert_data = self.vcc_dataset[pert_idx]
        
        # Compute perturbation effect (log2 fold change)
        pert_expr = pert_data['expression'].numpy()
        log2fc = np.log2((pert_expr + 1) / (self.control_mean + 1))
        
        # Get target gene index
        target_idx = self.vcc_dataset.get_gene_index(target_gene)
        
        return {
            'expression': pert_data['expression'],
            'target_gene': target_gene,
            'target_gene_idx': target_idx if target_idx is not None else -1,
            'log2fc': torch.FloatTensor(log2fc),
            'control_mean': torch.FloatTensor(self.control_mean),
            'batch': pert_data['batch'],
            'cell_id': pert_data['cell_id']
        }


def find_vcc_data_dir() -> Optional[str]:
    """Try to find VCC data directory in common locations."""
    possible_paths = [
        "/workspace/vcc/data/vcc_data",
        "/data/vcc_data",
        "./data/vcc_data",
        "./vcc_data",
        os.path.expanduser("~/vcc_data"),
        os.environ.get("VCC_DATA_DIR", "")
    ]
    
    for path in possible_paths:
        if path and os.path.exists(path):
            train_file = os.path.join(path, "adata_Training.h5ad")
            if os.path.exists(train_file):
                return path
    
    return None


def load_vcc_data(
    data_dir: Optional[str] = None,
    subset: Optional[str] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    max_cells: Optional[int] = None,
    use_hvgs: bool = True
) -> Tuple[VCCDataset, DataLoader]:
    """
    Convenience function to load VCC data and create dataloader.
    
    Args:
        data_dir: Directory containing VCC data. If None, will search common locations.
        subset: 'control', 'perturbed', or None for all
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle data
        num_workers: Number of workers for DataLoader
        max_cells: Maximum cells to load (for debugging)
        use_hvgs: Whether to use only HVG genes
    
    Returns:
        dataset: VCCDataset instance
        dataloader: DataLoader instance
    """
    # Find data directory if not provided
    if data_dir is None:
        data_dir = find_vcc_data_dir()
        if data_dir is None:
            raise FileNotFoundError(
                "Could not find VCC data directory. Please provide data_dir or "
                "set VCC_DATA_DIR environment variable."
            )
    
    data_path = Path(data_dir) / "adata_Training.h5ad"
    
    dataset = VCCDataset(str(data_path), subset=subset, max_cells=max_cells, use_hvgs=use_hvgs)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return dataset, dataloader


def load_validation_info(data_dir: Optional[str] = None) -> pd.DataFrame:
    """Load validation target genes and cell counts."""
    if data_dir is None:
        data_dir = find_vcc_data_dir()
        if data_dir is None:
            raise FileNotFoundError("Could not find VCC data directory")
    
    val_path = Path(data_dir) / "pert_counts_Validation.csv"
    if not val_path.exists():
        raise FileNotFoundError(f"Validation file not found: {val_path}")
        
    return pd.read_csv(val_path)


if __name__ == "__main__":
    # Example usage
    print("Testing VCC DataLoader...")
    
    try:
        # Load small subset for testing
        dataset, dataloader = load_vcc_data(max_cells=1000)
        
        # Print basic info
        print(f"\nDataset info:")
        print(f"  Total cells: {len(dataset)}")
        print(f"  Total genes: {len(dataset.gene_names)}")
        print(f"  Control cells: {len(dataset.get_control_cells())}")
        print(f"  Unique perturbations: {len(dataset.get_unique_perturbations())}")
        
        # Test iteration
        print("\nTesting iteration:")
        for i, batch in enumerate(dataloader):
            print(f"  Batch {i}: expression shape = {batch['expression'].shape}")
            print(f"    Target genes: {batch['target_gene'][:5]}")
            print(f"    Control cells: {batch['is_control'].sum().item()}/{len(batch['is_control'])}")
            if i >= 2:
                break
        
        # Test perturbation dataset
        print("\nTesting perturbation dataset:")
        pert_dataset = VCCPerturbationDataset(dataset)
        sample = pert_dataset[0]
        print(f"  Sample keys: {list(sample.keys())}")
        print(f"  Target gene: {sample['target_gene']}")
        print(f"  Target gene idx: {sample['target_gene_idx']}")
        print(f"  Log2FC shape: {sample['log2fc'].shape}")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure VCC data is available at one of these locations:")
        print("  - /workspace/vcc/data/vcc_data/")
        print("  - ./data/vcc_data/")
        print("  - Set VCC_DATA_DIR environment variable") 