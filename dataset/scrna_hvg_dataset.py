"""
ScRNA Dataset that uses cross-dataset HVGs for consistent gene indices.
"""

import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
import json


class ScRNADatasetWithHVGs(Dataset):
    """PyTorch Dataset for single-cell data using cross-dataset HVGs."""
    
    def __init__(
        self, 
        data_dir: str, 
        hvg_info_path: str,
        transform=None,
        use_cache: bool = True
    ):
        """
        Args:
            data_dir: Directory containing preprocessed scRNA batch files
            hvg_info_path: Path to cross_dataset_hvg_info.json
            transform: Optional transformation to apply
            use_cache: Whether to cache gene remapping
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.use_cache = use_cache
        
        # Load HVG info
        with open(hvg_info_path, 'r') as f:
            hvg_info = json.load(f)
        
        self.hvg_names = hvg_info['hvg_names']
        self.n_hvgs = len(self.hvg_names)
        self.scrna_to_hvg = {int(k): v for k, v in hvg_info['scrna_to_hvg'].items()}
        
        # Find all batch files
        self.batch_files = sorted(self.data_dir.glob("batch_*.h5"))
        if not self.batch_files:
            raise ValueError(f"No batch files found in {data_dir}")
        
        # Load original gene list from config
        config_path = self.data_dir.parent / "config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.original_genes = config['gene_list']
        
        # Count total cells and build index
        self.batch_starts = [0]
        total_cells = 0
        self.batch_info = []
        
        for batch_file in self.batch_files:
            with h5py.File(batch_file, 'r') as f:
                n_cells = f['X'].shape[0]
                total_cells += n_cells
                self.batch_starts.append(total_cells)
                self.batch_info.append({
                    'file': batch_file,
                    'n_cells': n_cells,
                    'start_idx': self.batch_starts[-2]
                })
        
        self.n_cells = total_cells
        
        # Cache for remapped batches
        self._cache = {} if use_cache else None
        
        print(f"Initialized ScRNA dataset with cross-dataset HVGs:")
        print(f"  Total cells: {self.n_cells:,}")
        print(f"  Original genes: {len(self.original_genes):,}")
        print(f"  HVG genes: {self.n_hvgs}")
        print(f"  Genes mapped to HVGs: {len(self.scrna_to_hvg)}")
    
    def _remap_genes(self, X: np.ndarray) -> np.ndarray:
        """Remap gene expression from original indices to HVG indices."""
        # Create remapped array with zeros
        X_hvg = np.zeros((X.shape[0], self.n_hvgs), dtype=X.dtype)
        
        # Copy values for genes that are in HVG set
        for orig_idx, hvg_idx in self.scrna_to_hvg.items():
            if orig_idx < X.shape[1]:  # Check bounds
                X_hvg[:, hvg_idx] = X[:, orig_idx]
        
        return X_hvg
    
    def _load_batch(self, batch_idx: int) -> np.ndarray:
        """Load and remap a batch of data."""
        # Check cache
        if self._cache is not None and batch_idx in self._cache:
            return self._cache[batch_idx]
        
        # Load batch
        batch_file = self.batch_files[batch_idx]
        with h5py.File(batch_file, 'r') as f:
            X = f['X'][:]
        
        # Remap to HVG indices
        X_hvg = self._remap_genes(X)
        
        # Cache if enabled
        if self._cache is not None:
            self._cache[batch_idx] = X_hvg
        
        return X_hvg
    
    def __len__(self):
        return self.n_cells
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a single cell's expression using HVG indices."""
        # Find which batch this index belongs to
        batch_idx = np.searchsorted(self.batch_starts[1:], idx, side='right')
        local_idx = idx - self.batch_starts[batch_idx]
        
        # Load remapped batch data
        X_hvg = self._load_batch(batch_idx)
        
        # Get specific cell
        x = X_hvg[local_idx]
        
        # Convert to tensor
        x = torch.from_numpy(x).float()
        
        # Apply transform if any
        if self.transform:
            x = self.transform(x)
        
        return x
    
    def get_gene_names(self) -> List[str]:
        """Get HVG gene names in order."""
        return self.hvg_names
    
    def clear_cache(self):
        """Clear the batch cache to free memory."""
        if self._cache is not None:
            self._cache.clear()


def create_scrna_hvg_dataloader(
    data_dir: str = "data/scRNA_1e5/processed",
    hvg_info_path: str = "data/vcc_data/cross_dataset_hvg_info.json",
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    use_cache: bool = True
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for scRNA data using cross-dataset HVGs.
    
    Args:
        data_dir: Directory containing preprocessed scRNA data
        hvg_info_path: Path to cross-dataset HVG info file
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle data
        num_workers: Number of workers for DataLoader
        use_cache: Whether to cache remapped batches
        
    Returns:
        DataLoader instance
    """
    dataset = ScRNADatasetWithHVGs(
        data_dir=data_dir,
        hvg_info_path=hvg_info_path,
        use_cache=use_cache
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0  # Keep workers alive
    )
    
    return dataset, dataloader 