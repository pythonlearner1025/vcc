"""
ScRNA Dataset that uses cross-dataset HVGs for consistent gene indices.
"""

import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
import scanpy as sc
import json


class ScRNADatasetWithHVGs(Dataset):
    """PyTorch Dataset for single-cell data using cross-dataset HVGs."""
    
    def __init__(
        self, 
        data_dir: str, 
        hvg_genes: List[str],
        transform=None,
        use_cache: bool = True,
        normalize: bool = True
    ):
        """
        Args:
            data_dir: Directory containing preprocessed scRNA batch files
            hvg_genes: List of Ensembl IDs for HVGs
            transform: Optional transformation to apply
            use_cache: Whether to cache gene remapping
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.use_cache = use_cache
        self.normalize = normalize
        self.normalize_check = False
        
        # ------------------------------------------------------------------
        # Filter HVG list to genes that are actually present in the scRNA data
        # ------------------------------------------------------------------
        # Load original gene list from config first (to know what exists)
        config_path = Path(data_dir).parent / "config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.original_genes = config['gene_list']
        original_gene_set = set(self.original_genes)

        # Keep only HVGs that are present in the union-of-genes for the corpus.
        # This mirrors the logic in VCCPairedDataset where HVGs not found are
        # dropped, ensuring both pre-train and fine-tune see the same gene set.
        self.hvg_genes = [g for g in hvg_genes if g in original_gene_set]
        self.n_hvgs = len(self.hvg_genes)
        
        if self.n_hvgs == 0:
            raise ValueError("None of the supplied HVGs exist in the scRNA dataset – check gene IDs")
        
        # Mapping: gene id → position in the pruned HVG list
        hvg_to_idx = {g: i for i, g in enumerate(self.hvg_genes)}
        
        # ------------------------------------------------------------------
        # Locate expression matrix files
        # ------------------------------------------------------------------
        # Find all batch files
        self.batch_files = sorted(self.data_dir.glob("batch_*.h5"))
        if not self.batch_files:
            raise ValueError(f"No batch files found in {data_dir}")
        
        # Create mapping from *original* gene position → pruned HVG index
        self.scrna_to_hvg = {
            orig_idx: hvg_to_idx[gene_id]
            for orig_idx, gene_id in enumerate(self.original_genes)
            if gene_id in hvg_to_idx
        }
        
        # Extract indices for efficient slicing
        self.original_indices = sorted(self.scrna_to_hvg.keys())
        self.hvg_indices = [self.scrna_to_hvg[idx] for idx in self.original_indices]
        
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
    
    def _extract_hvg_columns(self, X: np.ndarray) -> np.ndarray:
        """Extract only HVG columns from the full expression matrix."""
        # Extract relevant columns
        X_subset = X[:, self.original_indices]
        
        # Create final HVG matrix with proper ordering
        X_hvg = np.zeros((X.shape[0], self.n_hvgs), dtype=X.dtype)
        X_hvg[:, self.hvg_indices] = X_subset
        
        return X_hvg
    
    def _load_batch(self, batch_idx: int) -> np.ndarray:
        """Load and extract HVG columns from a batch of data."""
        # Check cache
        if self._cache is not None and batch_idx in self._cache:
            return self._cache[batch_idx]
        
        # Load batch
        batch_file = self.batch_files[batch_idx]
        with h5py.File(batch_file, 'r') as f:
            X = f['X'][:]

            # Apply normalization if requested
        if self.normalize:
            # Convert to float32 for normalization
            X = X.astype(np.float32)
            
            # CP10K normalization: scale each cell to 10,000 total counts
            row_sums = X.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            X = (X / row_sums) * 10000
            
            # Log1p transformation
            X = np.log1p(X)
        elif not self.normalize_check:
            # Verify if data is log1p (CP10K-normalised) when `normalize=False`.
            # After CP10K + log1p, each gene value should be ≤ log1p(10000) ≈ 9.2
            # and row sums should be reasonable (much larger than 9.2)
            max_possible_gene_value = np.log1p(10000)  # ≈ 9.2
            X_max = X.max()
            row_sums = X.sum(axis=1)
            
            # Check if individual gene values exceed theoretical maximum
            if X_max > max_possible_gene_value + 0.1:  # Small tolerance for numerical precision
                print(f"[ScRNADatasetWithHVGs] Detected NONE-log1p(CP10K-normalised) data. Max value {X_max:.2f} > {max_possible_gene_value:.2f}")
                raise Exception
            # Check if row sums are too small (indicating raw counts, not log1p)
            elif row_sums.mean() < 50:  # log1p normalized data should have much higher row sums
                print(f"[ScRNADatasetWithHVGs] Detected NONE-log1p(CP10K-normalised) data. Mean row sum {row_sums.mean():.1f} too small.")
                raise Exception
            else:
                print(f"[ScRNADatasetWithHVGs] Verified log1p(CP10K) normalization:")
                print(f"  Max gene value: {X_max:.2f} (≤ {max_possible_gene_value:.2f})")
                print(f"  Mean row sum: {row_sums.mean():.1f}")

            self.normalize_check = True

        # Extract HVG columns
        X_hvg = self._extract_hvg_columns(X)
        
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
        """Get HVG gene names (Ensembl IDs) in order."""
        return self.hvg_genes
    
    def clear_cache(self):
        """Clear the batch cache to free memory."""
        if self._cache is not None:
            self._cache.clear()


def create_scrna_hvg_dataloader(
    data_dir: str,
    hvg_genes: List[str],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    use_cache: bool = True,
    normalize: bool = True
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for scRNA data using cross-dataset HVGs.
    
    Args:
        data_dir: Directory containing preprocessed scRNA data
        hvg_genes: List of Ensembl IDs for HVGs
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle data
        num_workers: Number of workers for DataLoader
        use_cache: Whether to cache remapped batches
        
    Returns:
        Tuple of (dataset, dataloader)
    """
    dataset = ScRNADatasetWithHVGs(
        data_dir=data_dir,
        hvg_genes=hvg_genes,
        use_cache=use_cache,
        normalize=normalize
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