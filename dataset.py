"""
PyTorch Dataset for single-cell RNA-seq data.
"""

import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class ScRNADataset(Dataset):
    """PyTorch Dataset for single-cell gene expression data."""
    
    def __init__(self, data_dir: str, transform=None, max_genes: Optional[int] = None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.max_genes = max_genes
        
        # Find all batch files
        self.batch_files = sorted(self.data_dir.glob("batch_*.h5"))
        if not self.batch_files:
            raise ValueError(f"No batch files found in {data_dir}")
        
        # Load gene list from first batch
        with h5py.File(self.batch_files[0], 'r') as f:
            self.genes = [g.decode() for g in f['genes'][:]]
            if max_genes:
                self.genes = self.genes[:max_genes]
        
        # Count total cells
        self.batch_starts = [0]
        total_cells = 0
        for batch_file in self.batch_files:
            with h5py.File(batch_file, 'r') as f:
                n_cells = f['X'].shape[0]
                total_cells += n_cells
                self.batch_starts.append(total_cells)
        
        self.n_cells = total_cells
        self.n_genes = len(self.genes)
    
    def __len__(self):
        return self.n_cells
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        # Find which batch this index belongs to
        batch_idx = np.searchsorted(self.batch_starts[1:], idx, side='right')
        local_idx = idx - self.batch_starts[batch_idx]
        
        # Load data
        with h5py.File(self.batch_files[batch_idx], 'r') as f:
            # Gene expression vector
            x = f['X'][local_idx]
            if self.max_genes:
                x = x[:self.max_genes]
            
            # Metadata
            meta = {
                'gene_count': f['obs/gene_count'][local_idx],
                'umi_count': f['obs/umi_count'][local_idx],
                'srx_accession': f['obs/SRX_accession'][local_idx].decode()
            }
        
        # Convert to tensor
        x = torch.from_numpy(x).float()
        
        # Apply transform if any
        if self.transform:
            x = self.transform(x)
        
        return x, meta


def create_dataloader(data_dir: str, batch_size: int = 32, 
                     shuffle: bool = True, num_workers: int = 4) -> DataLoader:
    """Create a DataLoader for training."""
    dataset = ScRNADataset(data_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "data/scRNA/processed"
    
    print(f"Loading dataset from: {data_dir}")
    
    try:
        dataset = ScRNADataset(data_dir)
        print(f"Dataset loaded successfully!")
        print(f"  Total cells: {len(dataset):,}")
        print(f"  Number of genes: {dataset.n_genes:,}")
        print(f"  Number of batches: {len(dataset.batch_files)}")
        
        # Test loading a single sample
        x, meta = dataset[0]
        print(f"\nFirst sample:")
        print(f"  Shape: {x.shape}")
        print(f"  Gene count: {meta['gene_count']}")
        print(f"  UMI count: {meta['umi_count']}")
        print(f"  Source: {meta['srx_accession']}")
        
        # Test dataloader
        dataloader = create_dataloader(data_dir, batch_size=4)
        for batch_idx, (x_batch, meta_batch) in enumerate(dataloader):
            print(f"\nBatch test:")
            print(f"  Batch shape: {x_batch.shape}")
            print(f"  Gene counts: {meta_batch['gene_count']}")
            break
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) 