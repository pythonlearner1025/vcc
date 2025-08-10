"""
Efficient VAE DataLoader with pre-chunked storage support.

Handles loading pre-processed chunked datasets for VAE training with:
- Direct loading from pre-chunked HDF5 files
- Memory monitoring and optimization  
- Fast indexed retrieval by cell, SRX, or chunk
- Support for sparse matrices and mixed data types
- Lazy loading with intelligent caching

Authors: AG  
Created: July 2025
Updated: August 2025 - Added chunked storage support
"""

import json
import logging
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import scanpy as sc
import anndata as ad
from tqdm import tqdm
import warnings
import psutil
import gc
import time

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Utility class for monitoring memory usage."""
    
    @staticmethod
    def get_memory_info() -> Dict[str, float]:
        """Get current memory usage information."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        info = {
            'ram_gb': memory_info.rss / (1024**3),
            'virtual_gb': memory_info.vms / (1024**3)
        }
        
        if torch.cuda.is_available():
            info.update({
                'gpu_allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                'gpu_reserved_gb': torch.cuda.memory_reserved() / (1024**3)
            })
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            info['device'] = 'mps'
        elif torch.cuda.is_available():
            info['device'] = 'cuda'
        else:
            info['device'] = 'cpu'
            
        return info
    
    @staticmethod
    def log_memory_usage(stage: str):
        """Log current memory usage."""
        info = MemoryMonitor.get_memory_info()
        logger.info(f"[{stage}] Memory Usage:")
        logger.info(f"  RAM: {info['ram_gb']:.2f}GB, Virtual: {info['virtual_gb']:.2f}GB")
        if 'gpu_allocated_gb' in info:
            logger.info(f"  GPU: {info['gpu_allocated_gb']:.2f}GB allocated, {info['gpu_reserved_gb']:.2f}GB reserved")
        logger.info(f"  Device: {info['device']}")


# Global variable to store shared data for multiprocessing
_SHARED_DATA = None


@dataclass
class DatasetConfig:
    """Configuration for data loading with memory optimization."""
    
    # === Data Paths ===
    train_data_path: str = None
    val_data_path: str = None
    test_data_path: str = None
    
    # === Training Parameters ===
    batch_size: int = 256
    num_workers: int = 0  # Keep as configured by user
    pin_memory: bool = False  # Keep as configured by user
    shuffle_train: bool = True
    
    # === Metadata Validation ===
    needed_obs_columns: List[str] = None  # Columns that must be present in obs
    
    # === Data Splits ===
    train_split: float = 0.8
    val_split: float = 0.2
    test_split: float = 0.0  # Disable test split by default to avoid tiny remainders
    min_test_cells: int = 1000  # Minimum cells required for test set
    
    # === Memory Management ===
    preload_to_shared_memory: bool = True  # Preload data to shared memory for multiprocessing
    max_memory_gb: float = 16.0  # Maximum memory usage before switching to disk-based loading
    chunk_size: int = 10000  # Chunk size for processing large datasets
    
    transform: Optional[torch.nn.Module] = None  # Optional transform to apply to data

class OptimizedSingleCellDataset(Dataset):
    """
    Memory-optimized dataset class for single-cell RNA-seq data with multiprocessing support.
    
    Features:
    - Shared memory for multiprocessing to avoid data duplication
    - Efficient sparse matrix handling
    - Overflow prevention with value clamping
    - Memory monitoring
    """
    
    def __init__(self, 
                 data_tensor: torch.Tensor,
                 metadata_df: pd.DataFrame,
                 config: DatasetConfig,
                 indices: Optional[np.ndarray] = None,
                 stage: str = "train"
                ):
        """
        Initialize dataset with pre-loaded shared data.
        
        Args:
            data_tensor: Pre-loaded gene expression data [n_cells, n_genes] 
            metadata_df: Pre-loaded metadata DataFrame
            config: Data configuration
            indices: Subset indices for this split
            stage: Dataset stage (train/val/test)
        """
        self.config = config
        self.stage = stage
        self.data_tensor = data_tensor
        self.metadata_df = metadata_df
        self.indices = indices if indices is not None else np.arange(len(data_tensor))
        self.n_cells = len(self.indices)
        self.n_genes = data_tensor.shape[1]
        
        # Pre-compute statistics to avoid overflow during training
        self._precompute_statistics()
        
        logger.info(f"Initialized {stage} dataset: {self.n_cells} cells, {self.n_genes} genes")

    def _precompute_statistics(self):
        """Pre-compute statistics to avoid overflow during training."""
        try:
            # Get subset of data for this split
            subset_data = self.data_tensor[self.indices]
            
            # Compute UMI counts with overflow protection
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                umi_counts = subset_data.sum(dim=1)
            
            # Check for problematic values
            if torch.any(torch.isinf(umi_counts)) or torch.any(torch.isnan(umi_counts)):
                logger.warning(f"Found inf/nan values in UMI counts for {self.stage} dataset")
                # umi_counts = torch.clamp(umi_counts, 0, 1e6)  # More conservative clamp
            
            self.umi_counts = umi_counts
            
            # Compute data statistics for monitoring
            self.data_mean = subset_data.mean().item()
            self.data_std = subset_data.std().item()
            self.data_max = subset_data.max().item()
            
            logger.info(f"Pre-computed statistics for {self.stage}:")
            logger.info(f"  UMI counts - mean: {umi_counts.mean().item():.1f}, "
                       f"std: {umi_counts.std().item():.1f}, "
                       f"range: [{umi_counts.min().item():.1f}, {umi_counts.max().item():.1f}]")
            logger.info(f"  Expression - mean: {self.data_mean:.3f}, "
                       f"std: {self.data_std:.3f}, max: {self.data_max:.3f}")
            
        except Exception as e:
            logger.warning(f"Failed to pre-compute statistics: {e}")
            self.umi_counts = torch.zeros(self.n_cells)
            self.data_mean = 0.0
            self.data_std = 1.0
            self.data_max = 1.0

    def __len__(self) -> int:
        """Return number of cells in dataset."""
        return self.n_cells
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Get a single sample efficiently with overflow prevention.
        
        Returns:
            Tuple of (gene_expression, metadata)
        """
        if idx >= self.n_cells:
            raise IndexError(f"Index {idx} out of range for dataset with {self.n_cells} cells")
        
        # Get the actual data index
        actual_idx = self.indices[idx]
        
        # Get gene expression (already in tensor format with shared memory)
        gene_expression = self.data_tensor[actual_idx].clone()
        
        # Apply overflow prevention
        gene_expression = torch.clamp(gene_expression, 0, 1e6)  # Prevent overflow
        
        # Check for NaN/Inf values
        if torch.any(torch.isnan(gene_expression)) or torch.any(torch.isinf(gene_expression)):
            logger.warning(f"Found NaN/Inf in sample {idx}, replacing with zeros")
            gene_expression = torch.where(torch.isnan(gene_expression) | torch.isinf(gene_expression), 
                                        torch.zeros_like(gene_expression), gene_expression)
        
        # Get metadata efficiently
        try:
            metadata_row = self.metadata_df.iloc[actual_idx]
            metadata = {
                'UMI_count': float(self.umi_counts[idx].item()),
                'batch_var': str(metadata_row.get('batch_var', 'unknown')),
                'SRX_accession': str(metadata_row.get('SRX_accession', 'unknown')),
                'target_gene': str(metadata_row.get('target_gene', 'control')),
                'dataset_stage': self.stage,
                'data_stats': {
                    'mean': self.data_mean,
                    'std': self.data_std,
                    'max': self.data_max
                }
            }
        except Exception as e:
            logger.warning(f"Failed to extract metadata for sample {idx}: {e}")
            metadata = {
                'UMI_count': float(self.umi_counts[idx].item()),
                'batch_var': 'unknown',
                'SRX_accession': 'unknown', 
                'target_gene': 'control',
                'dataset_stage': self.stage,
                'data_stats': {'mean': self.data_mean, 'std': self.data_std, 'max': self.data_max}
            }
        
        return gene_expression, metadata


def setup_shared_memory(data_tensor: torch.Tensor, metadata_df: pd.DataFrame):
    """Setup shared memory for multiprocessing."""
    global _SHARED_DATA
    _SHARED_DATA = {
        'data_tensor': data_tensor.share_memory_(),
        'metadata_df': metadata_df
    }
    logger.info("Set up shared memory for multiprocessing")


def load_and_preprocess_data(data_path: str, 
                            config: DatasetConfig,
                            needed_obs_columns: Optional[List[str]] = None) -> Tuple[torch.Tensor, pd.DataFrame]:
    """
    Load and preprocess single-cell data with memory optimization and overflow prevention.
    
    Args:
        data_path: Path to data file
        config: Data configuration  
        needed_obs_columns: Required observation columns
    
    Returns:
        Tuple of (data_tensor, metadata_df)
    """
    MemoryMonitor.log_memory_usage("Before data loading")
    
    data_path = Path(data_path)
    
    # Load data
    if data_path.suffix == '.h5ad':
        adata = sc.read_h5ad(data_path)
    elif data_path.suffix == '.csv':
        df = pd.read_csv(data_path, index_col=0)
        adata = anndata.AnnData(df.values)
        adata.obs_names = df.index
        adata.var_names = df.columns
    elif data_path.suffix == '.h5':
        adata = sc.read_10x_h5(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")
    
    MemoryMonitor.log_memory_usage("After data loading")
    
    # Validate required columns
    if needed_obs_columns:
        missing_cols = [col for col in needed_obs_columns if col not in adata.obs.columns]
        if missing_cols:
            logger.warning(f"Missing observation columns: {missing_cols}")
            # Add missing columns with default values
            for col in missing_cols:
                if col == 'UMI_count':
                    # Calculate UMI count if missing
                    if hasattr(adata.X, 'toarray'):
                        umi_counts = np.array(adata.X.sum(axis=1)).flatten()
                    else:
                        umi_counts = adata.X.sum(axis=1)
                    adata.obs[col] = umi_counts
                else:
                    adata.obs[col] = 'unknown'
    
    # Convert to tensor format efficiently with chunking for large datasets
    logger.info("Converting expression data to tensor format...")
    
    n_cells = adata.n_obs
    estimated_memory = (n_cells * adata.n_vars * 4) / (1024**3)  # 4 bytes per float32
    
    if estimated_memory > config.max_memory_gb:
        logger.info(f"Large dataset detected ({estimated_memory:.1f}GB), using chunked processing")
        
    if hasattr(adata.X, 'toarray'):
        # Sparse matrix - convert in chunks to manage memory
        chunk_size = min(config.chunk_size, n_cells)
        
        data_list = []
        for start_idx in range(0, n_cells, chunk_size):
            end_idx = min(start_idx + chunk_size, n_cells)
            chunk = adata.X[start_idx:end_idx].toarray()
            
            # Prevent overflow and handle problematic values
            chunk = np.clip(chunk, 0, 1e6)  # Clip to prevent overflow
            chunk = np.nan_to_num(chunk, nan=0.0, posinf=1e6, neginf=0.0)  # Handle NaN/Inf
            
                
            if np.isnan(chunk).any():
                logger.warning(f"Still found NaN after cleaning in chunk {start_idx//chunk_size}")
                chunk = np.where(np.isnan(chunk), 0.0, chunk)
            if np.isinf(chunk).any():
                logger.warning(f"Still found Inf after cleaning in chunk {start_idx//chunk_size}")
                chunk = np.where(np.isinf(chunk), 0.0, chunk)
            
            chunk = chunk.astype(np.float32)  # Use float32 to save memory
            
            data_list.append(torch.from_numpy(chunk))
            
            if start_idx % (chunk_size * 5) == 0:
                logger.info(f"Processed {end_idx}/{n_cells} cells")
                gc.collect()  # Force garbage collection
        
        data_tensor = torch.cat(data_list, dim=0)
        del data_list  # Free memory immediately
        
    else:
        # Dense matrix
        data_array = np.clip(adata.X, 0, 1e6)  # Prevent overflow
        data_array = np.nan_to_num(data_array, nan=0.0, posinf=1e6, neginf=0.0)  # Handle NaN/Inf
        data_array = data_array.astype(np.float32)
        data_tensor = torch.from_numpy(data_array)
    
    # Extract metadata
    metadata_df = adata.obs.copy()
    
    # Clean up original data
    del adata
    gc.collect()
    
    MemoryMonitor.log_memory_usage("After tensor conversion")
    
    logger.info(f"Loaded data: {data_tensor.shape[0]} cells, {data_tensor.shape[1]} genes")
    logger.info(f"Data stats: mean={data_tensor.mean().item():.3f}, "
               f"std={data_tensor.std().item():.3f}, "
               f"max={data_tensor.max().item():.3f}")
    
    return data_tensor, metadata_df


def create_vae_dataloaders(config: DatasetConfig) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create memory-optimized dataloaders for VAE training with proper split handling.
    
    Args:
        config: Data configuration
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    MemoryMonitor.log_memory_usage("Before dataloader creation")
    
    if not config.train_data_path:
        raise ValueError("train_data_path must be specified")
    
    # Load and preprocess data once
    data_tensor, metadata_df = load_and_preprocess_data(
        config.train_data_path,
        config,
        config.needed_obs_columns
    )
    
    # Set up shared memory for multiprocessing if needed
    if config.num_workers > 0 and config.preload_to_shared_memory:
        setup_shared_memory(data_tensor, metadata_df)
    
    # Create proper splits that add up to 1.0
    n_cells = len(data_tensor)
    
    # Ensure splits add up correctly and handle test set creation
    total_split = config.train_split + config.val_split + config.test_split
    
    if total_split > 1.0:
        logger.warning(f"Splits sum to {total_split:.3f} > 1.0, normalizing...")
        config.train_split = config.train_split / total_split
        config.val_split = config.val_split / total_split
        config.test_split = config.test_split / total_split
    elif total_split < 1.0:
        # Redistribute remainder to validation to avoid tiny test sets
        remainder = 1.0 - total_split
        config.val_split += remainder
        logger.info(f"Added {remainder:.3f} remainder to validation split")
    
    # Calculate actual split sizes
    train_size = int(config.train_split * n_cells)
    val_size = int(config.val_split * n_cells)
    test_size = n_cells - train_size - val_size  # Use remaining cells for test
    
    # Check if test set should be created
    create_test_set = (config.test_split > 0 and test_size >= config.min_test_cells)
    
    if config.test_split > 0 and not create_test_set:
        logger.info(f"Test set too small ({test_size} < {config.min_test_cells}), "
                   f"redistributing {test_size} cells to validation")
        val_size += test_size
        test_size = 0
    
    # Create indices for splits
    indices = torch.randperm(n_cells).numpy()
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:train_size + val_size + test_size] if create_test_set else None
    
    logger.info(f"Data splits created:")
    logger.info(f"  Train: {len(train_indices)} cells ({len(train_indices)/n_cells*100:.1f}%)")
    logger.info(f"  Val: {len(val_indices)} cells ({len(val_indices)/n_cells*100:.1f}%)")
    if create_test_set:
        logger.info(f"  Test: {len(test_indices)} cells ({len(test_indices)/n_cells*100:.1f}%)")
    else:
        logger.info(f"  Test: Not created (test_split={config.test_split}, min_cells={config.min_test_cells})")
    
    # Create datasets with shared data
    train_dataset = OptimizedSingleCellDataset(
        data_tensor, metadata_df, config, train_indices, "train"
    )
    
    val_dataset = OptimizedSingleCellDataset(
        data_tensor, metadata_df, config, val_indices, "val"
    )
    
    test_dataset = None
    if create_test_set:
        test_dataset = OptimizedSingleCellDataset(
            data_tensor, metadata_df, config, test_indices, "test"
        )
    
    # Create dataloaders with optimized settings for multiprocessing
    base_dataloader_kwargs = {
        'batch_size': config.batch_size,
        'num_workers': config.num_workers,
        'pin_memory': config.pin_memory,
        'collate_fn': optimized_collate_fn,
    }
    
    # Add multiprocessing-specific options only if using workers
    if config.num_workers > 0:
        base_dataloader_kwargs.update({
            'persistent_workers': True,
            'prefetch_factor': 2,
        })
    
    train_loader = DataLoader(
        train_dataset,
        shuffle=config.shuffle_train,
        drop_last=True,  # Drop last for training to maintain consistent batch sizes
        **base_dataloader_kwargs
    )
    
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,  # Don't drop last batch for validation
        **base_dataloader_kwargs
    )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            drop_last=False,  # Don't drop last batch for test
            **base_dataloader_kwargs
        )
    
    # Log final dataloader information
    logger.info(f"Created optimized dataloaders:")
    logger.info(f"  Train: {len(train_loader)} batches ({len(train_dataset)} cells)")
    logger.info(f"  Val: {len(val_loader)} batches ({len(val_dataset)} cells)")
    if test_loader:
        logger.info(f"  Test: {len(test_loader)} batches ({len(test_dataset)} cells)")
    else:
        logger.info(f"  Test: Not created")
    
    MemoryMonitor.log_memory_usage("After dataloader creation")
    
    return train_loader, val_loader, test_loader


def optimized_collate_fn(batch) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """
    Optimized collate function with overflow prevention and error handling.
    
    Args:
        batch: List of tuples (gene_expression, metadata) from dataset
    
    Returns:
        Batched tensors and metadata list
    """
    try:
        # Stack gene expressions
        gene_expressions = torch.stack([sample[0] for sample in batch])
        
        # Apply final overflow prevention at batch level
        # gene_expressions = torch.clamp(gene_expressions, 0, 1e6)
        
        # Check for any remaining problematic values
        if torch.any(torch.isnan(gene_expressions)) or torch.any(torch.isinf(gene_expressions)):
            logger.warning("Found NaN/Inf in batch, cleaning...")
            gene_expressions = torch.where(
                torch.isnan(gene_expressions) | torch.isinf(gene_expressions),
                torch.zeros_like(gene_expressions),
                gene_expressions
            )
        
        # Extract metadata
        metadata_list = [sample[1] for sample in batch]
        
        return gene_expressions, metadata_list
        
    except Exception as e:
        logger.error(f"Error in collate function: {e}")
        # Return empty batch rather than crashing
        batch_size = len(batch)
        n_genes = len(batch[0][0]) if batch else 0
        return torch.zeros(batch_size, n_genes), [{}] * batch_size

if __name__ == "__main__":
    # Test the optimized dataloaders
    print("Testing Optimized VAE DataLoaders...")
    
    MemoryMonitor.log_memory_usage("Test start")
    
    # Create test configuration
    config = DatasetConfig(
        train_data_path="/Users/agreic/Documents/GitHub/virtual-cell/data/processed_data_dev/processed_data.h5ad",
        batch_size=64,
        num_workers=0,  # Start with 0 workers for testing
        test_split=0.05,  # Small test split
        min_test_cells=50  # Lower threshold for testing
    )
    
    try:
        print("Creating optimized dataloaders...")
        train_loader, val_loader, test_loader = create_vae_dataloaders(config)
        
        MemoryMonitor.log_memory_usage("After dataloader creation")
        
        # Test train loader
        print("Testing train loader...")
        for i, (gene_expressions, metadata_list) in enumerate(train_loader):
            print(f"Batch {i}: Gene expressions shape: {gene_expressions.shape}")
            print(f"  Data range: [{gene_expressions.min().item():.2f}, {gene_expressions.max().item():.2f}]")
            print(f"  Sample metadata keys: {list(metadata_list[0].keys())}")
            print(f"  UMI count example: {metadata_list[0]['UMI_count']:.1f}")
            
            # Check for overflow issues
            if torch.any(torch.isnan(gene_expressions)) or torch.any(torch.isinf(gene_expressions)):
                print("  WARNING: Found NaN/Inf values!")
            else:
                print("  Data clean: No NaN/Inf values found")
            
            if i >= 2:  # Test first 3 batches
                break
        
        # Test validation loader
        print("\nTesting validation loader...")
        val_batch = next(iter(val_loader))
        gene_expressions, metadata_list = val_batch
        print(f"Val batch shape: {gene_expressions.shape}")
        
        # Test test loader if available
        if test_loader:
            print("\nTesting test loader...")
            test_batch = next(iter(test_loader))
            gene_expressions, metadata_list = test_batch
            print(f"Test batch shape: {gene_expressions.shape}")
        else:
            print("\nTest loader not created (as configured)")
        
        MemoryMonitor.log_memory_usage("Test completed")
        print("\nOptimized dataloader test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        MemoryMonitor.log_memory_usage("Test failed")
