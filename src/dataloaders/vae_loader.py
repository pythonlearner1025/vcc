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
            info['gpu_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
            info['gpu_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
        
        return info
    
    @staticmethod
    def log_memory_usage(prefix: str = ""):
        """Log current memory usage."""
        info = MemoryMonitor.get_memory_info()
        logger.info(f"{prefix}Memory: RAM={info['ram_gb']:.2f}GB")
        if 'gpu_allocated_gb' in info:
            logger.info(f"{prefix}GPU: {info['gpu_allocated_gb']:.2f}GB allocated, {info['gpu_reserved_gb']:.2f}GB reserved")


@dataclass 
class DatasetConfig:
    """Configuration for dataset loading and processing."""
    data_path: str
    batch_size: int = 256
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True
    drop_last: bool = False
    
    # Chunked loading settings
    max_chunks_in_memory: int = 3
    chunk_cache_size: int = 5
    preload_chunks: bool = True
    
    # Memory settings  
    max_memory_gb: float = 16.0
    monitor_memory: bool = True
    
    # Data filtering
    min_cells_per_chunk: int = 1000
    include_srx_accessions: Optional[List[str]] = None
    exclude_srx_accessions: Optional[List[str]] = None


class ChunkedDataset(Dataset):
    """
    Efficient dataset for loading pre-chunked scRNA-seq data.
    
    Loads data from pre-processed chunks with intelligent caching
    and memory management for efficient training.
    """
    
    def __init__(self, data_path: str, config: DatasetConfig):
        """Initialize chunked dataset."""
        self.data_path = Path(data_path)
        self.config = config
        
        # Load processing index
        index_path = self.data_path / "chunk_index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Chunk index not found: {index_path}")
        
        self.processing_index = self._load_processing_index(index_path)
        
        # Filter chunks based on config
        self.valid_chunks = self._filter_chunks()
        
        # Create cell index mapping
        self.cell_to_chunk_map = self._create_cell_index()
        
        # Initialize chunk cache
        self.chunk_cache = {}
        self.cache_access_times = {}
        
        # Total cells across valid chunks
        self.total_cells = sum(chunk.n_cells for chunk in self.valid_chunks)
        
        logger.info(f"ChunkedDataset initialized:")
        logger.info(f"  Data path: {self.data_path}")
        logger.info(f"  Valid chunks: {len(self.valid_chunks)}")
        logger.info(f"  Total cells: {self.total_cells:,}")
        logger.info(f"  Genes: {len(self.processing_index.gene_list)}")
    
    def _load_processing_index(self, index_path: Path):
        """Load processing index from JSON."""
        with open(index_path, 'r') as f:
            index_dict = json.load(f)
        
        # Convert to ProcessingIndex object (simplified)
        class ProcessingIndex:
            def __init__(self, data):
                self.chunks = [self._dict_to_chunk_metadata(chunk) for chunk in data['chunks']]
                self.gene_list = data['gene_list']
                self.total_cells = data['total_cells']
                self.total_chunks = data['total_chunks']
                self.processing_config = data['processing_config']
                self.processing_timestamp = data['processing_timestamp']
            
            def _dict_to_chunk_metadata(self, chunk_dict):
                class ChunkMetadata:
                    def __init__(self, **kwargs):
                        for k, v in kwargs.items():
                            setattr(self, k, v)
                return ChunkMetadata(**chunk_dict)
        
        return ProcessingIndex(index_dict)
    
    def _filter_chunks(self) -> List:
        """Filter chunks based on configuration."""
        valid_chunks = []
        
        for chunk in self.processing_index.chunks:
            # Filter by minimum cells
            if chunk.n_cells < self.config.min_cells_per_chunk:
                continue
            
            # Filter by SRX accessions if specified
            if self.config.include_srx_accessions:
                if not any(srx in chunk.srx_accessions for srx in self.config.include_srx_accessions):
                    continue
            
            if self.config.exclude_srx_accessions:
                if any(srx in chunk.srx_accessions for srx in self.config.exclude_srx_accessions):
                    continue
            
            valid_chunks.append(chunk)
        
        logger.info(f"Filtered chunks: {len(self.processing_index.chunks)} -> {len(valid_chunks)}")
        return valid_chunks
    
    def _create_cell_index(self) -> Dict[int, Tuple[int, int]]:
        """Create mapping from global cell index to (chunk_id, local_cell_index)."""
        cell_to_chunk = {}
        global_cell_idx = 0
        
        for chunk in self.valid_chunks:
            for local_idx in range(chunk.n_cells):
                cell_to_chunk[global_cell_idx] = (chunk.chunk_id, local_idx)
                global_cell_idx += 1
        
        return cell_to_chunk
    
    def _load_chunk(self, chunk_id: int) -> Dict[str, Any]:
        """Load a chunk from disk with caching."""
        # Check cache first
        if chunk_id in self.chunk_cache:
            self.cache_access_times[chunk_id] = time.time()
            return self.chunk_cache[chunk_id]
        
        # Find chunk metadata
        chunk_metadata = None
        for chunk in self.valid_chunks:
            if chunk.chunk_id == chunk_id:
                chunk_metadata = chunk
                break
        
        if chunk_metadata is None:
            raise ValueError(f"Chunk {chunk_id} not found")
        
        # Load chunk from disk
        chunk_path = self.data_path / chunk_metadata.chunk_path
        chunk_data = self._load_chunk_hdf5(chunk_path)
        
        # Cache management - remove oldest if cache is full
        if len(self.chunk_cache) >= self.config.chunk_cache_size:
            self._evict_oldest_chunk()
        
        # Add to cache
        self.chunk_cache[chunk_id] = chunk_data
        self.cache_access_times[chunk_id] = time.time()
        
        return chunk_data
    
    def _load_chunk_hdf5(self, chunk_path: Path) -> Dict[str, Any]:
        """Load chunk data from HDF5 file."""
        with h5py.File(chunk_path, 'r') as f:
            # Load expression matrix
            X = f['X'][:]
            
            # Load gene names
            genes = [g.decode('utf-8') if isinstance(g, bytes) else str(g) for g in f['genes'][:]]
            
            # Load cell names
            cells = [c.decode('utf-8') if isinstance(c, bytes) else str(c) for c in f['cells'][:]]
            
            # Load metadata
            obs_data = {}
            if 'obs' in f:
                for key in f['obs'].keys():
                    data = f['obs'][key][:]
                    if data.dtype.kind == 'S':
                        data = [d.decode('utf-8') if isinstance(d, bytes) else str(d) for d in data]
                    obs_data[key] = data
            
            return {
                'X': X,
                'genes': genes,
                'cells': cells,
                'obs': obs_data
            }
    
    def _evict_oldest_chunk(self):
        """Remove the least recently used chunk from cache."""
        if not self.cache_access_times:
            return
        
        oldest_chunk = min(self.cache_access_times.keys(), 
                          key=lambda k: self.cache_access_times[k])
        
        del self.chunk_cache[oldest_chunk]
        del self.cache_access_times[oldest_chunk]
        
        logger.debug(f"Evicted chunk {oldest_chunk} from cache")
    
    def __len__(self) -> int:
        """Return total number of cells."""
        return self.total_cells
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Get a single cell's data."""
        if idx >= self.total_cells:
            raise IndexError(f"Index {idx} out of range (total cells: {self.total_cells})")
        
        # Find which chunk and local index
        chunk_id, local_idx = self.cell_to_chunk_map[idx]
        
        # Load chunk data
        chunk_data = self._load_chunk(chunk_id)
        
        # Extract cell data
        gene_expression = torch.tensor(chunk_data['X'][local_idx], dtype=torch.float32)
        
        # Extract metadata
        metadata = {}
        for key, values in chunk_data['obs'].items():
            metadata[key] = values[local_idx]
        
        return gene_expression, metadata
    
    def get_chunk_data(self, chunk_id: int) -> Dict[str, Any]:
        """Get entire chunk data for batch processing."""
        return self._load_chunk(chunk_id)
    
    def get_genes(self) -> List[str]:
        """Get list of gene names."""
        return self.processing_index.gene_list.copy()


class ChunkedDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for chunked datasets."""
    
    def __init__(self, config: DatasetConfig):
        super().__init__()
        self.config = config
        
        # Datasets will be created in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for different stages."""
        data_path = Path(self.config.data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")
        
        if stage == "fit" or stage is None:
            # Create train/val split
            full_dataset = ChunkedDataset(str(data_path), self.config)
            
            # Simple split for now - could be made more sophisticated
            total_size = len(full_dataset)
            train_size = int(0.8 * total_size)
            val_size = total_size - train_size
            
            indices = torch.randperm(total_size).tolist()
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]
            
            self.train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
            self.val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
            
            logger.info(f"Dataset splits: train={len(self.train_dataset)}, val={len(self.val_dataset)}")
        
        if stage == "test":
            self.test_dataset = ChunkedDataset(str(data_path), self.config)
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=self.config.drop_last,
            persistent_workers=True if self.config.num_workers > 0 else False
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=False,
            persistent_workers=True if self.config.num_workers > 0 else False
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=False
        )


def create_vae_dataloaders(config: DatasetConfig) -> ChunkedDataModule:
    """
    Create VAE dataloaders from chunked dataset.
    
    Args:
        config: Dataset configuration
        
    Returns:
        Configured PyTorch Lightning DataModule
    """
    logger.info("Creating VAE dataloaders with chunked storage...")
    
    # Log configuration
    logger.info(f"Configuration:")
    logger.info(f"  Data path: {config.data_path}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Max chunks in memory: {config.max_chunks_in_memory}")
    logger.info(f"  Chunk cache size: {config.chunk_cache_size}")
    
    # Create data module
    data_module = ChunkedDataModule(config)
    
    # Setup datasets
    data_module.setup("fit")
    
    # Log memory usage
    if config.monitor_memory:
        MemoryMonitor.log_memory_usage("DataLoader creation: ")
    
    logger.info("VAE dataloaders created successfully")
    return data_module


# Legacy compatibility functions (for existing code)
def load_and_preprocess_data(data_path: str, 
                            config: DatasetConfig,
                            needed_obs_columns: Optional[List[str]] = None) -> Tuple[torch.Tensor, pd.DataFrame]:
    """
    Legacy compatibility function - loads data from chunked format.
    
    Note: This loads ALL data into memory at once. For large datasets,
    prefer using ChunkedDataset with DataLoader.
    """
    logger.warning("Using legacy load_and_preprocess_data - consider using ChunkedDataset for better memory efficiency")
    
    # Load chunked dataset
    dataset = ChunkedDataset(data_path, config)
    
    # Load all data (memory intensive!)
    all_expressions = []
    all_metadata = []
    
    logger.info(f"Loading all {len(dataset)} cells into memory...")
    
    for i in tqdm(range(len(dataset)), desc="Loading cells"):
        gene_expression, metadata = dataset[i]
        all_expressions.append(gene_expression.numpy())
        all_metadata.append(metadata)
    
    # Convert to tensors and dataframes
    expression_matrix = torch.tensor(np.stack(all_expressions), dtype=torch.float32)
    metadata_df = pd.DataFrame(all_metadata)
    
    # Filter to needed columns if specified
    if needed_obs_columns:
        available_columns = [col for col in needed_obs_columns if col in metadata_df.columns]
        metadata_df = metadata_df[available_columns]
    
    logger.info(f"Loaded data: {expression_matrix.shape}")
    return expression_matrix, metadata_df
