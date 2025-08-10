"""
Modular Data Processor with Chunked Storage

A standalone, configurable data processing module that processes data into 
optimized chunks for efficient training. Inspired by dataset/download.py chunking strategy.

Key Features:
- Pre-chunked storage for efficient training data loading
- JSON-configurable preprocessing pipeline  
- Quality control filtering (genes, cells)
- Efficient indexing by SRX accession and batch
- Memory-optimized processing with HDF5 storage
- Reproducible outputs with comprehensive metadata

Usage:
    uv run src/data_processor.py --config config/data_config.json
    
    # Or programmatically:
    from src.data_processor import DataProcessor
    processor = DataProcessor.from_config('config/data_config.json')
    processor.process()

Output Format:
- chunk_*.h5: Pre-processed data chunks for fast loading
- chunk_index.json: Index mapping cells to chunks and metadata
- processing_config.json: Complete processing configuration
- gene_list.json: Consistent gene ordering across chunks
"""

import json
import logging
import argparse
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import warnings
from tqdm import tqdm
import gc

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """Metadata for a single chunk."""
    chunk_id: int
    chunk_path: str
    n_cells: int
    n_genes: int
    srx_accessions: List[str]
    datasets: List[str]
    file_size_mb: float
    cell_start_idx: int  # Global index where this chunk starts
    cell_end_idx: int    # Global index where this chunk ends


@dataclass
class ProcessingIndex:
    """Complete index for efficient data retrieval."""
    chunks: List[ChunkMetadata]
    gene_list: List[str]
    total_cells: int
    total_chunks: int
    processing_config: Dict[str, Any]
    processing_timestamp: str
    
    def save(self, path: Path):
        """Save index to JSON."""
        # Convert to serializable format
        chunks_dict = [asdict(chunk) for chunk in self.chunks]
        index_dict = {
            'chunks': chunks_dict,
            'gene_list': self.gene_list,
            'total_cells': self.total_cells,
            'total_chunks': self.total_chunks,
            'processing_config': self.processing_config,
            'processing_timestamp': self.processing_timestamp
        }
        
        with open(path, 'w') as f:
            json.dump(index_dict, f, indent=2)
        
        logger.info(f"Saved processing index: {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'ProcessingIndex':
        """Load index from JSON."""
        with open(path, 'r') as f:
            index_dict = json.load(f)
        
        # Convert chunks back to dataclass
        chunks = [ChunkMetadata(**chunk_dict) for chunk_dict in index_dict['chunks']]
        
        return cls(
            chunks=chunks,
            gene_list=index_dict['gene_list'],
            total_cells=index_dict['total_cells'],
            total_chunks=index_dict['total_chunks'],
            processing_config=index_dict['processing_config'],
            processing_timestamp=index_dict['processing_timestamp']
        )
    
    def get_chunk_for_cell(self, cell_idx: int) -> Optional[ChunkMetadata]:
        """Get chunk metadata for a specific cell index."""
        for chunk in self.chunks:
            if chunk.cell_start_idx <= cell_idx < chunk.cell_end_idx:
                return chunk
        return None
    
    def get_chunks_for_srx(self, srx_accession: str) -> List[ChunkMetadata]:
        """Get all chunks containing cells from a specific SRX accession."""
        return [chunk for chunk in self.chunks if srx_accession in chunk.srx_accessions]

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataProcessingConfig:
    """Configuration class for data processing pipeline."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize configuration from dictionary."""
        self.config = config_dict
        self._validate_config()
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'DataProcessingConfig':
        """Load configuration from JSON file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        logger.info(f"Loaded configuration from: {config_path}")
        return cls(config_dict)
    
    def _validate_config(self):
        """Validate configuration parameters."""
        # Handle both new comprehensive config and legacy config formats
        if 'paths' in self.config:
            # New comprehensive config format
            required_keys = ['paths']
            input_paths = self.config.get('paths', {}).get('input', {}).get('datasets', {})
            output_path = self.config.get('paths', {}).get('output', {}).get('chunks_dir')
        else:
            # Legacy config format
            required_keys = ['input_paths', 'output_path']
            input_paths = self.config.get('input_paths', {})
            output_path = self.config.get('output_path')
        
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Required configuration key missing: {key}")
        
        # Validate input paths
        if input_paths:
            for path_key, path_value in input_paths.items():
                if path_key != 'vcc_data_path' and not Path(path_value).exists():
                    logger.warning(f"Input path does not exist: {path_key} = {path_value}")
        
        # Validate output path parent directory
        if output_path:
            output_parent = Path(output_path).parent
            if not output_parent.exists():
                logger.info(f"Creating output directory: {output_parent}")
                output_parent.mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default=None):
        """Get configuration value."""
        return self.config.get(key, default)


class DataProcessor:
    """
    Modular data processor with chunked storage for efficient ML training.
    
    Processes data into optimized chunks for fast loading during training,
    with comprehensive indexing for efficient retrieval.
    """
    
    def __init__(self, config: DataProcessingConfig):
        """Initialize processor with configuration."""
        self.config = config
        
        # Handle both new comprehensive config and legacy config formats
        if 'paths' in config.config:
            # New comprehensive config format
            self.output_path = Path(config.config['paths']['output']['chunks_dir'])
            chunk_config = config.config.get('processing', {}).get('chunking', {})
            self.chunk_size = chunk_config.get('chunk_size', 25000)
            self.compression_level = chunk_config.get('compression_level', 4)
        else:
            # Legacy config format
            self.output_path = Path(config.get('output_path', '/workspace/vcc/data/processed/chunked'))
            self.chunk_size = config.get('chunk_size', 25000)
            self.compression_level = config.get('compression_level', 4)
        
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize processing state
        self.datasets = {}
        self.combined_data = None
        self.processing_log = []
        self.chunks_metadata = []
        
        logger.info(f"DataProcessor initialized with chunked storage")
        logger.info(f"Output path: {self.output_path}")
        logger.info(f"Chunk size: {self.chunk_size:,} cells per chunk")
        logger.info(f"Compression level: {self.compression_level}")
    
    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> 'DataProcessor':
        """Create processor from configuration file."""
        config = DataProcessingConfig.from_file(config_path)
        return cls(config)
    
    def process(self) -> ProcessingIndex:
        """Run the complete data processing pipeline."""
        logger.info("=" * 60)
        logger.info("STARTING CHUNKED DATA PROCESSING PIPELINE")
        logger.info("=" * 60)
        
        try:
            # Step 1: Load datasets
            self._load_datasets()
            
            # Step 2: Filter datasets
            self._filter_datasets()
            
            # Step 3: Quality control
            self._quality_control()
            
            # Step 4: Subset to control cells if requested
            self._subset_to_control_cells()
            
            # Step 5: Combine datasets
            self._combine_datasets()
            
            # Step 6: Process genes
            self._process_genes()
            
            # Step 7: Process cells
            self._process_cells()
            
            # Step 8: Create chunks and save
            processing_index = self._create_chunks_and_save()
            
            # Step 9: Save index and metadata
            self._save_processing_index(processing_index)
            
            logger.info("=" * 60)
            logger.info("CHUNKED DATA PROCESSING COMPLETE")
            logger.info(f"Total chunks: {len(self.chunks_metadata)}")
            logger.info(f"Total cells: {self.combined_data.n_obs:,}")
            logger.info(f"Total genes: {self.combined_data.n_vars:,}")
            logger.info(f"Output directory: {self.output_path}")
            logger.info("=" * 60)
            
            return processing_index
            
        except Exception as e:
            logger.error(f"Data processing failed: {e}")
            self._save_processing_log()
            raise
    
    def _load_datasets(self):
        """Load specified datasets."""
        logger.info("Step 1: Loading datasets...")
        
        # Handle both config formats
        if 'paths' in self.config.config:
            # New comprehensive config format
            input_paths = self.config.config['paths']['input']['datasets']
            dataset_config = self.config.config.get('processing', {}).get('datasets', {})
        else:
            # Legacy config format
            input_paths = self.config.get('input_paths', {})
            dataset_config = self.config.get('datasets', {})
        
        for dataset_name, dataset_path in input_paths.items():
            if dataset_name == 'vcc_data_path':
                continue  # Skip VCC metadata path
            
            # Check if this dataset should be included
            if dataset_name in dataset_config:
                dataset_settings = dataset_config[dataset_name]
                if not dataset_settings.get('include', True):
                    logger.info(f"Skipping {dataset_name} (excluded in config)")
                    continue
            
            try:
                logger.info(f"Loading {dataset_name}...")
                dataset_path = Path(dataset_path)
                
                if dataset_path.suffix == '.h5ad':
                    adata = ad.read_h5ad(dataset_path)
                elif dataset_path.suffix == '.h5':
                    adata = ad.read_h5ad(dataset_path)
                else:
                    logger.warning(f"Unsupported file format: {dataset_path}")
                    continue
                
                # Add dataset identifier
                adata.obs['dataset'] = dataset_name

                # Ensure UMI_count is present - calculate if missing
                if 'UMI_count' not in adata.obs.columns:
                    logger.info(f"  Calculating UMI_count for {dataset_name}...")
                    if hasattr(adata.X, 'toarray'):
                        umi_counts = np.array(adata.X.sum(axis=1)).flatten()
                    else:
                        umi_counts = adata.X.sum(axis=1)
                    adata.obs['UMI_count'] = umi_counts
                    logger.info(f"  Added UMI_count: mean={umi_counts.mean():.1f}")
                else:
                    # Verify existing UMI counts
                    self._verify_umi_counts(adata, f"Load_{dataset_name}")

                # Add cell type to metadata (if not already present)
                if 'cell_type' not in adata.obs.columns:
                    try: 
                        if dataset_name in dataset_config:
                            adata.obs['cell_type'] = dataset_config[dataset_name].get('cell_type', dataset_name)
                        else:
                            adata.obs['cell_type'] = dataset_name
                    except KeyError:
                        logger.warning(f"Cell type not found for {dataset_name}, using dataset name")
                        adata.obs['cell_type'] = dataset_name
                
                # Apply dataset-specific filters if configured
                if dataset_name in dataset_config:
                    adata = self._apply_dataset_filters(adata, dataset_config[dataset_name])
                
                self.datasets[dataset_name] = adata
                logger.info(f"Loaded {dataset_name}: {adata.shape}")
                
            except Exception as e:
                logger.error(f"Failed to load {dataset_name}: {e}")
                # Check if dataset is required
                is_required = False
                if dataset_name in dataset_config:
                    is_required = dataset_config[dataset_name].get('required', False)
                
                if is_required:
                    raise
        
        logger.info(f"Loaded {len(self.datasets)} datasets")
        self._log_step("dataset_loading", {"loaded_datasets": list(self.datasets.keys())})
    
    def _apply_dataset_filters(self, adata: ad.AnnData, dataset_config: Dict) -> ad.AnnData:
        """Apply dataset-specific filters."""
        original_shape = adata.shape
        
        # Subset to unique perturbations if requested
        if dataset_config.get('subset_to_unique', False):
            if 'target_gene' in adata.obs.columns:
                unique_perts = adata.obs['target_gene'].unique()
                if len(unique_perts) > 100:  # Only subset if many perturbations
                    selected_perts = np.random.choice(unique_perts, 100, replace=False)
                
                else:
                    selected_perts = unique_perts

                adata = adata[adata.obs['target_gene'].isin(selected_perts)].copy()
                logger.info(f"Subset to {len(selected_perts)} unique perturbations")
        
        # Apply cell count limits
        if 'max_cells' in dataset_config:
            max_cells = dataset_config.get('max_cells')
            if max_cells and adata.n_obs > max_cells:
                indices = np.random.choice(adata.n_obs, max_cells, replace=False)
                adata = adata[indices].copy()
                logger.info(f"Subset to {max_cells} cells")

        # Remove cells with target_gene matching the given gene list
        if 'remove_target_genes' in dataset_config:
            target_genes = dataset_config['remove_target_genes']
            if 'target_gene' in adata.obs.columns:
                # Check how many cells would be removed
                cells_before = adata.n_obs
                mask = ~adata.obs['target_gene'].isin(target_genes)
                adata = adata[mask].copy()
                cells_after = adata.n_obs
                logger.info(f"Removed cells with target genes: {target_genes}")
                logger.info(f"Cells before: {cells_before}, after: {cells_after}")
            else:
                logger.warning(f"No 'target_gene' column found in {adata.obs_names}, skipping removal of target genes")
        
        if adata.shape != original_shape:
            logger.info(f"Dataset filtered: {original_shape} -> {adata.shape}")
        
        return adata
    
    def _filter_datasets(self):
        """Filter datasets based on configuration."""
        logger.info("Step 2: Filtering datasets...")
        
        filtering_config = self.config.get('filtering', {})
        
        # Remove datasets that don't meet criteria
        datasets_to_remove = []
        for name, adata in self.datasets.items():
            # Check minimum cell count
            min_cells = filtering_config.get('min_cells_per_dataset', 0)
            if adata.n_obs < min_cells:
                logger.info(f"Removing {name}: {adata.n_obs} cells < {min_cells} minimum")
                datasets_to_remove.append(name)
                continue
            
            # Check minimum genes
            min_genes = filtering_config.get('min_genes_per_dataset', 0)
            if adata.n_vars < min_genes:
                logger.info(f"Removing {name}: {adata.n_vars} genes < {min_genes} minimum")
                datasets_to_remove.append(name)
                continue
        
        for name in datasets_to_remove:
            del self.datasets[name]
        
        logger.info(f"Retained {len(self.datasets)} datasets after filtering")
        self._log_step("dataset_filtering", {
            "removed_datasets": datasets_to_remove,
            "retained_datasets": list(self.datasets.keys())
        })
    
    def _quality_control(self):
        """Apply quality control filters to each dataset."""
        logger.info("Step 3: Quality control...")
        
        qc_config = self.config.get('quality_control', {})
        
        for name, adata in self.datasets.items():
            original_shape = adata.shape
            
            # Check UMI counts before QC
            if 'UMI_count' in adata.obs.columns:
                umi_stats_before = adata.obs['UMI_count'].describe()
                logger.info(f"  {name} UMI counts before QC: mean={umi_stats_before['mean']:.1f}")
            else:
                logger.warning(f"  {name}: UMI_count column missing before QC!")
            
            # Filter cells based on QC metrics
            if qc_config.get('filter_cells', True):
                            # Calculate QC metrics
                adata.var['mt'] = adata.var_names.str.startswith('MT-')
                sc.pp.calculate_qc_metrics(adata,
                                    qc_vars=['mt'],
                                    percent_top=None,
                                    log1p=False,
                                    inplace=True
                                    )
                # Filter outliers with permissive thresholds

                # [Can implement doublet filtering, UMI outliers etc]

                # Filter by mitochondrial gene percentage
                if 'pct_counts_mt' in adata.obs.columns:
                    max_mt_pct = qc_config.get('max_mitochondrial_percent', 15)
                    adata = adata[adata.obs.pct_counts_mt < max_mt_pct].copy()
                else:
                    logger.warning(f"No mitochondrial genes found in {name}, skipping MT% filtering")
            
            # Check UMI counts after QC
            if 'UMI_count' in adata.obs.columns:
                umi_stats_after = adata.obs['UMI_count'].describe()
                logger.info(f"  {name} UMI counts after QC: mean={umi_stats_after['mean']:.1f}")
            else:
                logger.warning(f"  {name}: UMI_count column lost during QC!")
            
            self.datasets[name] = adata
            
            if adata.shape != original_shape:
                logger.info(f"QC filtered {name}: {original_shape} -> {adata.shape}")
        
        self._log_step("quality_control", qc_config)
    
    def _combine_datasets(self):
        """Combine datasets into a single AnnData object."""
        logger.info("Step 4: Combining datasets...")
        
        if not self.datasets:
            raise ValueError("No datasets available for combination")
        
        # Check UMI counts before combination
        logger.info("Checking UMI counts before combination:")
        for name, adata in self.datasets.items():
            if 'UMI_count' in adata.obs.columns:
                umi_stats = adata.obs['UMI_count'].describe()
                logger.info(f"  {name}: UMI_count present, mean={umi_stats['mean']:.1f}, range=[{umi_stats['min']:.0f}, {umi_stats['max']:.0f}]")
            else:
                logger.warning(f"  {name}: UMI_count column missing!")
        
        # Combine datasets
        adata_list = list(self.datasets.values())
        self.combined_data = ad.concat(adata_list, join='outer', index_unique='-')
        
        # Check UMI counts after combination
        logger.info("Checking UMI counts after combination:")
        if 'UMI_count' in self.combined_data.obs.columns:
            umi_stats = self.combined_data.obs['UMI_count'].describe()
            logger.info(f"  Combined: UMI_count preserved, mean={umi_stats['mean']:.1f}, range=[{umi_stats['min']:.0f}, {umi_stats['max']:.0f}]")
            
            # Check for any NaN values that might indicate data loss
            nan_count = self.combined_data.obs['UMI_count'].isna().sum()
            if nan_count > 0:
                logger.warning(f"  Found {nan_count} NaN values in UMI_count after combination!")
        else:
            logger.error("  UMI_count column lost during combination!")
        
        logger.info(f"Combined dataset: {self.combined_data.shape}")
        logger.info(f"Datasets: {self.combined_data.obs['dataset'].value_counts().to_dict()}")
        
        self._log_step("dataset_combination", {
            "combined_shape": self.combined_data.shape,
            "dataset_counts": self.combined_data.obs['dataset'].value_counts().to_dict(),
            "umi_count_preserved": 'UMI_count' in self.combined_data.obs.columns
        })
    
    def _process_genes(self):
        """Process genes: filtering, ordering, VCC compliance."""
        logger.info("Step 5: Processing genes...")
        
        gene_config = self.config.get('gene_processing', {})
        
        # Check UMI counts before gene processing
        if 'UMI_count' in self.combined_data.obs.columns:
            umi_stats_before = self.combined_data.obs['UMI_count'].describe()
            logger.info(f"UMI counts before gene processing: mean={umi_stats_before['mean']:.1f}")
        else:
            logger.warning("UMI_count column missing before gene processing!")
        
        # Load VCC gene requirements if specified
        vcc_data_path = self.config.get('input_paths', {}).get('vcc_data_path')
        expected_genes = None
        
        if vcc_data_path and gene_config.get('use_vcc_genes', True):
            gene_names_file = Path(vcc_data_path) / 'gene_names.csv'
            if gene_names_file.exists():
                expected_genes = pd.read_csv(gene_names_file, header=None)[0].values
                logger.info(f"Loaded VCC gene requirements: {len(expected_genes)} genes")
        
        # Filter genes based on configuration
        original_n_genes = self.combined_data.n_vars
        
        # Keep only specified genes if provided
        include_genes = gene_config.get('include_genes', [])
        if include_genes:
            available_genes = self.combined_data.var_names.intersection(include_genes)
            self.combined_data = self.combined_data[:, available_genes].copy()
            logger.info(f"Filtered to specified genes: {len(available_genes)}/{len(include_genes)}")
        
        # Order genes according to VCC requirements
        if expected_genes is not None and gene_config.get('order_by_vcc', True):
            # Find intersection with expected genes
            available_expected = self.combined_data.var_names.intersection(expected_genes)
            missing_genes = set(expected_genes) - set(available_expected)
            
            if missing_genes:
                logger.warning(f"Missing {len(missing_genes)} VCC-required genes")
                if gene_config.get('strict_vcc_compliance', False):
                    raise ValueError(f"Missing required VCC genes: {list(missing_genes)[:10]}...")
            
            # Order by VCC gene list
            self.combined_data = self.combined_data[:, available_expected].copy()
            logger.info(f"Ordered genes by VCC requirements: {len(available_expected)} genes")
        
        # Check UMI counts after gene processing
        if 'UMI_count' in self.combined_data.obs.columns:
            umi_stats_after = self.combined_data.obs['UMI_count'].describe()
            logger.info(f"UMI counts after gene processing: mean={umi_stats_after['mean']:.1f}")
            
            # Check if UMI counts are still the same (they should be since we're only filtering genes)
            if 'umi_stats_before' in locals():
                if abs(umi_stats_before['mean'] - umi_stats_after['mean']) > 0.001:
                    logger.warning("UMI count statistics changed during gene processing - this shouldn't happen!")
        else:
            logger.error("UMI_count column lost during gene processing!")
        
        self._log_step("gene_processing", {
            "original_genes": original_n_genes,
            "final_genes": self.combined_data.n_vars,
            "vcc_compliance": expected_genes is not None,
            "umi_count_preserved": 'UMI_count' in self.combined_data.obs.columns
        })
    
    def _process_cells(self):
        """Process cells: filtering, balancing."""
        logger.info("Step 6: Processing cells...")
        
        cell_config = self.config.get('cell_processing', {})
        original_n_cells = self.combined_data.n_obs
        
        # Check UMI counts before cell processing
        if 'UMI_count' in self.combined_data.obs.columns:
            umi_stats_before = self.combined_data.obs['UMI_count'].describe()
            logger.info(f"UMI counts before cell processing: {original_n_cells} cells, mean={umi_stats_before['mean']:.1f}")
        else:
            logger.warning("UMI_count column missing before cell processing!")
        
        # Remove cells with too many/few non-zero genes
        if cell_config.get('filter_by_gene_count', False):
            min_genes = cell_config.get('min_genes_per_cell', 200)
            
            # Store original indices to track which cells are kept
            original_indices = self.combined_data.obs.index.copy()
            
            sc.pp.filter_cells(self.combined_data, min_genes=min_genes)
            
            # Check which cells were kept
            kept_indices = self.combined_data.obs.index
            n_removed = len(original_indices) - len(kept_indices)
            logger.info(f"Removed {n_removed} cells with <{min_genes} genes")
        
        # Check UMI counts after cell processing
        if 'UMI_count' in self.combined_data.obs.columns:
            umi_stats_after = self.combined_data.obs['UMI_count'].describe()
            logger.info(f"UMI counts after cell processing: {self.combined_data.n_obs} cells, mean={umi_stats_after['mean']:.1f}")
            
            # Check for data consistency
            nan_count = self.combined_data.obs['UMI_count'].isna().sum()
            if nan_count > 0:
                logger.warning(f"Found {nan_count} NaN values in UMI_count after cell processing!")
        else:
            logger.error("UMI_count column lost during cell processing!")
        
        self._log_step("cell_processing", {
            "original_cells": original_n_cells,
            "final_cells": self.combined_data.n_obs,
            "umi_count_preserved": 'UMI_count' in self.combined_data.obs.columns
        })

    def _identify_control_cells(self) -> np.ndarray:
        """Identify control cells in the combined dataset."""
        if self.combined_data is None:
            raise ValueError("Combined data not available")
        
        control_mask = np.zeros(self.combined_data.n_obs, dtype=bool)
        
        # Use the same control indicators as in _subset_to_control_cells
        control_indicators = self.config.get('control_indicators', {
            'target_gene': ['non-targeting', 'control', 'ctrl'],
            'perturbation_type': ['control', 'ctrl', 'none'],
            'condition': ['control', 'ctrl', 'dmso', 'vehicle']
        })
        
        # Check each potential control indicator column
        for col_name, control_values in control_indicators.items():
            if col_name in self.combined_data.obs.columns:
                col_control_mask = self.combined_data.obs[col_name].isin(control_values)
                control_mask |= col_control_mask
                logger.info(f"Found {col_control_mask.sum()} control cells via {col_name}")

                # Print unique target genes in control cells
                unique_target_genes = self.combined_data.obs.loc[col_control_mask, 'target_gene'].unique()
                logger.info(f"Unique target genes in control cells: {unique_target_genes}")

        
        # Check for additional control patterns in target_gene
        if 'target_gene' in self.combined_data.obs.columns:
            target_genes = self.combined_data.obs['target_gene'].astype(str).str.lower()
            additional_control_mask = (
                target_genes.str.contains('control', na=False) |
                target_genes.str.contains('non-target', na=False) |
                target_genes.str.contains('neg', na=False) |
                target_genes.str.contains('scramble', na=False)
            )
            control_mask |= additional_control_mask
            logger.info(f"Found {additional_control_mask.sum()} additional control cells via pattern matching")

            # Print all unique found target genes
            unique_target_genes = target_genes[control_mask].unique()
            logger.info(f"Unique target genes in control cells: {unique_target_genes}")
            # logger.info(f"  {', '.join(unique_target_genes[:10])}...")  # Show first 10 unique target genes
        
        total_control = control_mask.sum()
        logger.info(f"Total control cells identified: {total_control}/{self.combined_data.n_obs} ({100*total_control/self.combined_data.n_obs:.1f}%)")

        # Print total control target genes
        if 'target_gene' in self.combined_data.obs.columns:
            control_target_genes = self.combined_data.obs.loc[control_mask, 'target_gene'].unique()
            logger.info(f"Control target genes: {len(control_target_genes)} unique genes")
            logger.info(f"  {', '.join(control_target_genes[:-1])}; {control_target_genes[-1]}")  # Show all but last, then last separately
        
        return control_mask
    
    def _subset_to_control_cells(self):
        """Subset datasets to only control cells if specified in configuration."""
        config_subset = self.config.get('subset_to_control_cells', False)
        
        if not config_subset:
            logger.info("Step 3.5: Skipping control cell subsetting (not requested)")
            return
        
        logger.info("Step 3.5: Subsetting to control cells only...")
        
        control_indicators = self.config.get('control_indicators', {
            'target_gene': ['non-targeting', 'control', 'ctrl'],
            'perturbation_type': ['control', 'ctrl', 'none'],
            'condition': ['control', 'ctrl', 'dmso', 'vehicle']
        })
        
        datasets_to_remove = []
        
        for name, adata in self.datasets.items():
            original_shape = adata.shape
            control_mask = np.zeros(adata.n_obs, dtype=bool)
            
            # Check UMI counts before control filtering
            if 'UMI_count' in adata.obs.columns:
                umi_stats_before = adata.obs['UMI_count'].describe()
                logger.info(f"  {name} UMI counts before control filtering: mean={umi_stats_before['mean']:.1f}")
            
            # Check each potential control indicator column
            control_cells_found = False
            
            for col_name, control_values in control_indicators.items():
                if col_name in adata.obs.columns:
                    col_control_mask = adata.obs[col_name].isin(control_values)
                    n_control = col_control_mask.sum()
                    
                    if n_control > 0:
                        control_mask |= col_control_mask
                        control_cells_found = True
                        logger.info(f"    Found {n_control} control cells via {col_name} column")
            
            # Check for additional control patterns
            if 'target_gene' in adata.obs.columns:
                # Look for cells where target_gene contains common control patterns
                target_genes = adata.obs['target_gene'].astype(str).str.lower()
                additional_control_mask = (
                    target_genes.str.contains('control', na=False) |
                    target_genes.str.contains('non-target', na=False) |
                    target_genes.str.contains('neg', na=False) |
                    target_genes.str.contains('scramble', na=False)
                )
                
                if additional_control_mask.sum() > 0:
                    control_mask |= additional_control_mask
                    control_cells_found = True
                    logger.info(f"    Found {additional_control_mask.sum()} additional control cells via pattern matching")
            
            # Apply control filtering if control cells were found
            if control_cells_found:
                n_control_total = control_mask.sum()
                
                if n_control_total == 0:
                    logger.warning(f"  {name}: No control cells found despite indicators - removing dataset")
                    datasets_to_remove.append(name)
                    continue
                
                # Filter to only control cells
                adata_control = adata[control_mask].copy()
                
                # Verify UMI counts are preserved
                if 'UMI_count' in adata_control.obs.columns:
                    umi_stats_after = adata_control.obs['UMI_count'].describe()
                    logger.info(f"  {name} UMI counts after control filtering: mean={umi_stats_after['mean']:.1f}")
                
                self.datasets[name] = adata_control
                
                logger.info(f"  {name}: Filtered to control cells only: {original_shape} -> {adata_control.shape}")
                logger.info(f"    Kept {n_control_total}/{original_shape[0]} cells ({100*n_control_total/original_shape[0]:.1f}%)")
                
            else:
                logger.warning(f"  {name}: No control cell indicators found - keeping all cells")
                logger.info(f"    Available columns: {list(adata.obs.columns)}")
                logger.info(f"    Unique values in potential control columns:")
                
                for col_name in control_indicators.keys():
                    if col_name in adata.obs.columns:
                        unique_vals = adata.obs[col_name].unique()[:10]  # Show first 10
                        logger.info(f"      {col_name}: {unique_vals}")
        
        # Remove datasets that had no control cells
        for name in datasets_to_remove:
            logger.warning(f"Removing {name}: no control cells found")
            del self.datasets[name]
        
        # Summary
        total_cells_remaining = sum(adata.n_obs for adata in self.datasets.values())
        logger.info(f"Control cell filtering completed:")
        logger.info(f"  Datasets remaining: {len(self.datasets)}")
        logger.info(f"  Total cells remaining: {total_cells_remaining}")
        
        self._log_step("control_cell_subsetting", {
            "datasets_remaining": list(self.datasets.keys()),
            "datasets_removed": datasets_to_remove,
            "total_cells_remaining": total_cells_remaining,
            "control_indicators_used": control_indicators
        })
    
    def _finalize_and_save(self):
        """Finalize processing and save results."""
        logger.info("Step 8: Finalizing and saving...")
        
        if self.processed_data is None:
            self.processed_data = self.combined_data
        
        # Final UMI count check
        logger.info("Final UMI count verification:")
        if 'UMI_count' in self.processed_data.obs.columns:
            umi_stats = self.processed_data.obs['UMI_count'].describe()
            logger.info(f"  Final UMI counts: {self.processed_data.n_obs} cells, mean={umi_stats['mean']:.1f}, range=[{umi_stats['min']:.0f}, {umi_stats['max']:.0f}]")
            
            # Check for data quality issues
            nan_count = self.processed_data.obs['UMI_count'].isna().sum()
            zero_count = (self.processed_data.obs['UMI_count'] == 0).sum()
            
            if nan_count > 0:
                logger.warning(f"  Found {nan_count} NaN values in UMI_count!")
            if zero_count > 0:
                logger.warning(f"  Found {zero_count} cells with 0 UMI count!")
            
            logger.info("  ✓ UMI_count successfully preserved through processing pipeline")
        else:
            logger.error("  ✗ UMI_count column missing in final data!")
            
            # Try to reconstruct UMI counts from expression data if possible
            logger.info("  Attempting to reconstruct UMI counts from expression data...")
            if hasattr(self.processed_data.X, 'toarray'):
                umi_counts = np.array(self.processed_data.X.sum(axis=1)).flatten()
            else:
                umi_counts = self.processed_data.X.sum(axis=1)
            
            self.processed_data.obs['UMI_count_reconstructed'] = umi_counts
            logger.info(f"  Reconstructed UMI counts: mean={umi_counts.mean():.1f}")
        
        # Add processing metadata
        self.processed_data.uns['processing'] = {
            'timestamp': str(datetime.now().isoformat()),
            'config': str(self.config.config),
            'processing_log': str(self.processing_log),
            'umi_count_preserved': 'UMI_count' in self.processed_data.obs.columns
        }
        
        # Save processed data
        output_file = self.output_path / 'processed_data.h5ad'
        self.processed_data.write_h5ad(output_file)
        logger.info(f"Saved processed data: {output_file}")
        
        # Save processing log
        self._save_processing_log()
        
        # Save summary statistics
        self._save_summary()
    
    def _log_step(self, step_name: str, details: Dict):
        """Log a processing step."""
        log_entry = {
            'step': step_name,
            'timestamp': datetime.now().isoformat(),
            'details': details
        }
        self.processing_log.append(log_entry)
    
    def _save_processing_log(self):
        """Save processing log to file."""
        log_file = self.output_path / 'processing_log.json'
        with open(log_file, 'w') as f:
            json.dump(self.processing_log, f, indent=2, default=str)
        logger.info(f"Processing log saved: {log_file}")
    
    def _save_summary(self):
        """Save processing summary."""
        if self.processed_data is None:
            return
        
        summary = {
            'final_shape': {
                'n_cells': int(self.processed_data.n_obs),
                'n_genes': int(self.processed_data.n_vars)
            },
            'datasets': self.processed_data.obs['dataset'].value_counts().to_dict(),
            'processing_timestamp': datetime.now().isoformat(),
            'config_file': str(self.config.get('config_file', 'unknown'))
        }
        
        if 'target_gene' in self.processed_data.obs:
            summary['perturbations'] = {
                'n_unique': int(self.processed_data.obs['target_gene'].nunique()),
                'total_cells': int(self.processed_data.n_obs)
            }
        
        summary_file = self.output_path / 'processing_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Processing summary saved: {summary_file}")
    
    def _verify_umi_counts(self, adata: ad.AnnData, step_name: str) -> bool:
        """
        Verify UMI count integrity in a dataset.
        
        Args:
            adata: AnnData object to check
            step_name: Name of the processing step for logging
            
        Returns:
            bool: True if UMI counts are present and valid
        """
        if 'UMI_count' not in adata.obs.columns:
            logger.error(f"{step_name}: UMI_count column missing!")
            return False
        
        umi_counts = adata.obs['UMI_count']
        
        # Check for NaN values
        nan_count = umi_counts.isna().sum()
        if nan_count > 0:
            logger.warning(f"{step_name}: {nan_count} NaN values in UMI_count")
        
        # Check for zero values
        zero_count = (umi_counts == 0).sum()
        if zero_count > 0:
            logger.warning(f"{step_name}: {zero_count} cells with 0 UMI count")
        
        # Check for negative values
        negative_count = (umi_counts < 0).sum()
        if negative_count > 0:
            logger.error(f"{step_name}: {negative_count} cells with negative UMI count!")
        
        # Log statistics
        stats = umi_counts.describe()
        logger.info(f"{step_name}: UMI count stats - mean={stats['mean']:.1f}, "
                   f"median={stats['50%']:.1f}, range=[{stats['min']:.0f}, {stats['max']:.0f}]")
        
        return nan_count == 0 and negative_count == 0

    def _create_chunks_and_save(self) -> ProcessingIndex:
        """Create optimized chunks for efficient training data loading."""
        logger.info("Step 7: Creating optimized chunks...")
        
        if self.combined_data is None:
            raise ValueError("No combined data available for chunking")
        
        total_cells = self.combined_data.n_obs
        n_chunks = (total_cells + self.chunk_size - 1) // self.chunk_size
        
        logger.info(f"Creating {n_chunks} chunks from {total_cells:,} cells")
        logger.info(f"Chunk size: {self.chunk_size:,} cells per chunk")
        
        # Get gene list for consistent ordering
        gene_list = self.combined_data.var_names.tolist()
        
        # Create chunks
        chunks_metadata = []
        cell_start_idx = 0
        
        for chunk_idx in tqdm(range(n_chunks), desc="Creating chunks"):
            cell_end_idx = min(cell_start_idx + self.chunk_size, total_cells)
            
            # Extract chunk data
            chunk_data = self.combined_data[cell_start_idx:cell_end_idx].copy()
            
            # Create chunk file
            chunk_filename = f"chunk_{chunk_idx:04d}.h5"
            chunk_path = self.output_path / chunk_filename
            
            # Save chunk with optimized HDF5 format
            self._save_chunk_hdf5(chunk_data, chunk_path, chunk_idx)
            
            # Get chunk metadata
            chunk_metadata = self._get_chunk_metadata(
                chunk_data, chunk_idx, chunk_filename, cell_start_idx, cell_end_idx
            )
            chunks_metadata.append(chunk_metadata)
            
            # Log progress
            if chunk_idx % 10 == 0 or chunk_idx == n_chunks - 1:
                logger.info(f"  Created chunk {chunk_idx + 1}/{n_chunks}")
            
            cell_start_idx = cell_end_idx
            
            # Memory cleanup
            del chunk_data
            gc.collect()
        
        # Create processing index
        processing_index = ProcessingIndex(
            chunks=chunks_metadata,
            gene_list=gene_list,
            total_cells=total_cells,
            total_chunks=n_chunks,
            processing_config=self.config.config,
            processing_timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Successfully created {len(chunks_metadata)} chunks")
        self._log_step("chunk_creation", {
            "total_chunks": len(chunks_metadata),
            "total_cells": total_cells,
            "chunk_size": self.chunk_size
        })
        
        return processing_index
    
    def _save_chunk_hdf5(self, chunk_data: ad.AnnData, chunk_path: Path, chunk_idx: int):
        """Save chunk data in optimized HDF5 format."""
        with h5py.File(chunk_path, 'w') as f:
            # Save expression matrix (sparse format)
            if hasattr(chunk_data.X, 'toarray'):
                # Convert sparse to dense for HDF5 storage with compression
                X_dense = chunk_data.X.toarray().astype(np.float32)
            else:
                X_dense = chunk_data.X.astype(np.float32)
            
            # Store with compression
            f.create_dataset(
                'X', 
                data=X_dense, 
                compression='gzip', 
                compression_opts=self.compression_level,
                shuffle=True
            )
            
            # Save gene names
            gene_names_encoded = [name.encode('utf-8') for name in chunk_data.var_names]
            f.create_dataset('genes', data=gene_names_encoded)
            
            # Save cell names
            cell_names_encoded = [name.encode('utf-8') for name in chunk_data.obs_names]
            f.create_dataset('cells', data=cell_names_encoded)
            
            # Save metadata (obs)
            obs_group = f.create_group('obs')
            for col_name, col_data in chunk_data.obs.items():
                if col_data.dtype == 'object':
                    # String data
                    encoded_data = [str(val).encode('utf-8') for val in col_data]
                    obs_group.create_dataset(col_name, data=encoded_data)
                else:
                    # Numeric data
                    obs_group.create_dataset(col_name, data=col_data.values)
            
            # Save chunk metadata
            f.attrs['chunk_id'] = chunk_idx
            f.attrs['n_cells'] = chunk_data.n_obs
            f.attrs['n_genes'] = chunk_data.n_vars
            f.attrs['creation_time'] = datetime.now().isoformat()
    
    def _get_chunk_metadata(self, chunk_data: ad.AnnData, chunk_idx: int, 
                           chunk_filename: str, cell_start_idx: int, cell_end_idx: int) -> ChunkMetadata:
        """Extract metadata from a chunk."""
        chunk_path = self.output_path / chunk_filename
        file_size_mb = chunk_path.stat().st_size / (1024 * 1024)
        
        # Get unique SRX accessions in this chunk
        srx_accessions = []
        if 'SRX_accession' in chunk_data.obs.columns:
            srx_accessions = chunk_data.obs['SRX_accession'].unique().tolist()
        
        # Get unique datasets in this chunk
        datasets = []
        if 'dataset' in chunk_data.obs.columns:
            datasets = chunk_data.obs['dataset'].unique().tolist()
        
        return ChunkMetadata(
            chunk_id=chunk_idx,
            chunk_path=chunk_filename,
            n_cells=chunk_data.n_obs,
            n_genes=chunk_data.n_vars,
            srx_accessions=srx_accessions,
            datasets=datasets,
            file_size_mb=file_size_mb,
            cell_start_idx=cell_start_idx,
            cell_end_idx=cell_end_idx
        )
    
    def _save_processing_index(self, processing_index: ProcessingIndex):
        """Save the processing index and related files."""
        logger.info("Step 8: Saving processing index and metadata...")
        
        # Save main index
        index_path = self.output_path / "chunk_index.json"
        processing_index.save(index_path)
        
        # Save gene list separately for easy access
        gene_list_path = self.output_path / "gene_list.json"
        with open(gene_list_path, 'w') as f:
            json.dump(processing_index.gene_list, f, indent=2)
        
        # Save processing configuration
        config_path = self.output_path / "processing_config.json"
        with open(config_path, 'w') as f:
            json.dump(processing_index.processing_config, f, indent=2, default=str)
        
        # Save processing log
        self._save_processing_log()
        
        # Save summary statistics
        self._save_chunk_summary(processing_index)
        
        logger.info(f"Processing index and metadata saved to: {self.output_path}")
    
    def _save_chunk_summary(self, processing_index: ProcessingIndex):
        """Save summary statistics about the chunks."""
        summary = {
            'overview': {
                'total_chunks': processing_index.total_chunks,
                'total_cells': processing_index.total_cells,
                'total_genes': len(processing_index.gene_list),
                'chunk_size': self.chunk_size,
                'processing_timestamp': processing_index.processing_timestamp
            },
            'chunks': []
        }
        
        # Add chunk details
        for chunk in processing_index.chunks:
            chunk_summary = {
                'chunk_id': chunk.chunk_id,
                'file': chunk.chunk_path,
                'cells': chunk.n_cells,
                'size_mb': round(chunk.file_size_mb, 2),
                'srx_count': len(chunk.srx_accessions),
                'dataset_count': len(chunk.datasets)
            }
            summary['chunks'].append(chunk_summary)
        
        # Add storage statistics
        total_size_mb = sum(chunk.file_size_mb for chunk in processing_index.chunks)
        avg_chunk_size_mb = total_size_mb / len(processing_index.chunks)
        
        summary['storage'] = {
            'total_size_mb': round(total_size_mb, 2),
            'avg_chunk_size_mb': round(avg_chunk_size_mb, 2),
            'compression_level': self.compression_level
        }
        
        # Save summary
        summary_path = self.output_path / "chunk_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Chunk summary saved: {summary_path}")
        logger.info(f"Total storage: {total_size_mb:.1f} MB across {len(processing_index.chunks)} chunks")

def main():
    """Command-line interface for data processing."""
    parser = argparse.ArgumentParser(description="Virtual Cell Challenge Data Processor")
    parser.add_argument('--config', required=True, help='Path to configuration JSON file')
    parser.add_argument('--validate-only', action='store_true', help='Only validate configuration')
    parser.add_argument('--output', help='Override output path')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = DataProcessingConfig.from_file(args.config)
        
        if args.validate_only:
            logger.info("Configuration validation passed")
            return
        
        # Override output path if provided
        if args.output:
            config.config['output_path'] = args.output
        
        # Create processor and run
        processor = DataProcessor(config)
        result = processor.process()
        
        logger.info(f"Data processing completed successfully!")
        logger.info(f"Final dataset: {result.shape}")
        
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        raise


if __name__ == "__main__":
    main()
