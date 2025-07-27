#!/usr/bin/env python3
"""
Download a randomly sampled subset of human single-cell RNA-seq data from scBaseCount.
Optimized for discrete diffusion transformer training with PyTorch.

Usage:
    python download.py --cell-count 1000000 --output-dir data/scRNA
"""

import argparse
import os
import json
import random
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import scanpy as sc
import gcsfs
import pyarrow.dataset as ds
import h5py
from tqdm import tqdm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class SampleMetadata:
    """Metadata for a single sample/experiment."""
    srx_accession: str
    file_path: str
    obs_count: int
    tissue: str
    tech_10x: str
    cell_prep: str
    disease: str
    perturbation: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DatasetConfig:
    """Configuration for the downloaded dataset."""
    total_cells_requested: int
    total_cells_downloaded: int
    total_samples: int
    samples: List[SampleMetadata]
    download_timestamp: str
    gene_list: List[str]
    max_expression_value: int
    sparsity: float
    
    def save(self, path: Path):
        """Save configuration to JSON."""
        config_dict = asdict(self)
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: Path):
        """Load configuration from JSON."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


class ScRNADownloader:
    """Download and process single-cell RNA-seq data for ML training."""
    
    def __init__(self, output_dir: str = "data/scRNA"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize GCS filesystem
        self.fs = gcsfs.GCSFileSystem()
        
        # Data paths
        self.gcs_base_path = "gs://arc-scbasecount/2025-02-25/"
        self.feature_type = "GeneFull_Ex50pAS"
        
        # Create subdirectories
        self.raw_dir = self.output_dir / "raw"
        self.processed_dir = self.output_dir / "processed"
        self.raw_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized downloader with output directory: {self.output_dir}")
    
    def load_human_metadata(self) -> pd.DataFrame:
        """Load metadata for all human samples."""
        logger.info("Loading human sample metadata...")
        
        metadata_path = f"{self.gcs_base_path}metadata/{self.feature_type}/Homo_sapiens/sample_metadata.parquet"
        metadata = ds.dataset(metadata_path, filesystem=self.fs, format="parquet").to_table().to_pandas()
        
        logger.info(f"Found {len(metadata):,} human samples with {metadata['obs_count'].sum():,} total cells")
        return metadata
    
    def select_random_samples(self, metadata: pd.DataFrame, target_cells: int) -> List[SampleMetadata]:
        """Randomly select samples until we reach the target cell count."""
        logger.info(f"Selecting random samples to reach {target_cells:,} cells...")
        
        # Shuffle samples
        shuffled = metadata.sample(frac=1, random_state=42).reset_index(drop=True)
        
        selected_samples = []
        total_cells = 0
        
        for _, row in shuffled.iterrows():
            if total_cells >= target_cells:
                break
                
            sample = SampleMetadata(
                srx_accession=row['srx_accession'],
                file_path=row['file_path'],
                obs_count=row['obs_count'],
                tissue=row['tissue'],
                tech_10x=row['tech_10x'],
                cell_prep=row['cell_prep'],
                disease=row['disease'],
                perturbation=row['perturbation']
            )
            
            selected_samples.append(sample)
            total_cells += row['obs_count']
        
        logger.info(f"Selected {len(selected_samples)} samples with {total_cells:,} total cells")
        return selected_samples, total_cells
    
    def download_sample(self, sample: SampleMetadata) -> Optional[sc.AnnData]:
        """Download and load a single sample."""
        try:
            output_path = self.raw_dir / f"{sample.srx_accession}.h5ad"
            
            if output_path.exists():
                logger.info(f"Loading existing file: {sample.srx_accession}")
                return sc.read_h5ad(output_path)
            
            logger.info(f"Downloading {sample.srx_accession} ({sample.obs_count:,} cells)")
            self.fs.get(sample.file_path, str(output_path))
            
            return sc.read_h5ad(output_path)
            
        except Exception as e:
            logger.error(f"Failed to download {sample.srx_accession}: {e}")
            return None
    
    def process_and_save_batch(self, adata_list: List[sc.AnnData], batch_idx: int, 
                              gene_union: Optional[np.ndarray] = None) -> Dict:
        """Process a batch of samples and save in ML-friendly format."""
        logger.info(f"Processing batch {batch_idx} with {len(adata_list)} samples...")
        
        # Concatenate all samples
        adata_combined = sc.concat(adata_list, join='outer', fill_value=0)
        
        # If gene_union provided, reindex to ensure consistent gene order
        if gene_union is not None:
            adata_combined = adata_combined[:, gene_union]
        
        # Convert to dense array for ML (sparse is harder to work with in PyTorch)
        if hasattr(adata_combined.X, 'toarray'):
            X_dense = adata_combined.X.toarray()
        else:
            X_dense = adata_combined.X
        
        # Convert to int16 to save space (max gene count rarely exceeds 32k)
        X_dense = X_dense.astype(np.int16)
        
        # Save as HDF5 for efficient loading
        h5_path = self.processed_dir / f"batch_{batch_idx:04d}.h5"
        with h5py.File(h5_path, 'w') as f:
            # Expression matrix
            f.create_dataset('X', data=X_dense, compression='gzip', compression_opts=4)
            
            # Gene names
            f.create_dataset('genes', data=np.array(adata_combined.var_names, dtype='S'))
            
            # Cell barcodes
            f.create_dataset('cells', data=np.array(adata_combined.obs_names, dtype='S'))
            
            # Cell metadata
            obs_df = adata_combined.obs
            for col in ['gene_count', 'umi_count', 'SRX_accession']:
                if col in obs_df.columns:
                    if col == 'SRX_accession':
                        f.create_dataset(f'obs/{col}', data=np.array(obs_df[col], dtype='S'))
                    else:
                        f.create_dataset(f'obs/{col}', data=obs_df[col].values)
        
        # Calculate statistics
        stats = {
            'n_cells': X_dense.shape[0],
            'n_genes': X_dense.shape[1],
            'sparsity': (X_dense == 0).sum() / X_dense.size,
            'max_value': int(X_dense.max()),
            'mean_counts_per_cell': float(X_dense.sum(axis=1).mean()),
            'file_size_mb': os.path.getsize(h5_path) / 1024 / 1024
        }
        
        logger.info(f"Saved batch {batch_idx}: {stats['n_cells']:,} cells, "
                   f"{stats['sparsity']:.1%} sparse, {stats['file_size_mb']:.1f} MB")
        
        return stats
    
    def download_and_process(self, target_cells: int = 1_000_000, batch_size: int = 10):
        """Main download and processing pipeline."""
        start_time = datetime.now()
        
        # Load metadata
        metadata = self.load_human_metadata()
        
        # Select random samples
        selected_samples, total_cells = self.select_random_samples(metadata, target_cells)
        
        # Get union of all genes (for consistent indexing)
        logger.info("Determining gene union across samples...")
        gene_union = None
        
        # Process in batches to manage memory
        batch_stats = []
        batch_idx = 0
        current_batch = []
        
        for i, sample in enumerate(tqdm(selected_samples, desc="Downloading samples")):
            adata = self.download_sample(sample)
            
            if adata is not None:
                current_batch.append(adata)
                
                # Update gene union
                if gene_union is None:
                    gene_union = adata.var_names
                else:
                    gene_union = gene_union.union(adata.var_names)
            
            # Process batch when full or last sample
            if len(current_batch) >= batch_size or i == len(selected_samples) - 1:
                if current_batch:
                    stats = self.process_and_save_batch(current_batch, batch_idx, gene_union)
                    batch_stats.append(stats)
                    batch_idx += 1
                    current_batch = []
        
        # Save configuration
        config = DatasetConfig(
            total_cells_requested=target_cells,
            total_cells_downloaded=total_cells,
            total_samples=len(selected_samples),
            samples=[s.to_dict() for s in selected_samples],
            download_timestamp=datetime.now().isoformat(),
            gene_list=list(gene_union),
            max_expression_value=max(s['max_value'] for s in batch_stats),
            sparsity=np.mean([s['sparsity'] for s in batch_stats])
        )
        
        config.save(self.output_dir / "config.json")
        
        # Clean up raw files if requested
        if input("\nDelete raw .h5ad files to save space? (y/n): ").lower() == 'y':
            shutil.rmtree(self.raw_dir)
            logger.info("Deleted raw files")
        
        # Summary
        total_size_mb = sum(s['file_size_mb'] for s in batch_stats)
        duration = datetime.now() - start_time
        
        logger.info("\n" + "="*50)
        logger.info("DOWNLOAD COMPLETE")
        logger.info(f"Total cells: {total_cells:,}")
        logger.info(f"Total samples: {len(selected_samples)}")
        logger.info(f"Total genes: {len(gene_union):,}")
        logger.info(f"Average sparsity: {config.sparsity:.1%}")
        logger.info(f"Max expression value: {config.max_expression_value}")
        logger.info(f"Total size: {total_size_mb:.1f} MB")
        logger.info(f"Duration: {duration}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("="*50)


def main():
    parser = argparse.ArgumentParser(
        description="Download random subset of human scRNA-seq data for ML training"
    )
    parser.add_argument(
        "--cell-count",
        type=float,
        default=1e5,
        help="Number of cells to download (default: 1e6)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/scRNA",
        help="Output directory (default: data/scRNA)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of samples to process in each batch (default: 10)"
    )
    
    args = parser.parse_args()
    
    downloader = ScRNADownloader(output_dir=args.output_dir)
    downloader.download_and_process(
        target_cells=int(args.cell_count),
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main() 