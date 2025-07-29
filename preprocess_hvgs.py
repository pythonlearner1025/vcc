#!/usr/bin/env python3
"""
Preprocess and analyze highly variable genes (HVGs) from the downloaded dataset.
This script can be run independently to compute HVGs before training.

Usage:
    python preprocess_hvgs.py --data-dir data/scRNA/processed --n-hvgs 2000
"""

import argparse
import json
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import logging
import scanpy as sc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_gene_statistics(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Compute gene statistics across all batches.
    
    Returns:
        gene_means: Mean expression per gene
        gene_vars: Variance per gene
        gene_counts: Number of cells expressing each gene
        total_cells: Total number of cells
    """
    # Load gene list
    with open(Path(data_dir).parent / "config.json", 'r') as f:
        config = json.load(f)
    n_genes = len(config['gene_list']) # Gene Count, so 36k?
    logger.info(f"Total genes: {n_genes}")
    
    # Initialize statistics
    gene_means = np.zeros(n_genes)
    gene_m2 = np.zeros(n_genes)  # Sum of squared differences (for Welford's algorithm)
    gene_counts = np.zeros(n_genes, dtype=int)
    total_cells = 0

    # Process each batch
    batch_files = sorted(Path(data_dir).glob("batch_*.h5"))
    logger.info(f"Processing {len(batch_files)} batch files...")
    
    for batch_file in tqdm(batch_files, desc="Computing statistics"):
        with h5py.File(batch_file, 'r') as f:
            expression = f['X'][:]  # Load expression data (cells x genes)
            batch_n = expression.shape[0]
            
            # Compute batch statistics
            batch_mean = expression.mean(axis=0)
            batch_m2 = ((expression - batch_mean) ** 2).sum(axis=0)
            batch_nonzero = (expression > 0).sum(axis=0)
            
            # Parallel Welford's algorithm - combine batch statistics
            if total_cells == 0:
                # First batch
                gene_means = batch_mean.copy()
                gene_m2 = batch_m2.copy()
            else:
                # Combine with existing statistics
                delta = batch_mean - gene_means
                gene_means = (gene_means * total_cells + batch_mean * batch_n) / (total_cells + batch_n)
                gene_m2 = gene_m2 + batch_m2 + delta**2 * total_cells * batch_n / (total_cells + batch_n)
            
            gene_counts += batch_nonzero
            total_cells += batch_n

    # Finalize variance
    gene_vars = gene_m2 / (total_cells - 1) if total_cells > 1 else np.zeros(n_genes)

    return gene_means, gene_vars, gene_counts, total_cells


def compute_vcc_gene_statistics(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, List[str]]:
    """
    Compute gene statistics from VCC data.
    
    Returns:
        gene_means: Mean expression per gene
        gene_vars: Variance per gene
        gene_counts: Number of cells expressing each gene
        total_cells: Total number of cells
        gene_list: List of gene names
    """
    logger.info(f"Loading VCC data from {data_path}")
    adata = sc.read_h5ad(data_path)
    
    # Convert to dense array if sparse
    if hasattr(adata.X, 'toarray'):
        expression = adata.X.toarray()
    else:
        expression = adata.X.copy()
    
    # Compute statistics
    gene_means = expression.mean(axis=0)
    gene_vars = expression.var(axis=0)
    gene_counts = (expression > 0).sum(axis=0)
    total_cells = expression.shape[0]
    gene_list = adata.var.index.tolist()
    
    logger.info(f"Computed statistics for {total_cells} cells and {len(gene_list)} genes")
    
    return gene_means, gene_vars, gene_counts, total_cells, gene_list


def select_hvgs(gene_means: np.ndarray, gene_vars: np.ndarray, 
                gene_counts: np.ndarray, total_cells: int,
                n_hvgs: int = 2000, min_cells_pct: float = 0.1) -> np.ndarray:
    """
    Select highly variable genes using dispersion (variance/mean ratio).
    
    Args:
        gene_means: Mean expression per gene
        gene_vars: Variance per gene
        gene_counts: Number of cells expressing each gene
        total_cells: Total number of cells
        n_hvgs: Number of HVGs to select
        min_cells_pct: Minimum percentage of cells expressing a gene
        
    Returns:
        hvg_indices: Indices of selected HVGs
    """
    # Compute dispersion
    dispersions = np.divide(gene_vars, gene_means, 
                           out=np.zeros_like(gene_vars), 
                           where=gene_means > 0)
    
    # Filter genes by minimum cell percentage
    min_cells = int(min_cells_pct / 100 * total_cells)
    expressed_mask = gene_counts >= min_cells
    
    # Set dispersion to -1 for non-expressed genes
    dispersions[~expressed_mask] = -1
    
    # Get top HVGs by dispersion
    hvg_indices = np.argsort(dispersions)[-n_hvgs:][::-1]
    
    logger.info(f"Selected {n_hvgs} HVGs from {expressed_mask.sum()} expressed genes")
    logger.info(f"Dispersion range: {dispersions[hvg_indices].min():.2f} - {dispersions[hvg_indices].max():.2f}")
    
    return hvg_indices


def analyze_hvgs(hvg_indices: np.ndarray, gene_means: np.ndarray, 
                 gene_vars: np.ndarray, gene_counts: np.ndarray, 
                 total_cells: int, gene_list: List[str], 
                 output_dir: Path) -> Dict:
    """
    Analyze and visualize HVG statistics.
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute statistics
    dispersions = np.divide(gene_vars, gene_means, 
                           out=np.zeros_like(gene_vars), 
                           where=gene_means > 0)
    
    hvg_dispersions = dispersions[hvg_indices]
    hvg_means = gene_means[hvg_indices]
    hvg_pct_cells = 100 * gene_counts[hvg_indices] / total_cells
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Dispersion vs Mean (all genes + HVGs highlighted)
    ax = axes[0, 0]
    # Plot all genes
    mask = gene_means > 0
    ax.scatter(np.log1p(gene_means[mask]), dispersions[mask], 
               alpha=0.1, s=1, c='gray', label='All genes')
    # Highlight HVGs
    ax.scatter(np.log1p(hvg_means), hvg_dispersions, 
               alpha=0.8, s=10, c='red', label='HVGs')
    ax.set_xlabel('log(Mean expression + 1)')
    ax.set_ylabel('Dispersion (Var/Mean)')
    ax.set_title('Gene Dispersion vs Expression')
    ax.legend()
    
    # 2. HVG expression distribution
    ax = axes[0, 1]
    ax.hist(np.log1p(hvg_means), bins=50, alpha=0.7, color='blue')
    ax.set_xlabel('log(Mean expression + 1)')
    ax.set_ylabel('Count')
    ax.set_title('HVG Expression Distribution')
    
    # 3. Percentage of cells expressing HVGs
    ax = axes[1, 0]
    ax.hist(hvg_pct_cells, bins=50, alpha=0.7, color='green')
    ax.set_xlabel('Percentage of cells expressing gene')
    ax.set_ylabel('Count')
    ax.set_title('HVG Cell Expression Coverage')
    
    # 4. Top 20 HVGs
    ax = axes[1, 1]
    top_20_idx = hvg_indices[:20]
    top_20_names = [gene_list[i] for i in top_20_idx]
    top_20_disp = dispersions[top_20_idx]
    
    y_pos = np.arange(len(top_20_names))
    ax.barh(y_pos, top_20_disp, alpha=0.7, color='purple')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_20_names, fontsize=8)
    ax.set_xlabel('Dispersion')
    ax.set_title('Top 20 Highly Variable Genes')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'hvg_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Prepare statistics
    stats = {
        'total_cells_analyzed': int(total_cells),
        'total_genes': len(gene_list),
        'expressed_genes': int((gene_counts > 0).sum()),
        'n_hvgs': len(hvg_indices),
        'mean_dispersion': float(hvg_dispersions.mean()),
        'median_dispersion': float(np.median(hvg_dispersions)),
        'min_dispersion': float(hvg_dispersions.min()),
        'max_dispersion': float(hvg_dispersions.max()),
        'mean_expression': float(hvg_means.mean()),
        'median_pct_cells': float(np.median(hvg_pct_cells)),
        'top_10_hvgs': [gene_list[i] for i in hvg_indices[:10]],
        'top_10_dispersions': [float(dispersions[i]) for i in hvg_indices[:10]],
        'top_10_pct_cells': [float(100 * gene_counts[i] / total_cells) for i in hvg_indices[:10]]
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Preprocess highly variable genes')
    parser.add_argument('--data-dir', type=str, default='data/scRNA/processed',
                        help='Directory containing processed data')
    parser.add_argument('--n-hvgs', type=int, default=2000,
                        help='Number of HVGs to select')
    parser.add_argument('--min-cells-pct', type=float, default=0.1,
                        help='Minimum percentage of cells expressing a gene')
    parser.add_argument('--output-dir', type=str, default='data/scRNA',
                        help='Output directory for results')
    parser.add_argument('--force', action='store_true',
                        help='Force recomputation even if cache exists')
    parser.add_argument('--vcc-data-path', type=str, default=None,
                        help='Path to VCC data (adata_Training.h5ad)')
    parser.add_argument('--process-vcc', action='store_true',
                        help='Process VCC data for HVGs')
    
    args = parser.parse_args()
    
    # Process VCC data if requested
    if args.process_vcc or args.vcc_data_path:
        # Find VCC data path
        if args.vcc_data_path:
            vcc_path = Path(args.vcc_data_path)
        else:
            # Try to find VCC data automatically
            possible_paths = [
                Path("/workspace/vcc/data/vcc_data/adata_Training.h5ad"),
                Path("/data/vcc_data/adata_Training.h5ad"),
                Path("./data/vcc_data/adata_Training.h5ad"),
            ]
            vcc_path = None
            for p in possible_paths:
                if p.exists():
                    vcc_path = p
                    break
            
            if vcc_path is None:
                logger.error("Could not find VCC data. Please specify --vcc-data-path")
                return
        
        output_dir = vcc_path.parent
        cache_file = output_dir / 'hvg_info.json'
        
        # Check if already computed
        if cache_file.exists() and not args.force:
            logger.info(f"VCC HVG info already exists at {cache_file}")
            logger.info("Use --force to recompute")
            
            # Load and display summary
            with open(cache_file, 'r') as f:
                hvg_info = json.load(f)
            
            print("\nVCC HVG Summary:")
            print(f"  Number of HVGs: {len(hvg_info['hvg_indices'])}")
            print(f"  Top 5 HVGs: {hvg_info['hvg_names'][:5]}")
            return
        
        # Compute gene statistics for VCC
        logger.info("Computing gene statistics for VCC data...")
        gene_means, gene_vars, gene_counts, total_cells, gene_list = compute_vcc_gene_statistics(str(vcc_path))
        
        # Select HVGs
        logger.info(f"Selecting top {args.n_hvgs} highly variable genes from VCC data...")
        hvg_indices = select_hvgs(gene_means, gene_vars, gene_counts, total_cells, 
                                 args.n_hvgs, args.min_cells_pct)
        
        # Analyze HVGs
        logger.info("Analyzing VCC HVG statistics...")
        stats = analyze_hvgs(hvg_indices, gene_means, gene_vars, gene_counts, 
                            total_cells, gene_list, output_dir)
        
        # Create mappings
        hvg_names = [gene_list[i] for i in hvg_indices]
        gene_to_hvg_idx = {int(gene_idx): int(hvg_idx) 
                           for hvg_idx, gene_idx in enumerate(hvg_indices)}
        
        # Also create gene name to HVG index mapping
        gene_name_to_hvg_idx = {gene_list[gene_idx]: int(hvg_idx) 
                                for hvg_idx, gene_idx in enumerate(hvg_indices)}
        
        # Prepare final output
        hvg_info = {
            'hvg_indices': hvg_indices.tolist(),
            'hvg_names': hvg_names,
            'gene_to_hvg_idx': gene_to_hvg_idx,
            'gene_name_to_hvg_idx': gene_name_to_hvg_idx,
            'total_genes': len(gene_list),
            'statistics': stats
        }
        
        # Save results
        with open(cache_file, 'w') as f:
            json.dump(hvg_info, f, indent=2)
        
        logger.info(f"Saved VCC HVG info to {cache_file}")
        logger.info(f"Saved visualizations to {output_dir}")
        
        # Print summary
        print("\nVCC HVG Selection Summary:")
        print(f"  Total cells analyzed: {stats['total_cells_analyzed']:,}")
        print(f"  Total genes: {stats['total_genes']:,}")
        print(f"  Selected HVGs: {stats['n_hvgs']:,}")
        print(f"\nTop 10 VCC HVGs:")
        for i, (gene, disp, pct) in enumerate(zip(stats['top_10_hvgs'], 
                                                  stats['top_10_dispersions'],
                                                  stats['top_10_pct_cells'])):
            print(f"  {i+1:2d}. {gene:<15} (dispersion: {disp:6.2f}, cells: {pct:5.1f}%)")
        
        return
    
    # Original scRNA processing
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    cache_file = output_dir / 'hvg_info.json'
    
    # Check if already computed
    if cache_file.exists() and not args.force:
        logger.info(f"HVG info already exists at {cache_file}")
        logger.info("Use --force to recompute")
        
        # Load and display summary
        with open(cache_file, 'r') as f:
            hvg_info = json.load(f)
        
        print("\nHVG Summary:")
        print(f"  Number of HVGs: {len(hvg_info['hvg_indices'])}")
        print(f"  Top 5 HVGs: {hvg_info['hvg_names'][:5]}")
        print(f"  Dispersion range: {hvg_info['statistics']['min_dispersion']:.2f} - "
              f"{hvg_info['statistics']['max_dispersion']:.2f}")
        return
    
    # Load config
    with open(data_dir.parent / "config.json", 'r') as f:
        config = json.load(f)
    gene_list = config['gene_list']
    
    # Compute gene statistics
    logger.info("Computing gene statistics across all batches...")
    gene_means, gene_vars, gene_counts, total_cells = compute_gene_statistics(args.data_dir)
    
    # Select HVGs
    logger.info(f"Selecting top {args.n_hvgs} highly variable genes...")
    hvg_indices = select_hvgs(gene_means, gene_vars, gene_counts, total_cells, 
                             args.n_hvgs, args.min_cells_pct)
    
    # Analyze HVGs
    logger.info("Analyzing HVG statistics...")
    stats = analyze_hvgs(hvg_indices, gene_means, gene_vars, gene_counts, 
                        total_cells, gene_list, output_dir)
    
    # Create mappings
    hvg_names = [gene_list[i] for i in hvg_indices]
    gene_to_hvg_idx = {int(gene_idx): int(hvg_idx) 
                       for hvg_idx, gene_idx in enumerate(hvg_indices)}
    
    # Prepare final output
    hvg_info = {
        'hvg_indices': hvg_indices.tolist(),
        'hvg_names': hvg_names,
        'gene_to_hvg_idx': gene_to_hvg_idx,
        'statistics': stats
    }
    
    # Save results
    with open(cache_file, 'w') as f:
        json.dump(hvg_info, f, indent=2)
    
    logger.info(f"Saved HVG info to {cache_file}")
    logger.info(f"Saved visualizations to {output_dir}")
    
    # Print summary
    print("\nHVG Selection Summary:")
    print(f"  Total cells analyzed: {stats['total_cells_analyzed']:,}")
    print(f"  Total genes: {stats['total_genes']:,}")
    print(f"  Expressed genes: {stats['expressed_genes']:,}")
    print(f"  Selected HVGs: {stats['n_hvgs']:,}")
    print(f"\nTop 10 HVGs:")
    for i, (gene, disp, pct) in enumerate(zip(stats['top_10_hvgs'], 
                                              stats['top_10_dispersions'],
                                              stats['top_10_pct_cells'])):
        print(f"  {i+1:2d}. {gene:<15} (dispersion: {disp:6.2f}, cells: {pct:5.1f}%)")
    
    print(f"\nVisualization saved to: {output_dir / 'hvg_analysis.png'}")


if __name__ == "__main__":
    main() 