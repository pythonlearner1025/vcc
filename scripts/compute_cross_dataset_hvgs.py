#!/usr/bin/env python3
"""
Compute HVGs across both scRNA and VCC datasets for consistent gene indices.

This ensures that pretraining and fine-tuning use the same gene ordering,
which is critical for transfer learning to work properly.
"""

import json
import numpy as np
import scanpy as sc
import h5py
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Tuple, Set
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_scrna_gene_stats(data_dir: str) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, int]:
    """Load gene statistics from preprocessed scRNA data."""
    logger.info(f"Loading scRNA data from {data_dir}")
    
    # Load gene list from config
    config_path = Path(data_dir).parent / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    gene_list = config['gene_list']
    n_genes = len(gene_list)
    
    # Initialize statistics
    gene_means = np.zeros(n_genes)
    gene_vars = np.zeros(n_genes)
    gene_counts = np.zeros(n_genes)
    total_cells = 0
    
    # Process each batch
    batch_files = sorted(Path(data_dir).glob("batch_*.h5"))
    logger.info(f"Processing {len(batch_files)} scRNA batch files...")
    
    for batch_file in tqdm(batch_files, desc="Processing scRNA batches"):
        with h5py.File(batch_file, 'r') as f:
            X = f['X'][:]
            n_cells = X.shape[0]
            
            # Update statistics
            gene_means += X.sum(axis=0)
            gene_counts += (X > 0).sum(axis=0)
            total_cells += n_cells
    
    # Compute means
    gene_means = gene_means / total_cells
    
    # Second pass for variance
    for batch_file in tqdm(batch_files, desc="Computing variances"):
        with h5py.File(batch_file, 'r') as f:
            X = f['X'][:]
            gene_vars += ((X - gene_means) ** 2).sum(axis=0)
    
    gene_vars = gene_vars / total_cells
    
    logger.info(f"Loaded {n_genes} genes from {total_cells:,} scRNA cells")
    return gene_list, gene_means, gene_vars, gene_counts, total_cells


def load_vcc_gene_stats(vcc_path: str) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, int, Set[str]]:
    """Load gene statistics and perturbed genes from VCC data."""
    logger.info(f"Loading VCC data from {vcc_path}")
    
    adata = sc.read_h5ad(vcc_path)
    
    # Use Ensembl IDs for matching with scRNA data
    if 'gene_id' in adata.var.columns:
        gene_list = adata.var['gene_id'].tolist()
        logger.info(f"Using Ensembl IDs from 'gene_id' column for gene matching")
        
        # Create mapping from gene symbols to Ensembl IDs for perturbed genes
        symbol_to_ensembl = dict(zip(adata.var.index, adata.var['gene_id']))
    else:
        # Fallback to gene symbols if no Ensembl IDs
        gene_list = adata.var.index.tolist()
        symbol_to_ensembl = {}
        logger.warning("No Ensembl IDs found in VCC data, using gene symbols")
    
    # Get perturbed genes (these are gene symbols)
    target_genes = adata.obs['target_gene'].unique()
    perturbed_gene_symbols = set(g for g in target_genes if g != 'non-targeting')
    
    # Convert perturbed gene symbols to Ensembl IDs
    perturbed_genes = set()
    for symbol in perturbed_gene_symbols:
        if symbol in symbol_to_ensembl:
            perturbed_genes.add(symbol_to_ensembl[symbol])
        else:
            logger.warning(f"Perturbed gene {symbol} not found in gene list")
    
    logger.info(f"Found {len(perturbed_gene_symbols)} perturbed gene symbols, mapped to {len(perturbed_genes)} Ensembl IDs")
    
    # Convert to dense for statistics
    if hasattr(adata.X, 'toarray'):
        X = adata.X.toarray()
    else:
        X = adata.X.copy()
    
    # Compute statistics
    gene_means = X.mean(axis=0)
    gene_vars = X.var(axis=0)
    gene_counts = (X > 0).sum(axis=0)
    total_cells = len(adata)
    
    logger.info(f"Loaded {len(gene_list)} genes from {total_cells:,} VCC cells")
    return gene_list, gene_means, gene_vars, gene_counts, total_cells, perturbed_genes


def compute_cross_dataset_hvgs(
    scrna_stats: Tuple,
    vcc_stats: Tuple,
    vcc_data_path: str,  # Add this parameter to load symbol mapping
    n_hvgs: int = 2000,
    weight_scrna: float = 0.5,
    weight_vcc: float = 0.5,
    min_cells_pct: float = 0.01
) -> Tuple[List[str], Dict[str, int], Dict, Dict[str, str]]:
    """
    Compute HVGs across both datasets with proper weighting.
    
    Strategy:
    1. Find gene intersection between datasets
    2. Include all perturbed genes from VCC (critical for fine-tuning)
    3. Compute weighted dispersion across both datasets
    4. Select top HVGs to fill remaining slots
    
    Returns:
        selected_genes_sorted: List of selected Ensembl IDs
        gene_to_hvg_idx: Mapping from Ensembl ID to HVG index
        hvg_stats: Statistics dictionary
        ensembl_to_symbol: Mapping from Ensembl ID to gene symbol
    """
    # Unpack stats
    scrna_genes, scrna_means, scrna_vars, scrna_counts, scrna_cells = scrna_stats[:5]
    vcc_genes, vcc_means, vcc_vars, vcc_counts, vcc_cells, perturbed_genes = vcc_stats
    
    # Load VCC data to get symbol mapping
    logger.info("Loading VCC gene symbol mapping...")
    import scanpy as sc
    adata = sc.read_h5ad(vcc_data_path)
    ensembl_to_symbol = dict(zip(adata.var['gene_id'], adata.var.index))
    
    # Create gene mappings
    scrna_gene_to_idx = {g: i for i, g in enumerate(scrna_genes)}
    vcc_gene_to_idx = {g: i for i, g in enumerate(vcc_genes)}
    
    # Find gene intersection
    common_genes = set(scrna_genes) & set(vcc_genes)
    logger.info(f"Found {len(common_genes)} genes common to both datasets")
    
    # Include all perturbed genes that exist in both datasets
    perturbed_in_common = perturbed_genes & common_genes
    logger.info(f"{len(perturbed_in_common)} perturbed genes found in common genes")
    
    # Compute weighted statistics for common genes
    combined_stats = []
    
    for gene in common_genes:
        scrna_idx = scrna_gene_to_idx[gene]
        vcc_idx = vcc_gene_to_idx[gene]
        
        # Weighted mean and variance
        weighted_mean = (weight_scrna * scrna_means[scrna_idx] * scrna_cells + 
                        weight_vcc * vcc_means[vcc_idx] * vcc_cells) / (weight_scrna * scrna_cells + weight_vcc * vcc_cells)
        
        weighted_var = (weight_scrna * scrna_vars[scrna_idx] * scrna_cells + 
                       weight_vcc * vcc_vars[vcc_idx] * vcc_cells) / (weight_scrna * scrna_cells + weight_vcc * vcc_cells)
        
        # Total expression percentage
        scrna_pct = scrna_counts[scrna_idx] / scrna_cells
        vcc_pct = vcc_counts[vcc_idx] / vcc_cells
        total_pct = (weight_scrna * scrna_pct + weight_vcc * vcc_pct)
        
        # Compute dispersion
        dispersion = weighted_var / weighted_mean if weighted_mean > 0 else 0
        
        combined_stats.append({
            'gene': gene,
            'mean': weighted_mean,
            'var': weighted_var,
            'dispersion': dispersion,
            'pct_cells': total_pct * 100,
            'is_perturbed': gene in perturbed_genes
        })
    
    # Filter by minimum cell percentage
    min_pct = min_cells_pct
    filtered_stats = [s for s in combined_stats if s['pct_cells'] >= min_pct]
    logger.info(f"{len(filtered_stats)} genes pass minimum cell filter ({min_pct}%)")
    
    # Select HVGs
    # First, include all perturbed genes
    selected_genes = []
    selected_set = set()
    
    for stat in filtered_stats:
        if stat['is_perturbed']:
            selected_genes.append(stat['gene'])
            selected_set.add(stat['gene'])
    
    logger.info(f"Included {len(selected_genes)} perturbed genes")
    
    # Then add top HVGs by dispersion
    remaining_slots = n_hvgs - len(selected_genes)
    if remaining_slots > 0:
        # Sort by dispersion
        non_perturbed_stats = [s for s in filtered_stats if not s['is_perturbed']]
        non_perturbed_stats.sort(key=lambda x: x['dispersion'], reverse=True)
        
        for stat in non_perturbed_stats[:remaining_slots]:
            selected_genes.append(stat['gene'])
            selected_set.add(stat['gene'])
    
    logger.info(f"Selected {len(selected_genes)} total HVGs")
    
    # Create mappings (using consistent ordering)
    selected_genes_sorted = sorted(selected_genes)
    gene_to_hvg_idx = {gene: idx for idx, gene in enumerate(selected_genes_sorted)}
    
    # Compute additional statistics
    stats_dict = {g['gene']: g for g in combined_stats}
    hvg_stats = {
        'n_hvgs': len(selected_genes_sorted),
        'n_perturbed_included': len([g for g in selected_genes_sorted if g in perturbed_genes]),
        'n_common_genes': len(common_genes),
        'n_scrna_only': len(set(scrna_genes) - set(vcc_genes)),
        'n_vcc_only': len(set(vcc_genes) - set(scrna_genes)),
        'mean_dispersion': np.mean([stats_dict[g]['dispersion'] for g in selected_genes_sorted]) if selected_genes_sorted else 0,
        'mean_pct_cells': np.mean([stats_dict[g]['pct_cells'] for g in selected_genes_sorted]) if selected_genes_sorted else 0,
    }
    
    return selected_genes_sorted, gene_to_hvg_idx, hvg_stats, ensembl_to_symbol


def save_cross_dataset_hvgs(
    hvg_genes: List[str],
    gene_to_hvg_idx: Dict[str, int],
    stats: Dict,
    scrna_genes: List[str],
    vcc_genes: List[str],
    ensembl_to_symbol: Dict[str, str],
    output_dir: str
):
    """Save cross-dataset HVG information for both pretraining and fine-tuning."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create mappings from original indices to HVG indices
    scrna_gene_to_idx = {g: i for i, g in enumerate(scrna_genes)}
    vcc_gene_to_idx = {g: i for i, g in enumerate(vcc_genes)}
    
    # Mapping from scRNA gene indices to HVG indices
    scrna_to_hvg = {}
    for gene, hvg_idx in gene_to_hvg_idx.items():
        if gene in scrna_gene_to_idx:
            scrna_to_hvg[scrna_gene_to_idx[gene]] = hvg_idx
    
    # Mapping from VCC gene indices to HVG indices
    vcc_to_hvg = {}
    for gene, hvg_idx in gene_to_hvg_idx.items():
        if gene in vcc_gene_to_idx:
            vcc_to_hvg[vcc_gene_to_idx[gene]] = hvg_idx
    
    # Create list of HVG names (gene symbols for VCC compatibility)
    hvg_names = []
    for ensembl_id in hvg_genes:
        if ensembl_id in ensembl_to_symbol:
            hvg_names.append(ensembl_to_symbol[ensembl_id])
        else:
            # Fallback to Ensembl ID if no symbol available
            hvg_names.append(ensembl_id)
    
    # Create gene name to HVG index mapping (using symbols)
    gene_name_to_hvg_idx = {name: idx for idx, name in enumerate(hvg_names)}
    
    # Main HVG info file
    hvg_info = {
        'hvg_names': hvg_names,  # Gene symbols for VCC
        'hvg_ensembl_ids': hvg_genes,  # Ensembl IDs
        'gene_name_to_hvg_idx': gene_name_to_hvg_idx,  # Symbol to HVG index
        'ensembl_to_hvg_idx': gene_to_hvg_idx,  # Ensembl ID to HVG index
        'n_hvgs': len(hvg_genes),
        'statistics': stats,
        'created_from': 'cross_dataset_analysis',
        'scrna_to_hvg': scrna_to_hvg,  # For pretraining
        'vcc_to_hvg': vcc_to_hvg,      # For fine-tuning
    }
    
    # Save main file
    main_file = output_dir / 'cross_dataset_hvg_info.json'
    with open(main_file, 'w') as f:
        json.dump(hvg_info, f, indent=2)
    logger.info(f"Saved cross-dataset HVG info to {main_file}")
    
    # Save VCC-compatible hvg_info.json
    # For VCC, we need to map from original VCC indices (which use symbols) to HVG indices
    import scanpy as sc
    vcc_path = Path(output_dir) / "adata_Training.h5ad"
    if not vcc_path.exists():
        # Try parent directory
        vcc_path = Path(output_dir).parent / "vcc_data" / "adata_Training.h5ad"
    adata = sc.read_h5ad(vcc_path)
    vcc_symbol_to_idx = {g: i for i, g in enumerate(adata.var.index)}
    
    # Create mapping from VCC symbol indices to HVG indices
    vcc_symbol_to_hvg = {}
    for symbol, hvg_idx in gene_name_to_hvg_idx.items():
        if symbol in vcc_symbol_to_idx:
            vcc_symbol_to_hvg[vcc_symbol_to_idx[symbol]] = hvg_idx
    
    vcc_hvg_info = {
        'hvg_indices': sorted(vcc_symbol_to_hvg.keys()),  # Original VCC indices that map to HVGs
        'hvg_names': hvg_names,
        'gene_to_hvg_idx': {str(k): v for k, v in vcc_symbol_to_hvg.items()},
        'gene_name_to_hvg_idx': gene_name_to_hvg_idx,
        'total_genes': len(adata.var),
        'n_hvgs': len(hvg_genes),
        'statistics': stats
    }
    
    vcc_file = output_dir / 'hvg_info.json'
    with open(vcc_file, 'w') as f:
        json.dump(vcc_hvg_info, f, indent=2)
    logger.info(f"Saved VCC HVG info to {vcc_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("CROSS-DATASET HVG SELECTION COMPLETE")
    print("="*60)
    print(f"Total HVGs selected: {stats['n_hvgs']}")
    print(f"Perturbed genes included: {stats['n_perturbed_included']}")
    print(f"Genes common to both datasets: {stats['n_common_genes']}")
    print(f"Genes only in scRNA: {stats['n_scrna_only']}")
    print(f"Genes only in VCC: {stats['n_vcc_only']}")
    print(f"Mean dispersion of HVGs: {stats['mean_dispersion']:.2f}")
    print(f"Mean % cells expressing HVGs: {stats['mean_pct_cells']:.2f}%")
    print("\nThese HVGs will be used for BOTH pretraining and fine-tuning,")
    print("ensuring consistent gene indices across phases.")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Compute HVGs across scRNA and VCC datasets')
    parser.add_argument('--scrna-dir', type=str, default='data/scRNA_1e5/processed',
                        help='Directory containing preprocessed scRNA data')
    parser.add_argument('--vcc-path', type=str, default='data/vcc_data/adata_Training.h5ad',
                        help='Path to VCC training data')
    parser.add_argument('--output-dir', type=str, default='data/vcc_data',
                        help='Output directory for HVG info')
    parser.add_argument('--n-hvgs', type=int, default=2000,
                        help='Number of HVGs to select')
    parser.add_argument('--weight-scrna', type=float, default=0.5,
                        help='Weight for scRNA data (0-1)')
    parser.add_argument('--weight-vcc', type=float, default=0.5,
                        help='Weight for VCC data (0-1)')
    parser.add_argument('--min-cells-pct', type=float, default=0.01,
                        help='Minimum percentage of cells expressing a gene')
    
    args = parser.parse_args()
    
    # Load data statistics
    logger.info("Loading gene statistics from both datasets...")
    scrna_stats = load_scrna_gene_stats(args.scrna_dir)
    vcc_stats = load_vcc_gene_stats(args.vcc_path)
    
    # Compute cross-dataset HVGs
    logger.info("Computing cross-dataset HVGs...")
    hvg_genes, gene_to_hvg_idx, stats, ensembl_to_symbol = compute_cross_dataset_hvgs(
        scrna_stats, vcc_stats,
        vcc_data_path=args.vcc_path, # Pass vcc_data_path
        n_hvgs=args.n_hvgs,
        weight_scrna=args.weight_scrna,
        weight_vcc=args.weight_vcc,
        min_cells_pct=args.min_cells_pct
    )
    
    # Save results
    logger.info("Saving results...")
    save_cross_dataset_hvgs(
        hvg_genes, gene_to_hvg_idx, stats,
        scrna_stats[0], vcc_stats[0],  # gene lists
        ensembl_to_symbol, # Pass ensembl_to_symbol
        args.output_dir
    )


if __name__ == "__main__":
    main() 