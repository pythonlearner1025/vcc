#!/usr/bin/env python3
"""
Compute combined HVGs by taking union of:
1. Top N highly variable genes from expression data
2. All perturbed genes from VCC data
"""

import json
import numpy as np
import scanpy as sc
from pathlib import Path
import argparse


def get_vcc_perturbed_genes(vcc_data_path: str):
    """Get all unique perturbed genes from VCC data."""
    print(f"Loading VCC data from {vcc_data_path}")
    adata = sc.read_h5ad(vcc_data_path)
    
    # Get unique target genes (excluding controls)
    target_genes = adata.obs['target_gene'].unique()
    perturbed_genes = [g for g in target_genes if g != 'non-targeting']
    
    print(f"Found {len(perturbed_genes)} unique perturbed genes in VCC data")
    return perturbed_genes


def compute_combined_hvgs(vcc_data_path: str, n_hvgs: int = 2000, output_dir: str = None):
    """
    Compute combined HVGs from VCC data.
    
    Strategy:
    1. Get all perturbed genes from VCC
    2. Compute HVGs from expression data
    3. Take union, prioritizing perturbed genes
    """
    if output_dir is None:
        output_dir = Path(vcc_data_path).parent
    else:
        output_dir = Path(output_dir)
    
    # Load VCC data
    print("Loading VCC data...")
    adata = sc.read_h5ad(vcc_data_path)
    gene_list = adata.var.index.tolist()
    
    # Get perturbed genes
    perturbed_genes = get_vcc_perturbed_genes(vcc_data_path)
    perturbed_gene_indices = []
    for gene in perturbed_genes:
        if gene in gene_list:
            perturbed_gene_indices.append(gene_list.index(gene))
    
    print(f"Found {len(perturbed_gene_indices)} perturbed genes in gene list")
    
    # Compute HVGs from expression data
    print("Computing highly variable genes...")
    
    # Convert to dense array if sparse
    if hasattr(adata.X, 'toarray'):
        expression = adata.X.toarray()
    else:
        expression = adata.X.copy()
    
    # Compute gene statistics
    gene_means = expression.mean(axis=0)
    gene_vars = expression.var(axis=0)
    gene_counts = (expression > 0).sum(axis=0)
    
    # Compute dispersion (variance/mean ratio)
    dispersions = np.divide(gene_vars, gene_means, 
                           out=np.zeros_like(gene_vars), 
                           where=gene_means > 0)
    
    # Filter genes by minimum cell percentage (0.1%)
    min_cells = int(0.001 * len(adata))
    expressed_mask = gene_counts >= min_cells
    
    # Set dispersion to -1 for non-expressed genes
    dispersions[~expressed_mask] = -1
    
    # Get top HVGs by dispersion
    hvg_indices = np.argsort(dispersions)[-n_hvgs:][::-1].tolist()
    hvg_indices = [idx for idx in hvg_indices if dispersions[idx] > 0]  # Remove non-expressed

    # Combine: union of perturbed genes and HVGs
    combined_indices = list(set(perturbed_gene_indices + hvg_indices))
    
    # If we have more than n_hvgs, we need to prioritize
    if len(combined_indices) > n_hvgs:
        print(f"Combined set has {len(combined_indices)} genes, limiting to {n_hvgs}")
        
        # First include all perturbed genes
        final_indices = perturbed_gene_indices.copy()
        
        # Then add HVGs until we reach n_hvgs
        remaining_slots = n_hvgs - len(final_indices)
        if remaining_slots > 0:
            # Add HVGs that aren't already included
            additional_hvgs = [idx for idx in hvg_indices if idx not in final_indices]
            
            # Sort by dispersion to get the most variable          
            additional_hvgs_sorted = sorted(additional_hvgs, 
                                          key=lambda x: dispersions[x], 
                                          reverse=True)
            final_indices.extend(additional_hvgs_sorted[:remaining_slots])
    else:
        final_indices = combined_indices
    
    # Sort indices for consistency
    final_indices = sorted(final_indices)
    
    # Create output
    hvg_names = [gene_list[i] for i in final_indices]
    gene_to_hvg_idx = {int(gene_idx): int(hvg_idx) 
                       for hvg_idx, gene_idx in enumerate(final_indices)}
    gene_name_to_hvg_idx = {gene_list[gene_idx]: int(hvg_idx) 
                            for hvg_idx, gene_idx in enumerate(final_indices)}
    
    # Statistics
    n_perturbed_in_hvg = len([g for g in perturbed_genes if g in hvg_names])
    n_natural_hvg = len([i for i in final_indices if i not in perturbed_gene_indices])
    
    hvg_info = {
        'hvg_indices': final_indices,
        'hvg_names': hvg_names,
        'gene_to_hvg_idx': gene_to_hvg_idx,
        'gene_name_to_hvg_idx': gene_name_to_hvg_idx,
        'total_genes': len(gene_list),
        'n_hvgs': len(final_indices),
        'n_perturbed_genes_included': n_perturbed_in_hvg,
        'n_natural_hvgs_included': n_natural_hvg,
        'perturbed_genes': perturbed_genes,
        'statistics': {
            'total_cells_analyzed': len(adata),
            'total_genes': len(gene_list),
            'n_hvgs': len(final_indices),
            'n_perturbed_genes': len(perturbed_genes),
            'n_perturbed_in_hvg': n_perturbed_in_hvg,
            'n_natural_hvg': n_natural_hvg
        }
    }
    
    # Save
    output_file = output_dir / 'hvg_info_combined.json'
    with open(output_file, 'w') as f:
        json.dump(hvg_info, f, indent=2)
    
    print(f"\nSaved combined HVG info to {output_file}")
    print(f"Total HVGs: {len(final_indices)}")
    print(f"  - Perturbed genes included: {n_perturbed_in_hvg}/{len(perturbed_genes)}")
    print(f"  - Natural HVGs included: {n_natural_hvg}")
    
    # Also update the main hvg_info.json to use combined version
    main_hvg_file = output_dir / 'hvg_info.json'
    with open(main_hvg_file, 'w') as f:
        json.dump(hvg_info, f, indent=2)
    print(f"Updated main HVG file: {main_hvg_file}")
    
    return hvg_info


def main():
    parser = argparse.ArgumentParser(description='Compute combined HVGs from VCC data')
    parser.add_argument('--vcc-data-path', type=str, 
                        default='/workspace/vcc/data/vcc_data/adata_Training.h5ad',
                        help='Path to VCC data')
    parser.add_argument('--n-hvgs', type=int, default=2000,
                        help='Number of HVGs to select')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: same as VCC data)')
    
    args = parser.parse_args()
    
    compute_combined_hvgs(args.vcc_data_path, args.n_hvgs, args.output_dir)


if __name__ == "__main__":
    main()