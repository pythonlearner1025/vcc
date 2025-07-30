#!/usr/bin/env python3
"""Extract all perturbed genes from VCC dataset for HVG whitelisting."""

import scanpy as sc
import argparse
from pathlib import Path

def extract_vcc_genes(adata_path: str, output_path: str):
    """Extract all perturbed genes from VCC dataset."""
    print(f"Loading VCC data from {adata_path}...")
    adata = sc.read_h5ad(adata_path)
    
    print(f"Dataset shape: {adata.shape[0]} cells x {adata.shape[1]} genes")
    
    # Get all unique perturbed genes (excluding non-targeting)
    all_target_genes = adata.obs['target_gene'].unique()
    perturbed_genes = [g for g in all_target_genes if g != 'non-targeting']
    
    print(f"Found {len(perturbed_genes)} unique perturbed genes")
    print(f"First 10 genes: {perturbed_genes[:10]}")
    
    # Convert gene names to Ensembl IDs
    gene_name_to_id = dict(zip(adata.var.index, adata.var['gene_id']))
    
    # Create whitelist with Ensembl IDs
    whitelist_genes = []
    missing_genes = []
    
    for gene_name in perturbed_genes:
        if gene_name in gene_name_to_id:
            ensembl_id = gene_name_to_id[gene_name]
            whitelist_genes.append(ensembl_id)
        else:
            missing_genes.append(gene_name)
    
    print(f"Successfully mapped {len(whitelist_genes)} genes to Ensembl IDs")
    if missing_genes:
        print(f"Warning: {len(missing_genes)} genes could not be mapped:")
        for gene in missing_genes:
            print(f"  - {gene}")
    
    # Save whitelist
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for gene_id in sorted(whitelist_genes):
            f.write(f"{gene_id}\n")
    
    print(f"Saved {len(whitelist_genes)} Ensembl IDs to {output_path}")
    return whitelist_genes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract perturbed genes from VCC dataset")
    parser.add_argument("--vcc_path", default="/vcc_data/adata_Training.h5ad", help="Path to VCC dataset")
    parser.add_argument("--output", default="data/vcc_perturbed_genes.txt", help="Output whitelist file")
    
    args = parser.parse_args()
    extract_vcc_genes(args.vcc_path, args.output)
