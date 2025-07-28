#!/usr/bin/env python3
"""
Explore and understand the VCC (Virtual Cell Challenge) evaluation dataset.
This script inspects the training and validation data to understand:
- Data structure and format
- Gene perturbation information
- Control cells
- Evaluation requirements
"""

import os
import pandas as pd
import scanpy as sc
import numpy as np
from pathlib import Path
import json
import h5py
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


def explore_training_data(data_path: str):
    """Explore the training AnnData file."""
    print("=" * 80)
    print("EXPLORING TRAINING DATA")
    print("=" * 80)
    
    # Load AnnData
    print(f"\nLoading training data from: {data_path}")
    adata = sc.read_h5ad(data_path)
    
    # Basic information
    print(f"\nData shape: {adata.shape}")
    print(f"  Cells (n_obs): {adata.n_obs:,}")
    print(f"  Genes (n_vars): {adata.n_vars:,}")
    
    # Explore obs (cell metadata)
    print("\n--- Cell Metadata (obs) ---")
    print(f"Columns: {list(adata.obs.columns)}")
    print(f"\nFirst 5 rows:")
    print(adata.obs.head())
    
    # Perturbation information
    print("\n--- Perturbation Information ---")
    print(f"Unique target genes: {adata.obs['target_gene'].nunique()}")
    print(f"Total perturbations (including controls): {len(adata.obs)}")
    
    # Control cells
    control_mask = adata.obs['target_gene'] == 'non-targeting'
    n_controls = control_mask.sum()
    print(f"\nControl cells (non-targeting): {n_controls:,} ({100*n_controls/len(adata):.1f}%)")
    
    # Target gene distribution
    target_counts = adata.obs['target_gene'].value_counts()
    print(f"\nTop 10 most frequent target genes:")
    print(target_counts.head(10))
    
    # Batch information
    print(f"\n--- Batch Information ---")
    print(f"Unique batches: {adata.obs['batch'].nunique()}")
    print(f"Batch distribution:")
    print(adata.obs['batch'].value_counts().head())
    
    # Guide information
    print(f"\n--- Guide Information ---")
    print(f"Unique guide IDs: {adata.obs['guide_id'].nunique()}")
    # Check if multiple guides per gene
    guides_per_gene = adata.obs.groupby('target_gene')['guide_id'].nunique()
    print(f"Average guides per target gene: {guides_per_gene.mean():.2f}")
    print(f"Max guides per target gene: {guides_per_gene.max()}")
    
    # Explore var (gene metadata)
    print("\n--- Gene Metadata (var) ---")
    print(f"Columns: {list(adata.var.columns)}")
    print(f"\nFirst 5 genes:")
    print(adata.var.head())
    
    # Expression data
    print("\n--- Expression Data ---")
    print(f"Data type: {type(adata.X)}")
    print(f"Sparse matrix: {hasattr(adata.X, 'sparse')}")
    
    # Calculate basic statistics
    if hasattr(adata.X, 'toarray'):
        # Sparse matrix
        expr_data = adata.X
        print(f"Sparsity: {1 - expr_data.nnz / (expr_data.shape[0] * expr_data.shape[1]):.2%}")
        print(f"Max value: {expr_data.max()}")
        
        # Sample some statistics
        sample_idx = np.random.choice(adata.n_obs, min(1000, adata.n_obs), replace=False)
        sample_data = expr_data[sample_idx].toarray()
        print(f"\nExpression statistics (1000 cell sample):")
        print(f"  Mean expression: {sample_data.mean():.2f}")
        print(f"  Median expression: {np.median(sample_data):.2f}")
        print(f"  95th percentile: {np.percentile(sample_data, 95):.2f}")
        print(f"  99th percentile: {np.percentile(sample_data, 99):.2f}")
    else:
        # Dense matrix
        print(f"Mean expression: {adata.X.mean():.2f}")
        print(f"Max expression: {adata.X.max()}")
    
    # UMI counts per cell
    print("\n--- UMI Counts ---")
    umi_counts = np.array(adata.X.sum(axis=1)).flatten()
    print(f"Mean UMI per cell: {umi_counts.mean():.0f}")
    print(f"Median UMI per cell: {np.median(umi_counts):.0f}")
    print(f"Min UMI per cell: {umi_counts.min():.0f}")
    print(f"Max UMI per cell: {umi_counts.max():.0f}")
    
    return adata


def explore_validation_data(data_path: str):
    """Explore the validation CSV file."""
    print("\n" + "=" * 80)
    print("EXPLORING VALIDATION DATA")
    print("=" * 80)
    
    # Load CSV
    print(f"\nLoading validation data from: {data_path}")
    val_df = pd.read_csv(data_path)
    
    print(f"\nValidation data shape: {val_df.shape}")
    print(f"Columns: {list(val_df.columns)}")
    
    print("\nFirst 10 rows:")
    print(val_df.head(10))
    
    print("\n--- Statistics ---")
    print(f"Total target genes to predict: {len(val_df)}")
    print(f"Total cells to predict: {val_df['n_cells'].sum():,}")
    print(f"\nCells per perturbation:")
    print(f"  Mean: {val_df['n_cells'].mean():.0f}")
    print(f"  Median: {val_df['n_cells'].median():.0f}")
    print(f"  Min: {val_df['n_cells'].min()}")
    print(f"  Max: {val_df['n_cells'].max()}")
    
    print(f"\nMedian UMI per cell:")
    print(f"  Mean: {val_df['median_umi_per_cell'].mean():.0f}")
    print(f"  Min: {val_df['median_umi_per_cell'].min():.0f}")
    print(f"  Max: {val_df['median_umi_per_cell'].max():.0f}")
    
    return val_df


def analyze_perturbation_effects(adata, n_examples: int = 5):
    """Analyze the effects of perturbations on gene expression."""
    print("\n" + "=" * 80)
    print("ANALYZING PERTURBATION EFFECTS")
    print("=" * 80)
    
    # Get control cells
    control_mask = adata.obs['target_gene'] == 'non-targeting'
    control_expr = adata[control_mask].X
    
    # Calculate mean control expression
    if hasattr(control_expr, 'toarray'):
        control_mean = np.array(control_expr.mean(axis=0)).flatten()
    else:
        control_mean = control_expr.mean(axis=0)
    
    print(f"\nControl cells: {control_mask.sum():,}")
    print(f"Mean control expression: {control_mean.mean():.2f}")
    
    # Analyze a few example perturbations
    perturbed_genes = [g for g in adata.obs['target_gene'].unique() 
                       if g != 'non-targeting'][:n_examples]
    
    for gene in perturbed_genes:
        print(f"\n--- Perturbation: {gene} ---")
        
        # Get perturbed cells
        pert_mask = adata.obs['target_gene'] == gene
        n_pert = pert_mask.sum()
        print(f"Number of cells: {n_pert}")
        
        if n_pert > 0:
            pert_expr = adata[pert_mask].X
            
            # Calculate mean perturbed expression
            if hasattr(pert_expr, 'toarray'):
                pert_mean = np.array(pert_expr.mean(axis=0)).flatten()
            else:
                pert_mean = pert_expr.mean(axis=0)
            
            # Find gene index
            if gene in adata.var.index:
                gene_idx = adata.var.index.get_loc(gene)
                print(f"Target gene expression:")
                print(f"  Control: {control_mean[gene_idx]:.2f}")
                print(f"  Perturbed: {pert_mean[gene_idx]:.2f}")
                print(f"  Fold change: {pert_mean[gene_idx] / (control_mean[gene_idx] + 1e-6):.2f}")
            
            # Find most affected genes
            log_fc = np.log2((pert_mean + 1) / (control_mean + 1))
            top_up = np.argsort(log_fc)[-5:][::-1]
            top_down = np.argsort(log_fc)[:5]
            
            print(f"\nTop upregulated genes:")
            for idx in top_up:
                print(f"  {adata.var.index[idx]}: {log_fc[idx]:.2f} log2FC")
            
            print(f"\nTop downregulated genes:")
            for idx in top_down:
                print(f"  {adata.var.index[idx]}: {log_fc[idx]:.2f} log2FC")


def check_gene_overlap(adata, val_df):
    """Check if validation target genes exist in training data."""
    print("\n" + "=" * 80)
    print("CHECKING GENE OVERLAP")
    print("=" * 80)
    
    # Get all genes in training data
    train_genes = set(adata.var.index)
    train_target_genes = set(adata.obs['target_gene'].unique())
    train_target_genes.discard('non-targeting')
    
    # Get validation target genes
    val_target_genes = set(val_df['target_gene'])
    
    print(f"\nTraining data:")
    print(f"  Total genes in expression matrix: {len(train_genes):,}")
    print(f"  Unique perturbed genes: {len(train_target_genes)}")
    
    print(f"\nValidation data:")
    print(f"  Target genes to predict: {len(val_target_genes)}")
    
    # Check overlap
    in_training = val_target_genes.intersection(train_target_genes)
    not_in_training = val_target_genes - train_target_genes
    
    print(f"\nOverlap analysis:")
    print(f"  Validation genes in training perturbations: {len(in_training)} ({100*len(in_training)/len(val_target_genes):.1f}%)")
    print(f"  Validation genes NOT in training perturbations: {len(not_in_training)} ({100*len(not_in_training)/len(val_target_genes):.1f}%)")
    
    if not_in_training:
        print(f"\nValidation genes not perturbed in training:")
        for gene in sorted(not_in_training)[:10]:
            if gene in train_genes:
                print(f"  {gene} (exists in expression matrix)")
            else:
                print(f"  {gene} (NOT in expression matrix!)")
    
    # Check if all validation genes exist in expression matrix
    not_in_expr = val_target_genes - train_genes
    if not_in_expr:
        print(f"\n⚠️  WARNING: {len(not_in_expr)} validation genes not in expression matrix!")
        print(f"Missing genes: {sorted(not_in_expr)}")


def save_analysis_summary(adata, val_df, output_path: str):
    """Save a summary of the analysis."""
    summary = {
        'training_data': {
            'n_cells': int(adata.n_obs),
            'n_genes': int(adata.n_vars),
            'n_control_cells': int((adata.obs['target_gene'] == 'non-targeting').sum()),
            'n_perturbed_genes': int(adata.obs['target_gene'].nunique() - 1),
            'n_batches': int(adata.obs['batch'].nunique()),
            'sparsity': float(1 - adata.X.nnz / (adata.X.shape[0] * adata.X.shape[1]))
                       if hasattr(adata.X, 'nnz') else None
        },
        'validation_data': {
            'n_target_genes': len(val_df),
            'total_cells_to_predict': int(val_df['n_cells'].sum()),
            'mean_cells_per_gene': float(val_df['n_cells'].mean()),
            'mean_umi_per_cell': float(val_df['median_umi_per_cell'].mean())
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaved analysis summary to: {output_path}")


def main():
    # VCC data path
    vcc_path = Path("/workspace/vcc/data/vcc_data")
    
    # Check if path exists
    if not vcc_path.exists():
        print(f"Error: VCC data path not found: {vcc_path}")
        return
    
    # List files in directory
    print(f"Files in {vcc_path}:")
    for f in sorted(vcc_path.iterdir()):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name} ({size_mb:.1f} MB)")
    
    # File paths
    train_path = vcc_path / "adata_Training.h5ad"
    val_path = vcc_path / "pert_counts_Validation.csv"
    
    # Check if files exist
    if not train_path.exists():
        print(f"\nError: Training data not found: {train_path}")
        return
    
    if not val_path.exists():
        print(f"\nError: Validation data not found: {val_path}")
        return
    
    # Explore training data
    adata = explore_training_data(str(train_path))
    
    # Explore validation data
    val_df = explore_validation_data(str(val_path))
    
    # Analyze perturbation effects
    analyze_perturbation_effects(adata, n_examples=5)
    
    # Check gene overlap
    check_gene_overlap(adata, val_df)
    
    # Save summary
    save_analysis_summary(adata, val_df, "vcc_data_summary.json")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main() 