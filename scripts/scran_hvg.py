#!/usr/bin/env python3
"""
Compute HVGs across both scRNA and VCC datasets for consistent gene indices
using the robust scran method.

This ensures that pretraining and fine-tuning use the same gene ordering,
which is critical for transfer learning to work properly.
"""

import numpy as np
import scanpy as sc
import anndata
from pathlib import Path
import argparse
import logging
from typing import List, Set
import h5py
import glob

# rpy2 imports for using R in Python
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.conversion import localconverter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Activate rpy2 conversion

def compute_hvg_with_scran(adata: anndata.AnnData, n_hvgs: int = 2000) -> List[str]:
    """
    Computes Highly Variable Genes (HVGs) using the scran R package.

    This function models the gene-specific variance, decomposing it into technical
    and biological components, and returns the top genes with the highest
    biological variance.

    Args:
        adata: An AnnData object with raw counts in adata.X.
        n_hvgs: The number of highly variable genes to return.

    Returns:
        A list of gene names for the selected HVGs.
    """
    logger.info("Converting AnnData object to R SingleCellExperiment...")

    # Import R packages
    scran = importr('scran')
    scuttle = importr('scuttle')
    sce = importr('SingleCellExperiment')

    # Use modern rpy2 conversion context
    with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
        try:
            # Convert data to R format
            # R expects genes x cells, so transpose the matrix
            counts_matrix = adata.X.T
            gene_names = adata.var.index.tolist()
            cell_names = adata.obs.index.tolist()
            
            # Convert sparse matrix to dense numpy array if needed
            if hasattr(counts_matrix, 'toarray'):
                counts_matrix = counts_matrix.toarray()
            
            # Convert to R objects using the simpler approach
            r_counts = ro.r.matrix(ro.FloatVector(counts_matrix.flatten()), nrow=counts_matrix.shape[0])
            r_gene_names = ro.StrVector(gene_names)
            r_cell_names = ro.StrVector(cell_names)

            # Create a SingleCellExperiment object in R
            r_sce = sce.SingleCellExperiment(
                assays=ro.ListVector({"counts": r_counts})
            )
            
            # Set row and column names
            ro.r("rownames")(r_sce, r_gene_names)
            ro.r("colnames")(r_sce, r_cell_names)

            logger.info(f"Normalizing counts and modeling variance for {n_hvgs} HVGs...")

            # --- scran HVG Selection Workflow ---
            # 1. Normalize counts. scuttle::logNormCounts is a standard scran workflow step.
            r_sce = scuttle.logNormCounts(r_sce)

            # 2. Model the mean-variance relationship[cite: 2131].
            # This identifies the biological component of variance for each gene[cite: 2179].
            var_stats = scran.modelGeneVar(r_sce)

            # 3. Get the top N highly variable genes[cite: 1782].
            # We specify n to get the desired number of HVGs[cite: 1803].
            top_hvgs_r = scran.getTopHVGs(var_stats, n=n_hvgs)

            # Convert the R character vector back to a Python list
            top_hvgs_py = list(top_hvgs_r)

            logger.info(f"Successfully computed {len(top_hvgs_py)} HVGs using scran.")
            return top_hvgs_py
            
        except Exception as e:
            logger.error(f"Error during R conversion: {e}")
            raise


def load_scrna_from_batches(scrna_dir: str) -> anndata.AnnData:
    """Load scRNA data from batch files created by download.py."""
    import h5py
    import glob
    
    batch_files = sorted(glob.glob(str(Path(scrna_dir) / "batch_*.h5")))
    if not batch_files:
        raise FileNotFoundError(f"No batch files found in {scrna_dir}")
    
    logger.info(f"Found {len(batch_files)} batch files")
    
    # Load first batch to get structure
    with h5py.File(batch_files[0], 'r') as f:
        genes = f['genes'][:].astype(str)
        
    # Collect all data
    all_X = []
    all_cells = []
    
    for i, batch_file in enumerate(batch_files):
        with h5py.File(batch_file, 'r') as f:
            X = f['X'][:]
            cells = f['cells'][:].astype(str)
            all_X.append(X)
            all_cells.extend(cells)
    
    # Combine all batches
    X_combined = np.vstack(all_X)
    
    # Create AnnData object
    adata = anndata.AnnData(X=X_combined)
    adata.obs_names = all_cells
    adata.var_names = genes
    
    logger.info(f"Loaded {adata.shape[0]:,} cells and {adata.shape[1]} genes from scRNA batches")
    return adata


def main(scrna_dir: str, vcc_path: str, output_path: str, n_hvgs: int = 2000, max_cells: int = None):
    """
    Load data, combine, compute HVGs with scran, and save the gene list.
    
    Args:
        scrna_dir: Directory containing scRNA data
        vcc_path: Path to VCC h5ad file
        output_path: Directory to save output
        n_hvgs: Number of HVGs to compute
        max_cells: Maximum number of cells to use (for testing)
    """
    # --- 1. Load Data ---
    logger.info(f"Loading scRNA data from {scrna_dir}")
    scrna_adata = load_scrna_from_batches(scrna_dir)
    
    logger.info(f"Loading VCC data from {vcc_path}")
    vcc_adata = sc.read_h5ad(vcc_path)
    
    # Ensure both datasets use the same gene identifier system
    if 'gene_id' in vcc_adata.var.columns:
        vcc_adata.var.index = vcc_adata.var['gene_id']
        logger.info("Using Ensembl IDs from 'gene_id' as index for VCC data.")
        
    # --- 2. Combine Datasets for Joint HVG Selection ---
    # Ensure gene lists are identical before combining
    common_genes = scrna_adata.var.index.intersection(vcc_adata.var.index)
    logger.info(f"Found {len(common_genes)} common genes between datasets.")
    
    scrna_adata = scrna_adata[:, common_genes].copy()
    vcc_adata = vcc_adata[:, common_genes].copy()
    
    # Print column info
    logger.info("scRNA data columns:")
    logger.info(scrna_adata.var.columns.tolist())
    logger.info("\nVCC data columns:")
    logger.info(vcc_adata.var.columns.tolist())
    
    # Subsample if requested (for testing purposes)
    if max_cells is not None:
        if scrna_adata.shape[0] > max_cells // 2:
            scrna_adata = scrna_adata[:max_cells // 2, :].copy()
            logger.info(f"Subsampled scRNA data to {scrna_adata.shape[0]} cells")
        if vcc_adata.shape[0] > max_cells // 2:
            vcc_adata = vcc_adata[:max_cells // 2, :].copy()
            logger.info(f"Subsampled VCC data to {vcc_adata.shape[0]} cells")
    
    # Concatenate AnnData objects
    combined_adata = anndata.concat([scrna_adata, vcc_adata], join='inner', label='dataset', keys=['scrna', 'vcc'])
    logger.info(f"Combined dataset has {combined_adata.shape[0]:,} cells and {combined_adata.shape[1]} genes.")
    # Filter out cells with very low counts and genes with no expression
    sc.pp.filter_cells(combined_adata, min_counts=1)
    sc.pp.filter_genes(combined_adata, min_cells=3)
    logger.info(f"After filtering: {combined_adata.shape[0]} cells and {combined_adata.shape[1]} genes")

    # --- 3. Compute HVGs using scran ---
    hvg_list = compute_hvg_with_scran(combined_adata, n_hvgs=n_hvgs)

    # Save the list
    output_file = Path(output_path, f"hvg_scran_{len(hvg_list)}.txt")
    with open(output_file, 'w') as f:
        for gene in hvg_list:
            f.write(f"{gene}\n")
            
    logger.info(f"HVG list saved to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute HVGs using scran across scRNA and VCC data.")
    parser.add_argument("--scrna_dir", type=str, default="/Users/minjunes/vcc/data/scRNA_1e5/processed", help="Directory with scRNA AnnData file.")
    parser.add_argument("--vcc_path", type=str, default="/Users/minjunes/vcc/data/vcc_data/adata_Training.h5ad", help="Path to the VCC .h5ad file.")
    parser.add_argument("--output_path", type=str, default=".", help="Directory to save the HVG list.")
    parser.add_argument("--n_hvgs", type=int, default=2000, help="Number of HVGs to compute.")
    parser.add_argument("--max_cells", type=int, default=None, help="Maximum number of cells to use (for testing).")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    
    main(args.scrna_dir, args.vcc_path, args.output_path, args.n_hvgs, args.max_cells)